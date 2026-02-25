"""
Step 0 基线评估：加载原始 Qwen3-8B-FP8（FP8→BF16 反量化，不加 LoRA），
计算验证集 PPL + 生成样本 + n-gram StyleOverlap，
保存 outputs/eval_reports/step_000000.json 并更新 TRAINING_LOG.md。

用法：
    conda run -n stone python3 scripts/baseline_eval.py --config configs/training_config.yaml
"""

import argparse
import json
import math
import torch
import torch.nn as nn
import yaml
from collections import Counter
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

EVAL_PROMPTS = [
    "话说那日贾宝玉在怡红院中闲坐，忽见",
    "林黛玉独自一人倚在潇湘馆的竹林旁，泪珠儿不觉滚落，只见",
    "贾母笑道：",
]


# ── Dataset ──────────────────────────────────────────────────
class ChunkDataset(Dataset):
    def __init__(self, jsonl_path, max_seq_len=2048):
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = json.loads(line)["input_ids"][:max_seq_len]
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return {
            "input_ids":      torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels":         torch.tensor(ids, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list, mask_list, labels_list = [], [], []
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids_list.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)]))
        mask_list.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"], torch.full((pad_len,), -100)]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels":         torch.stack(labels_list),
    }


# ── N-gram ────────────────────────────────────────────────────
def _char_ngrams(text, n):
    return Counter(text[i: i + n] for i in range(len(text) - n + 1))


def _style_overlap(gen_text, ref_text, n):
    gen_ngrams = _char_ngrams(gen_text, n)
    ref_set    = set(ref_text[i: i + n] for i in range(len(ref_text) - n + 1))
    overlap = sum(v for k, v in gen_ngrams.items() if k in ref_set)
    total   = sum(gen_ngrams.values())
    return overlap / total if total > 0 else 0.0


# ── FP8 → BF16 dequantization ────────────────────────────────
def _replace_fp8_recursive(module, seen_ids):
    from transformers.integrations import FP8Linear
    replaced = 0
    for child_name in list(module._modules.keys()):
        child = module._modules[child_name]
        if child is None:
            continue
        if isinstance(child, FP8Linear) and id(child) not in seen_ids:
            seen_ids.add(id(child))
            dev = child.weight.device
            out_f, in_f = child.weight.shape
            w_cpu  = child.weight.data.to("cpu").to(torch.float32)
            si_cpu = child.weight_scale_inv.data.to("cpu").float()
            if child.block_size is None:
                w_bf16 = (w_cpu * si_cpu.item()).to(torch.bfloat16)
            else:
                bh, bw = child.block_size
                n_out = (out_f + bh - 1) // bh
                n_in  = (in_f  + bw - 1) // bw
                if n_out * bh != out_f or n_in * bw != in_f:
                    w_cpu = nn.functional.pad(w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f))
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)
            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None, dtype=torch.bfloat16)
            new_lin.weight.data = w_bf16
            if child.bias is not None:
                new_lin.bias.data = child.bias.data.to("cpu").to(torch.bfloat16)
            del child, w_cpu, si_cpu, w_bf16
            setattr(module, child_name, new_lin)
            new_lin = new_lin.to(dev)
            setattr(module, child_name, new_lin)
            torch.cuda.empty_cache()
            replaced += 1
        else:
            replaced += _replace_fp8_recursive(child, seen_ids)
    return replaced


# ── PPL evaluation ───────────────────────────────────────────
@torch.no_grad()
def evaluate_ppl(model, val_jsonl, max_seq_len, pad_token_id):
    dataset    = ChunkDataset(val_jsonl, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_token_id))
    device = next(model.parameters()).device
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss_val = out.loss.item()
        if not math.isnan(loss_val):
            total_loss += loss_val
            n_batches  += 1
        if i % 20 == 0:
            print(f"  [{i}/{len(dataset)}] running avg_loss={total_loss/max(n_batches,1):.4f}")
    avg_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    return avg_loss, math.exp(avg_loss)


# ── main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg       = yaml.safe_load(open(args.config, encoding="utf-8"))
    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]

    reports_dir = ROOT / "outputs" / "eval_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_file    = ROOT / "TRAINING_LOG.md"
    corpus_path = ROOT / "data" / "processed" / "honglou_80_cleaned.txt"

    # ── 加载模型 ──
    local_path = model_cfg.get("local_model_path")
    model_id   = local_path if (local_path and Path(local_path).exists()) else model_cfg["base_model_id"]
    print(f"加载基线模型：{model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n = _replace_fp8_recursive(model, set())
    print(f"[FP8→BF16] 已将 {n} 个 FP8Linear 反量化并替换为 BF16 Linear（无 LoRA）")

    # ── 评估 PPL ──
    val_jsonl = ROOT / data_cfg["val_file"]
    print("计算验证集 PPL ...")
    eval_loss, ppl = evaluate_ppl(model, val_jsonl, data_cfg["max_seq_len"],
                                  tokenizer.pad_token_id or 0)
    print(f"\n[Baseline] eval_loss={eval_loss:.4f}  PPL={ppl:.4f}")

    # ── 生成样本 ──
    device = next(model.parameters()).device
    model.eval()
    samples = []
    for prompt in EVAL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200,
                temperature=0.8, top_p=0.9,
                repetition_penalty=1.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        samples.append({"prompt": prompt, "generated": gen})
        print(f"  [{prompt[:15]}...] → {gen[:60]}...")

    # ── N-gram ──
    ngram_info = {}
    if corpus_path.exists():
        corpus  = corpus_path.read_text(encoding="utf-8")
        all_gen = "".join(s["generated"] for s in samples)
        for n in [2, 3, 4]:
            ov = _style_overlap(all_gen, corpus, n)
            ngram_info[f"{n}gram"] = round(ov, 4)
            print(f"[Baseline]   {n}-gram overlap: {ov:.4f}")

    # ── 保存 JSON ──
    report = {
        "step": 0, "eval_loss": eval_loss, "ppl": round(ppl, 4),
        "ngram": ngram_info, "samples": samples,
    }
    out_path = reports_dir / "step_000000.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{out_path}")

    # ── 更新 TRAINING_LOG.md ──
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    g2 = ngram_info.get("2gram", "-")
    g3 = ngram_info.get("3gram", "-")
    g4 = ngram_info.get("4gram", "-")
    table_line = f"| 0 | 基线（无微调）| **{ppl:.4f}** | {eval_loss:.4f} | {g2} | {g3} | {g4} | {now} |"

    content = log_file.read_text(encoding="utf-8")
    # 替换占位行
    content = content.replace(
        "| 0 | 基线（无微调）| — | — | — | — | — | 待填充 |",
        table_line,
    )

    # 生成样本块，插在 Step 25 之前
    block  = f"\n## Step 0 — 基线（无微调）\n\n"
    block += f"**时间**：{now} | **PPL**：{ppl:.4f} | **NLL**：{eval_loss:.4f}"
    if ngram_info:
        block += f" | **2-gram**：{g2} | **3-gram**：{g3} | **4-gram**：{g4}"
    block += "\n\n### 生成样本节选\n\n"
    for s in samples:
        block += f"**Prompt**：{s['prompt']}\n\n"
        block += f"> {s['generated'][:300].replace(chr(10), ' ')}\n\n"
    block += "---\n"

    insert_marker = "## Step 25 — 训练中评估"
    if insert_marker in content:
        content = content.replace(insert_marker, block + insert_marker)
    else:
        content += block

    log_file.write_text(content, encoding="utf-8")
    print("TRAINING_LOG.md 已更新")
    print(f"\n基线 PPL = {ppl:.4f}")


if __name__ == "__main__":
    main()
