"""
评估脚本：PPL + N-gram 风格评估
文档参考：README.md §7

用法：
    # 仅计算 PPL
    python scripts/evaluate.py --checkpoint outputs/checkpoint-best --mode ppl

    # 仅计算 n-gram 风格重叠
    python scripts/evaluate.py --checkpoint outputs/checkpoint-best --mode ngram

    # 完整评估
    python scripts/evaluate.py --checkpoint outputs/checkpoint-best --mode all
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────
# PPL 计算
# ─────────────────────────────────────────────────────────────
def compute_ppl(model, tokenizer, val_jsonl: Path, device: str,
                batch_size: int = 4, max_seq_len: int = 2048) -> dict:
    """
    计算验证集 PPL。

    实现细节：
    - 每个 chunk 作为独立序列，计算 token 级平均 NLL
    - 加权平均（权重为 chunk token 数量）再转换为 PPL
    - pad token 位置不计入 loss（labels=-100）
    """
    from torch.utils.data import DataLoader

    model.eval()
    total_nll    = 0.0
    total_tokens = 0

    samples = [json.loads(l) for l in val_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"  验证集 chunk 数：{len(samples)}")

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i: i + batch_size]

            # 截断 + pad 到 batch 内最长
            batch_ids = [s["input_ids"][:max_seq_len] for s in batch_samples]
            max_len   = max(len(ids) for ids in batch_ids)

            input_ids_list = []
            labels_list    = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded  = ids + [tokenizer.pad_token_id] * pad_len
                labels  = ids + [-100] * pad_len
                input_ids_list.append(padded)
                labels_list.append(labels)

            input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
            labels    = torch.tensor(labels_list,    dtype=torch.long).to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            # outputs.loss = 所有非-100位置的平均 NLL
            n_tokens      = (labels != -100).sum().item()
            total_nll    += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

            if (i // batch_size) % 10 == 0:
                print(f"    [{i}/{len(samples)}] running NLL={total_nll/total_tokens:.4f}")

    avg_nll = total_nll / total_tokens
    ppl     = math.exp(avg_nll)
    return {"avg_nll": avg_nll, "ppl": ppl, "total_tokens": total_tokens}


# ─────────────────────────────────────────────────────────────
# N-gram 风格评估
# ─────────────────────────────────────────────────────────────
def char_ngrams(text: str, n: int) -> Counter:
    """字符级 n-gram 计数"""
    return Counter(text[i: i + n] for i in range(len(text) - n + 1))


def style_overlap(gen_text: str, ref_text: str, n: int) -> float:
    """
    StyleOverlap_n = |ngram_n(gen) ∩ ngram_n(ref)| / |ngram_n(gen)|
    计算生成文本的 n-gram 在参考语料中的覆盖率
    """
    gen_ngrams = char_ngrams(gen_text, n)
    ref_set    = set(ref_text[i: i + n] for i in range(len(ref_text) - n + 1))

    overlap = sum(v for k, v in gen_ngrams.items() if k in ref_set)
    total   = sum(gen_ngrams.values())
    return overlap / total if total > 0 else 0.0


def generate_texts(model, tokenizer, prompts: list[str], device: str,
                   num_return: int = 3, max_new_tokens: int = 512) -> list[str]:
    """对每个 prompt 生成 num_return 条文本"""
    model.eval()
    all_texts = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for _ in range(num_return):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_texts.append(gen_text)

    return all_texts


def compute_ngram_metrics(gen_texts: list[str], train_corpus: str) -> dict:
    """计算 2/3/4-gram StyleOverlap"""
    all_gen = "".join(gen_texts)
    result  = {}
    for n in [2, 3, 4]:
        result[f"{n}gram_overlap"] = style_overlap(all_gen, train_corpus, n)
    return result


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────
LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)


def load_model(checkpoint_path: str, base_model_id: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    peft_cfg = PeftConfig.from_pretrained(checkpoint_path)
    if base_model_id is None:
        # 优先使用本地 FP8 缓存
        base_model_id = (
            LOCAL_FP8_PATH
            if Path(LOCAL_FP8_PATH).exists()
            else peft_cfg.base_model_name_or_path
        )

    print(f"加载 FP8 基座模型：{base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype="auto",     # 自动识别 FP8 权重
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载 LoRA 权重：{checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# 主评估流程
# ─────────────────────────────────────────────────────────────
def run_evaluation(checkpoint: str, mode: str, base_model_id: str | None = None,
                   step: int | None = None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(checkpoint, base_model_id)

    report = {}

    # ── PPL ──
    if mode in ("ppl", "all"):
        print("\n[PPL 评估]")
        val_jsonl = ROOT / "data" / "processed" / "val.jsonl"
        assert val_jsonl.exists(), f"验证集文件不存在：{val_jsonl}"
        ppl_result = compute_ppl(model, tokenizer, val_jsonl, device)
        report["ppl"]         = ppl_result["ppl"]
        report["avg_nll"]     = ppl_result["avg_nll"]
        report["val_tokens"]  = ppl_result["total_tokens"]
        print(f"  Val PPL:  {ppl_result['ppl']:.4f}")
        print(f"  Avg NLL:  {ppl_result['avg_nll']:.4f}")

    # ── N-gram ──
    if mode in ("ngram", "all"):
        print("\n[N-gram 风格评估]")

        prompts_file = ROOT / "configs" / "eval_prompts.txt"
        prompts = [
            line.strip()
            for line in prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"  使用 {len(prompts)} 个固定 prompt")

        gen_texts = generate_texts(model, tokenizer, prompts, device,
                                   num_return=3, max_new_tokens=512)
        print(f"  生成 {len(gen_texts)} 条文本")

        # 加载训练语料（字符级，不需要 token）
        train_corpus_path = ROOT / "data" / "processed" / "honglou_80_cleaned.txt"
        train_corpus = train_corpus_path.read_text(encoding="utf-8")

        ngram_result = compute_ngram_metrics(gen_texts, train_corpus)
        report.update(ngram_result)

        for n in [2, 3, 4]:
            print(f"  {n}-gram StyleOverlap: {ngram_result[f'{n}gram_overlap']:.4f}")

        # 保存生成样本
        samples_out = ROOT / "outputs" / "eval_reports" / f"samples_step{step or 'final'}.txt"
        samples_out.parent.mkdir(parents=True, exist_ok=True)
        with open(samples_out, "w", encoding="utf-8") as f:
            for i, (prompt, text) in enumerate(zip(prompts * 3, gen_texts)):
                f.write(f"=== 样本 {i+1} ===\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated:\n{text}\n\n")
        print(f"  生成样本已保存：{samples_out}")

    # ── 保存报告 ──
    report_name = f"step_{step:06d}.json" if step else "final.json"
    report_path = ROOT / "outputs" / "eval_reports" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评估报告已保存：{report_path}")
    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="红楼梦 LoRA 评估")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint 路径")
    parser.add_argument(
        "--mode",
        choices=["ppl", "ngram", "all"],
        default="all",
        help="评估模式（默认：all）",
    )
    parser.add_argument("--base_model", default=None, help="基座模型 ID（可选）")
    parser.add_argument("--step", type=int, default=None, help="当前训练步数（用于命名报告）")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.mode, args.base_model, args.step)


if __name__ == "__main__":
    main()
