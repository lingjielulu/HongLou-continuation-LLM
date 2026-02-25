"""
QLoRA 训练主脚本
文档参考：README.md §2, §6

用法：
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --resume outputs/checkpoint-200
"""

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class ChunkDataset(Dataset):
    """从 JSONL 文件加载预切分的 token chunk"""

    def __init__(self, jsonl_path: str | Path, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.samples: list[list[int]] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ids  = item["input_ids"][:max_seq_len]
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ids = self.samples[idx]
        return {
            "input_ids":      torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels":         torch.tensor(ids, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """动态 padding 到 batch 内最长序列"""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list      = []
    attention_mask_list = []
    labels_list         = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        attention_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])  # -100 忽略 pad loss
        )

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels":         torch.stack(labels_list),
    }


# ─────────────────────────────────────────────────────────────
# 训练日志 Callback
# ─────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    "话说那日贾宝玉在怡红院中闲坐，忽见",
    "林黛玉独自一人倚在潇湘馆的竹林旁，泪珠儿不觉滚落，只见",
    "贾母笑道："
]


def _char_ngrams(text: str, n: int) -> Counter:
    return Counter(text[i: i + n] for i in range(len(text) - n + 1))


def _style_overlap(gen_text: str, ref_text: str, n: int) -> float:
    """生成文本的 n-gram 在参考语料中的覆盖率"""
    gen_ngrams = _char_ngrams(gen_text, n)
    ref_set    = set(ref_text[i: i + n] for i in range(len(ref_text) - n + 1))
    overlap = sum(v for k, v in gen_ngrams.items() if k in ref_set)
    total   = sum(gen_ngrams.values())
    return overlap / total if total > 0 else 0.0


class StyleLogCallback(TrainerCallback):
    """
    每次 eval 结束后：
    1. 用 3 个固定 prompt 生成样本
    2. 计算 2/3/4-gram StyleOverlap（生成文本 vs 训练语料）
    3. 将 PPL + n-gram 写入 TRAINING_LOG.md 的表格行
    4. 将生成样本节选追加到日志
    5. 将完整 eval report 保存为 JSON
    """

    def __init__(self, tokenizer, log_file: Path, reports_dir: Path,
                 corpus_path: Path):
        self.tokenizer   = tokenizer
        self.log_file    = log_file
        self.reports_dir = reports_dir
        self._eval_count = 0
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 加载训练语料用于 n-gram 计算（字符级）
        if corpus_path.exists():
            self.corpus = corpus_path.read_text(encoding="utf-8")
            print(f"[StyleLog] 已加载语料：{corpus_path.name}（{len(self.corpus):,} 字符）")
        else:
            self.corpus = ""
            print(f"[StyleLog] ⚠ 语料文件不存在：{corpus_path}，n-gram 将跳过")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl,
                    metrics: dict, model=None, **kwargs):
        self._eval_count += 1
        step      = state.global_step
        eval_loss = metrics.get("eval_loss", float("nan"))
        ppl       = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")

        print(f"\n[StyleLog] step={step}  eval_loss={eval_loss:.4f}  PPL={ppl:.4f}")

        report = {"step": step, "eval_loss": eval_loss, "ppl": round(ppl, 4)}

        # ── 每次 eval 都生成样本 ──
        samples    = []
        ngram_info = {}
        if model is not None:
            gen_device = next(model.parameters()).device
            model.eval()
            for prompt in EVAL_PROMPTS:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(gen_device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.8, top_p=0.9,
                        repetition_penalty=1.1, do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                gen = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                samples.append({"prompt": prompt, "generated": gen})

            # ── 计算 n-gram StyleOverlap ──
            if self.corpus:
                all_gen = "".join(s["generated"] for s in samples)
                for n in [2, 3, 4]:
                    ov = _style_overlap(all_gen, self.corpus, n)
                    ngram_info[f"{n}gram"] = round(ov, 4)
                    print(f"[StyleLog]   {n}-gram overlap: {ov:.4f}")
                report["ngram"] = ngram_info

            report["samples"] = samples

        # ── 保存 JSON ──
        rpt_path = self.reports_dir / f"step_{step:06d}.json"
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # ── 更新 TRAINING_LOG.md ──
        self._write_to_log(step, ppl, eval_loss, ngram_info, samples)

    def _write_to_log(self, step: int, ppl: float, nll: float,
                      ngram_info: dict, samples: list):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        g2 = f"{ngram_info.get('2gram', '-')}"
        g3 = f"{ngram_info.get('3gram', '-')}"
        g4 = f"{ngram_info.get('4gram', '-')}"
        table_line = f"| {step} | 训练中 | **{ppl:.4f}** | {nll:.4f} | {g2} | {g3} | {g4} | {now} |"
        self._update_ppl_table(table_line)

        if not samples:
            return

        block = f"\n## Step {step} — 训练中评估\n\n"
        block += f"**时间**：{now} | **PPL**：{ppl:.4f} | **NLL**：{nll:.4f}"
        if ngram_info:
            block += f" | **2-gram**：{ngram_info.get('2gram','-')} | **3-gram**：{ngram_info.get('3gram','-')} | **4-gram**：{ngram_info.get('4gram','-')}"
        block += "\n\n### 生成样本节选\n\n"
        for s in samples:
            block += f"**Prompt**：{s['prompt']}\n\n"
            block += f"> {s['generated'][:300].replace(chr(10), ' ')}\n\n"
        block += "---\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(block)

    def _update_ppl_table(self, new_row: str):
        """在 TRAINING_LOG.md 的 PPL 表格末尾追加一行"""
        content = self.log_file.read_text(encoding="utf-8")
        lines   = content.split("\n")

        last_table_idx = None
        in_ppl_section = False
        for i, line in enumerate(lines):
            if "## PPL 变化曲线" in line:
                in_ppl_section = True
            if in_ppl_section and line.startswith("| ") and "|---" not in line and "Step" not in line:
                last_table_idx = i

        if last_table_idx is not None:
            lines.insert(last_table_idx + 1, new_row)
            self.log_file.write_text("\n".join(lines), encoding="utf-8")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"\n## 训练完成\n\n**时间**：{now}\n**总步数**：{state.global_step}\n\n---\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(block)


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────
def _replace_fp8_recursive(module, seen_ids: set) -> int:
    """
    递归替换模块树中的 FP8Linear。
    使用递归而非 list(model.modules()) 快照，确保旧模块可被 GC 释放。
    """
    from transformers.integrations import FP8Linear
    import torch.nn as nn

    replaced = 0
    for child_name in list(module._modules.keys()):
        child = module._modules[child_name]
        if child is None:
            continue

        if isinstance(child, FP8Linear) and id(child) not in seen_ids:
            seen_ids.add(id(child))
            dev    = child.weight.device
            out_f, in_f = child.weight.shape

            # 在 CPU 上做反量化，避免 GPU float32 中间张量 OOM
            w_cpu  = child.weight.data.to("cpu").to(torch.float32)
            si_cpu = child.weight_scale_inv.data.to("cpu").float()

            if child.block_size is None:
                w_bf16 = (w_cpu * si_cpu.item()).to(torch.bfloat16)
            else:
                bh, bw = child.block_size
                n_out = (out_f + bh - 1) // bh
                n_in  = (in_f  + bw - 1) // bw
                if n_out * bh != out_f or n_in * bw != in_f:
                    w_cpu = torch.nn.functional.pad(
                        w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f)
                    )
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)

            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None, dtype=torch.bfloat16)
            new_lin.weight.data = w_bf16
            if child.bias is not None:
                new_lin.bias.data = child.bias.data.to("cpu").to(torch.bfloat16)

            # 替换并立即释放旧 FP8 模块（递归时不持有对 child 的额外引用）
            del child, w_cpu, si_cpu, w_bf16
            setattr(module, child_name, new_lin)
            new_lin = new_lin.to(dev)
            setattr(module, child_name, new_lin)
            torch.cuda.empty_cache()
            replaced += 1
        else:
            # 继续递归子树
            replaced += _replace_fp8_recursive(child, seen_ids)

    return replaced


def _dequantize_fp8_to_bf16(model) -> int:
    """
    将模型中所有 FP8Linear 正确反量化为标准 BF16 nn.Linear。
    在 CPU 上计算，逐层替换，及时释放 GPU FP8 内存。
    """
    seen_ids: set = set()
    return _replace_fp8_recursive(model, seen_ids)


def load_fp8_model(model_cfg: dict, lora_cfg: dict) -> tuple:
    """
    加载 Qwen3-8B-FP8，将所有 FP8Linear 正确反量化为 BF16，再挂载 LoRA 进行训练。

    FP8Linear 使用自定义 CUDA kernel（w8a8_block_fp8_matmul），不支持 PyTorch autograd。
    必须先反量化到 BF16，才能进行标准 LoRA 反向传播。
    VRAM 消耗约 16GB（BF16 权重）+ LoRA + 梯度检查点激活值 ≈ 19GB，RTX 4090 可容纳。
    """
    # bypass Trainer 对 FP8 训练的拦截检查（基础权重转为 BF16 后仍携带 FP8 quantization_method 标记）
    import transformers.trainer as _trainer_module
    _trainer_module.validate_quantization_for_training = lambda model: None

    # 优先使用本地缓存路径，避免重复下载
    local_path = model_cfg.get("local_model_path")
    model_id   = local_path if (local_path and Path(local_path).exists()) \
                 else model_cfg["base_model_id"]

    print(f"加载 FP8 模型：{model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FP8 → BF16 反量化（block-wise）：替换所有 FP8Linear 为标准 nn.Linear
    n = _dequantize_fp8_to_bf16(model)
    print(f"[FP8→BF16] 已将 {n} 个 FP8Linear 反量化并替换为 BF16 Linear")

    # 启用梯度检查点（重计算激活值，节省 ~8GB 显存）
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 挂载 LoRA（适配器为 BF16，基础权重 BF16 冻结）
    lora_config = LoraConfig(
        r=lora_cfg["lora"]["r"],
        lora_alpha=lora_cfg["lora"]["lora_alpha"],
        lora_dropout=lora_cfg["lora"]["lora_dropout"],
        bias=lora_cfg["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────
def train(config_path: str, resume_from: str | None = None):
    ROOT = Path(__file__).parent.parent

    # 加载配置
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lora_config_path = ROOT / "configs" / "lora_config.yaml"
    with open(lora_config_path, encoding="utf-8") as f:
        lora_cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]

    # 加载模型
    model, tokenizer = load_fp8_model(cfg["model"], lora_cfg)

    # 加载数据集
    print("加载数据集...")
    train_dataset = ChunkDataset(
        ROOT / cfg["data"]["train_file"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )
    val_dataset = ChunkDataset(
        ROOT / cfg["data"]["val_file"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )
    print(f"  训练集：{len(train_dataset):,} samples")
    print(f"  验证集：{len(val_dataset):,} samples")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(ROOT / train_cfg["output_dir"]),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        tf32=train_cfg.get("tf32", True),
        optim=train_cfg["optim"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        save_strategy=cfg["checkpointing"]["save_strategy"],
        save_steps=cfg["checkpointing"]["save_steps"],
        save_total_limit=cfg["checkpointing"]["save_total_limit"],
        load_best_model_at_end=cfg["checkpointing"]["load_best_model_at_end"],
        metric_for_best_model=cfg["checkpointing"]["metric_for_best_model"],
        greater_is_better=cfg["checkpointing"]["greater_is_better"],
        eval_strategy=cfg["evaluation"]["eval_strategy"],
        eval_steps=cfg["evaluation"]["eval_steps"],
        logging_steps=cfg["logging"]["logging_steps"],
        report_to=cfg["logging"]["report_to"],
        logging_dir=str(ROOT / cfg["logging"]["logging_dir"]),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # 日志 Callback
    log_callback = StyleLogCallback(
        tokenizer=tokenizer,
        log_file=ROOT / "TRAINING_LOG.md",
        reports_dir=ROOT / "outputs" / "eval_reports",
        corpus_path=ROOT / "data" / "processed" / "honglou_80_cleaned.txt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        callbacks=[log_callback],
    )

    # 训练
    print("\n开始训练...")
    trainer.train(resume_from_checkpoint=resume_from)

    # 保存最终 LoRA 权重
    best_path = ROOT / "outputs" / "checkpoint-best"
    model.save_pretrained(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    print(f"\n训练完成，最佳模型保存至：{best_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="红楼梦 QLoRA 训练")
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="从 checkpoint 恢复训练",
    )
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
