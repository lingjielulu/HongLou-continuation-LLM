"""
Instruct 格式 QLoRA 训练脚本
训练目标：给定章节大纲 → 生成对应章节正文（label masking，只计算 assistant 回答的 loss）

与 train.py 的核心区别：
  - 数据集带有预设 labels（-100 屏蔽 prompt，只对 assistant 部分计算 loss）
  - 验证时使用 chat 格式生成，而非 raw causal LM
  - 输出到 outputs/instruct_lora/

用法：
    conda run -n stone python3 scripts/train_instruct.py
    conda run -n stone python3 scripts/train_instruct.py --epochs 5 --lr 1e-4
    conda run -n stone python3 scripts/train_instruct.py --resume outputs/instruct_lora/checkpoint-50
"""

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, TaskType, get_peft_model


ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTLINE_DIR   = ROOT / "outline" / "前80回大纲"

LOCAL_MODEL_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

# 评估时用于生成样本的示例大纲（取自原著第5回）
_EVAL_OUTLINE = {
    "title": "第五回　游幻境指迷十二钗，饮仙醪曲演红楼梦",
    "核心情节": [
        "贾宝玉随贾母往宁府赏梅，在秦氏房中小憩，入梦至太虚幻境",
        "警幻仙姑引宝玉观金陵十二钗正副册，暗示众女儿命运",
        "宝玉聆听《红楼梦》十二支曲，隐喻各人结局",
        "警幻以情欲之事警示宝玉，令其与可卿成婚，梦中悟道未成",
    ],
    "主要人物": "贾宝玉、警幻仙姑、秦可卿（梦中）",
    "关键场景": "秦氏内室、太虚幻境薄命司",
    "情感基调": "虚幻迷离，哀婉悠远，盛极之乐中潜藏末世之悲",
    "叙事功能": "全书命运总纲，以判词与曲词预示十二钗结局，奠定悲剧基调",
}

SYSTEM_PROMPT = """你是一位专业的《红楼梦》续写作家，深谙曹雪芹的文风与叙事技法。
【写作要求】
1. 语言风格：文言夹白话的章回体小说文风，典雅而不晦涩
2. 叙事手法：工笔与写意并重，人物对话符合各自身份性格
3. 情节衔接：与上回结尾自然衔接，严格依照大纲展开情节
4. 体量控制：每回约 2000-3000 字
5. 禁止出现任何现代词汇、网络用语
6. 以"话说""却说"等章回体起首句开篇
7. 结尾以"正是：[对仗诗句]\\n欲知后事如何，且听下回分解。"收束"""


# ─────────────────────────────────────────────────────────────
# Dataset：加载预构建的 instruct 格式 JSONL（含 label masking）
# ─────────────────────────────────────────────────────────────
class InstructDataset(Dataset):
    """
    加载 build_instruct_data.py 生成的 JSONL 文件。
    每条样本已包含：
      - input_ids: 完整对话 token ids
      - labels:    -100 屏蔽 prompt，只对 assistant 计算 loss
    """

    def __init__(self, jsonl_path: str | Path, max_seq_len: int = 4096):
        self.samples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ids    = item["input_ids"][:max_seq_len]
                labels = item["labels"][:max_seq_len]
                self.samples.append({
                    "input_ids": ids,
                    "labels":    labels,
                    "chapter":   item.get("chapter", 0),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(len(item["input_ids"]), dtype=torch.long),
            "labels":         torch.tensor(item["labels"],    dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """动态 padding 到 batch 内最长序列"""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list, attn_mask_list, labels_list = [], [], []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        attn_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
        "labels":         torch.stack(labels_list),
    }


# ─────────────────────────────────────────────────────────────
# 评估 Callback（instruct 版本）
# ─────────────────────────────────────────────────────────────
def _char_ngrams(text: str, n: int) -> Counter:
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _style_overlap(gen_text: str, ref_text: str, n: int) -> float:
    gen_ngrams = _char_ngrams(gen_text, n)
    ref_set    = set(ref_text[i:i + n] for i in range(len(ref_text) - n + 1))
    overlap = sum(v for k, v in gen_ngrams.items() if k in ref_set)
    total   = sum(gen_ngrams.values())
    return overlap / total if total > 0 else 0.0


def _outline_to_user_content(outline: dict) -> str:
    title = outline.get("title", "")
    plot_items = outline.get("核心情节", [])
    if isinstance(plot_items, list):
        plot_str = "\n".join(f"- {p}" for p in plot_items)
    else:
        plot_str = str(plot_items)

    parts = [f"## {title}", f"\n**核心情节：**\n{plot_str}"]
    for field in ["主要人物", "关键场景", "情感基调", "叙事功能"]:
        val = outline.get(field, "")
        if val:
            parts.append(f"\n**{field}：** {val}")
    parts.append("\n请根据以上大纲续写本回正文。")
    return "\n".join(parts)


class InstructStyleLogCallback(TrainerCallback):
    """
    每次 eval 结束后：
    1. 用示例大纲生成续写样本（chat 格式）
    2. 计算 n-gram StyleOverlap
    3. 写入 TRAINING_LOG_INSTRUCT.md
    4. 保存 JSON report
    """

    def __init__(self, tokenizer, log_file: Path, reports_dir: Path, corpus_path: Path):
        self.tokenizer   = tokenizer
        self.log_file    = log_file
        self.reports_dir = reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        if corpus_path.exists():
            self.corpus = corpus_path.read_text(encoding="utf-8")
            print(f"[StyleLog] 已加载语料：{corpus_path.name}（{len(self.corpus):,} 字）")
        else:
            self.corpus = ""

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl,
                    metrics: dict, model=None, **kwargs):
        step      = state.global_step
        eval_loss = metrics.get("eval_loss", float("nan"))
        ppl       = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")
        print(f"\n[StyleLog] step={step}  eval_loss={eval_loss:.4f}  PPL={ppl:.4f}")

        report     = {"step": step, "eval_loss": eval_loss, "ppl": round(ppl, 4)}
        samples    = []
        ngram_info = {}

        if model is not None:
            gen_device = next(model.parameters()).device
            model.eval()

            # 构建 chat 格式 prompt
            user_content = _outline_to_user_content(_EVAL_OUTLINE)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer(
                [prompt_text], return_tensors="pt"
            ).to(gen_device)

            with torch.no_grad():
                out_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_ids = out_ids[0][model_inputs.input_ids.shape[1]:]
            gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            samples.append({"outline_title": _EVAL_OUTLINE["title"], "generated": gen_text})

            if self.corpus:
                for n in [2, 3, 4]:
                    ov = _style_overlap(gen_text, self.corpus, n)
                    ngram_info[f"{n}gram"] = round(ov, 4)
                    print(f"[StyleLog]   {n}-gram overlap: {ov:.4f}")
                report["ngram"] = ngram_info

            report["samples"] = samples

        # 保存 JSON
        rpt_path = self.reports_dir / f"step_{step:06d}.json"
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._write_to_log(step, ppl, eval_loss, ngram_info, samples)

    def _write_to_log(self, step, ppl, nll, ngram_info, samples):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        g2  = ngram_info.get("2gram", "-")
        g3  = ngram_info.get("3gram", "-")
        g4  = ngram_info.get("4gram", "-")

        block = f"\n## Step {step} — {now}\n\n"
        block += f"**PPL**：{ppl:.4f} | **NLL**：{nll:.4f}"
        if ngram_info:
            block += f" | **2-gram**：{g2} | **3-gram**：{g3} | **4-gram**：{g4}"
        block += "\n\n"

        if samples:
            s = samples[0]
            block += f"**大纲**：{s['outline_title']}\n\n"
            block += f"**生成（前400字）**：\n> {s['generated'][:400].replace(chr(10), ' ')}\n\n"
        block += "---\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(block)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## 训练完成\n\n**时间**：{now}  **总步数**：{state.global_step}\n\n---\n")


# ─────────────────────────────────────────────────────────────
# FP8 → BF16 反量化（复用自 train.py）
# ─────────────────────────────────────────────────────────────
def _replace_fp8_recursive(module, seen_ids: set) -> int:
    try:
        from transformers.integrations import FP8Linear
    except ImportError:
        return 0

    replaced = 0
    for child_name in list(module._modules.keys()):
        child = module._modules[child_name]
        if child is None:
            continue

        if isinstance(child, FP8Linear) and id(child) not in seen_ids:
            seen_ids.add(id(child))
            dev         = child.weight.device
            out_f, in_f = child.weight.shape

            w_cpu  = child.weight.data.to("cpu").to(torch.float32)
            si_cpu = child.weight_scale_inv.data.to("cpu").float()

            if child.block_size is None:
                w_bf16 = (w_cpu * si_cpu.item()).to(torch.bfloat16)
            else:
                bh, bw = child.block_size
                n_out  = (out_f + bh - 1) // bh
                n_in   = (in_f  + bw - 1) // bw
                if n_out * bh != out_f or n_in * bw != in_f:
                    import torch.nn.functional as F
                    w_cpu = F.pad(w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f))
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)

            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None, dtype=torch.bfloat16)
            new_lin.weight.data = w_bf16
            if child.bias is not None:
                new_lin.bias.data = child.bias.data.to("cpu").to(torch.bfloat16)

            del child, w_cpu, si_cpu, w_bf16
            setattr(module, child_name, new_lin.to(dev))
            torch.cuda.empty_cache()
            replaced += 1
        else:
            replaced += _replace_fp8_recursive(child, seen_ids)

    return replaced


def load_model(model_id: str) -> tuple:
    import transformers.trainer as _trainer_module
    _trainer_module.validate_quantization_for_training = lambda model: None

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

    n = _replace_fp8_recursive(model, set())
    print(f"[FP8→BF16] 已将 {n} 个 FP8Linear 反量化为 BF16")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Memory-efficient Trainer：分 chunk 计算 CE loss，避免 logits.float() OOM
# ─────────────────────────────────────────────────────────────
class MemoryEfficientTrainer(Trainer):
    """
    覆盖 compute_loss，将 (seq_len × vocab_size) 的 logits 分块转 float32，
    每次只处理 CHUNK_SIZE=512 行（512×151643×4bytes≈310MB），而非一次性 2.5GB。
    """
    CHUNK_SIZE = 512

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"},
                        labels=None)  # 不让模型内部计算 loss
        logits = outputs.logits  # BF16: (batch, seq, vocab)

        # Causal LM shift
        shift_logits = logits[..., :-1, :].contiguous()   # (B, S-1, V)
        shift_labels = labels[..., 1:].contiguous()        # (B, S-1)

        flat_logits = shift_logits.view(-1, shift_logits.shape[-1])  # (B*(S-1), V)
        flat_labels = shift_labels.view(-1)                           # (B*(S-1),)

        # 分 chunk 计算 CE loss（保留梯度图）
        # 先统计有效 token 数
        total_tokens = int((flat_labels != -100).sum().item())
        if total_tokens == 0:
            loss = flat_logits.sum() * 0  # dummy loss with grad
            return (loss, outputs) if return_outputs else loss

        loss = flat_logits.new_zeros(())  # scalar, dtype=bfloat16, on same device
        for i in range(0, flat_logits.shape[0], self.CHUNK_SIZE):
            chunk_logits = flat_logits[i:i + self.CHUNK_SIZE].float()  # 仅本 chunk 转 float32
            chunk_labels = flat_labels[i:i + self.CHUNK_SIZE]
            valid = (chunk_labels != -100)
            if not valid.any():
                del chunk_logits
                continue
            chunk_loss = F.cross_entropy(
                chunk_logits, chunk_labels, ignore_index=-100, reduction="sum"
            )
            loss = loss + chunk_loss / total_tokens  # 保留 grad_fn
            del chunk_logits, chunk_loss

        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────
# 主训练流程
# ─────────────────────────────────────────────────────────────
def _write_training_info(
    out_dir: Path,
    *,
    run_id: str,
    epochs: int,
    lr: float,
    grad_accum: int,
    max_seq_len: int,
    train_path: Path,
    val_path: Path,
    train_n: int,
    val_n: int,
    steps_per_epoch: int,
    total_steps: int,
    base_model: str,
):
    """在输出目录内写一份人读版训练说明 training_info.md"""
    sw = "sw" in train_path.name
    data_mode = "滑窗分片（sliding window）" if sw else "截断（truncation）"
    lines = [
        f"# LoRA 训练说明",
        f"",
        f"**Run ID**：`{run_id}`",
        f"**开始时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## 基座模型",
        f"",
        f"- 路径：`{base_model}`",
        f"- 量化：FP8（训练前反量化为 BF16）",
        f"",
        f"## 训练数据",
        f"",
        f"- 数据模式：{data_mode}",
        f"- 训练集：`{train_path.name}`  ({train_n} 条样本，第 1-72 回）",
        f"- 验证集：`{val_path.name}`  ({val_n} 条样本，第 73-80 回）",
        f"- 大纲来源：`outline/前80回大纲/`（由 generate_outlines.py 生成）",
        f"",
        f"## 超参数",
        f"",
        f"| 参数 | 值 |",
        f"|------|----|",
        f"| epochs | {epochs} |",
        f"| learning_rate | {lr} |",
        f"| grad_accumulation | {grad_accum} |",
        f"| effective_batch_size | {grad_accum} |",
        f"| max_seq_len | {max_seq_len} |",
        f"| steps_per_epoch | {steps_per_epoch} |",
        f"| total_steps | {total_steps} |",
        f"| lr_scheduler | cosine |",
        f"| warmup_ratio | 0.1 |",
        f"| optimizer | adamw_torch |",
        f"| bf16 | True |",
        f"",
        f"## LoRA 配置",
        f"",
        f"| 参数 | 值 |",
        f"|------|----|",
        f"| r | 16 |",
        f"| lora_alpha | 32 |",
        f"| target_modules | q/k/v/o/gate/up/down_proj |",
        f"| lora_dropout | 0.05 |",
        f"| task_type | CAUSAL_LM |",
        f"",
        f"## 生成方式",
        f"",
        f"```bash",
        f"python3 scripts/generate_all_instruct.py --scenes \\",
        f"    --lora_checkpoint {out_dir}/best",
        f"```",
    ]
    (out_dir / "training_info.md").write_text("\n".join(lines), encoding="utf-8")


def train(
    epochs:       int   = 10,
    lr:           float = 1e-4,
    grad_accum:   int   = 8,
    max_seq_len:  int   = 4096,
    output_dir:   str   = "outputs/instruct_lora",
    resume_from:  str | None = None,
    train_file:   str | None = None,
    val_file:     str | None = None,
):
    model_id = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "Qwen/Qwen3-8B-FP8"
    model, tokenizer = load_model(model_id)

    # ── 数据集 ────────────────────────────────────────────────
    train_path = Path(train_file) if train_file else PROCESSED_DIR / "instruct_train.jsonl"
    val_path   = Path(val_file)   if val_file   else PROCESSED_DIR / "instruct_val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"训练数据不存在：{train_path}\n"
            "请先依次运行：\n"
            "  1. conda run -n stone python3 scripts/generate_outlines.py --all\n"
            "  2. conda run -n stone python3 scripts/build_instruct_data.py [--sliding_window]"
        )

    train_dataset = InstructDataset(train_path, max_seq_len=max_seq_len)
    val_dataset   = InstructDataset(val_path,   max_seq_len=max_seq_len)

    print(f"  训练集：{len(train_dataset)} 条（第1-72回）")
    print(f"  验证集：{len(val_dataset)} 条（第73-80回）")

    # 自动计算 eval/save 步数（约每 epoch 一次）
    steps_per_epoch = max(1, len(train_dataset) // grad_accum)
    eval_save_steps = max(1, steps_per_epoch)
    total_steps     = steps_per_epoch * epochs
    print(f"  每 epoch 约 {steps_per_epoch} optimizer steps，共 {total_steps} steps")

    # ── 自动时间戳目录（若使用默认目录名则追加时间戳） ────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    sw_tag = "_sw" if "sw" in train_path.name else ""
    # 仅当用户没有手动指定具体目录时才自动加时间戳
    if output_dir in ("outputs/instruct_lora", "outputs/instruct_lora_sw"):
        output_dir = f"outputs/lora_{ts}{sw_tag}"
    run_id = Path(output_dir).name

    out_dir = ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 写训练说明文件 ──────────────────────────────────────────
    _write_training_info(
        out_dir,
        run_id=run_id,
        epochs=epochs,
        lr=lr,
        grad_accum=grad_accum,
        max_seq_len=max_seq_len,
        train_path=train_path,
        val_path=val_path,
        train_n=len(train_dataset),
        val_n=len(val_dataset),
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        base_model=model_id,
    )
    print(f"  输出目录：{out_dir}")

    # ── 训练参数 ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        logging_steps=max(1, eval_save_steps // 3),
        report_to="tensorboard",
        logging_dir=str(out_dir / "logs"),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # ── 日志 Callback ─────────────────────────────────────────
    log_callback = InstructStyleLogCallback(
        tokenizer=tokenizer,
        log_file=ROOT / "TRAINING_LOG_INSTRUCT.md",
        reports_dir=out_dir / "eval_reports",
        corpus_path=PROCESSED_DIR / "honglou_80_cleaned.txt",
    )

    # ── 初始化日志文件 ────────────────────────────────────────
    log_file = ROOT / "TRAINING_LOG_INSTRUCT.md"
    if not log_file.exists():
        log_file.write_text(
            "# Instruct LoRA 训练日志\n\n"
            f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"配置：epochs={epochs}, lr={lr}, grad_accum={grad_accum}, "
            f"max_seq_len={max_seq_len}\n\n---\n",
            encoding="utf-8",
        )

    # ── Trainer ───────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        callbacks=[log_callback],
    )

    print("\n开始 instruct 格式训练...")
    trainer.train(resume_from_checkpoint=resume_from)

    best_path = out_dir / "best"
    model.save_pretrained(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    print(f"\n训练完成，最佳模型保存至：{best_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="红楼梦 Instruct QLoRA 训练")
    parser.add_argument("--epochs",      type=int,   default=10,   help="训练轮数（默认 10）")
    parser.add_argument("--lr",          type=float, default=1e-4, help="学习率（默认 1e-4）")
    parser.add_argument("--grad_accum",  type=int,   default=8,    help="梯度累积步数（默认 8）")
    parser.add_argument("--max_seq_len", type=int,   default=4096, help="最大序列长度（默认 4096）")
    parser.add_argument("--output_dir",  type=str,   default="outputs/instruct_lora",
                        help="输出目录（相对项目根）")
    parser.add_argument("--resume",      type=str,   default=None, help="从 checkpoint 恢复")
    parser.add_argument("--train_file",  type=str,   default=None,
                        help="训练集 JSONL 路径（默认 data/processed/instruct_train.jsonl）")
    parser.add_argument("--val_file",    type=str,   default=None,
                        help="验证集 JSONL 路径（默认 data/processed/instruct_val.jsonl）")
    parser.add_argument("--sw",          action="store_true",
                        help="使用滑窗数据集（自动选 instruct_train_sw.jsonl / instruct_val_sw.jsonl）")
    args = parser.parse_args()

    # --sw 快捷方式：自动选滑窗数据集和默认输出目录
    train_file  = args.train_file
    val_file    = args.val_file
    output_dir  = args.output_dir
    if args.sw and train_file is None:
        train_file = str(PROCESSED_DIR / "instruct_train_sw.jsonl")
        val_file   = str(PROCESSED_DIR / "instruct_val_sw.jsonl")
        if output_dir == "outputs/instruct_lora":
            output_dir = "outputs/instruct_lora_sw"   # 触发时间戳逻辑

    train(
        epochs=args.epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        output_dir=output_dir,
        resume_from=args.resume,
        train_file=train_file,
        val_file=val_file,
    )


if __name__ == "__main__":
    main()
