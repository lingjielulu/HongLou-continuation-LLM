# LoRA 训练说明

**Run ID**：`lora_20260226_1616`
**训练时间**：2026-02-26 16:16 → 17:49（约 93 分钟）

## 基座模型

- 路径：`/home/lulingjie/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3`
- 量化：FP8（训练前递归反量化为 BF16，651 层）

## 训练数据

- 数据模式：**截断（truncation）**
- 训练集：`instruct_train.jsonl`（72 条样本，第 1-72 回）
- 验证集：`instruct_val.jsonl`（8 条样本，第 73-80 回）
- 大纲来源：`outline/前80回大纲/`（由 `generate_outlines.py` 从原著提取）
- 每章仅取前 ~1200 字（约 12% 正文覆盖率）

## 超参数

| 参数 | 值 |
|------|----|
| epochs | 20 |
| learning_rate | 1e-4 |
| grad_accumulation | 32 |
| effective_batch_size | 32 |
| max_seq_len | 1700 |
| steps_per_epoch | 2 |
| total_steps | 60 |
| lr_scheduler | cosine |
| warmup_ratio | 0.1 |
| optimizer | adamw_torch |
| bf16 | True |

## LoRA 配置

| 参数 | 值 |
|------|----|
| r | 16 |
| lora_alpha | 32 |
| target_modules | q/k/v/o/gate/up/down\_proj |
| lora_dropout | 0.05 |
| task_type | CAUSAL_LM |

## 训练结果

| 指标 | Step 2 | Step 60（最终） |
|------|--------|----------------|
| PPL | 29.13 | 13.42 |
| NLL | 3.37 | 2.60 |
| 4-gram | 0.107 | 0.401 |

## 已知问题

- 每章只训练开头 ~12% 内容，模型不了解章节后半部分写法
- 生成第81回时，第一场景输出了错误的章回标题（第85回）
- 部分生成结果混入日文字符和英文词

## 生成方式

```bash
python3 scripts/generate_all_instruct.py --scenes \
    --lora_checkpoint outputs/lora_20260226_1616/best \
    --chapters 81
```
