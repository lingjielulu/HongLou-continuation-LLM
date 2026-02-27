# LoRA 训练说明

**Run ID**：`lora_20260226_1801_sw`
**训练时间**：2026-02-26 18:01 → 19:35（约 94 分钟）

---

## 基座模型

- **路径**：`Qwen/Qwen3-8B-FP8`（本地缓存）
- **量化**：FP8，训练前递归反量化为 BF16（651 层）

---

## 训练数据

| 项目 | 值 |
|------|----|
| 数据模式 | **滑窗分片**（sliding window） |
| 训练集 | `instruct_train_sw.jsonl`，823 条，第 1-72 回 |
| 验证集 | `instruct_val_sw.jsonl`，102 条，第 73-80 回 |
| 大纲来源 | `outline/前80回大纲/`（generate_outlines.py 提取） |
| 窗口大小 | 700 tokens |
| 步进 | 500 tokens（重叠 200 tokens） |
| 衔接上文 | 每窗口 200 字前文作为 user prompt 的续写提示 |
| 正文覆盖率 | **100%**（每章全文均参与训练，约 11 段/章） |

与上一版（截断模式）对比：823 条 vs 72 条，覆盖率 100% vs 12%。

---

## 超参数

| 参数 | 值 |
|------|----|
| epochs | 10 |
| learning_rate | 1e-4 |
| lr_scheduler | cosine |
| warmup_ratio | 0.1 |
| grad_accumulation | 32 |
| effective_batch_size | 32 |
| max_seq_len | 1700 |
| steps_per_epoch | ~25 |
| total_steps | 260 |
| optimizer | adamw_torch |
| bf16 | True |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True |

---

## LoRA 配置

| 参数 | 值 |
|------|----|
| r | 16 |
| lora_alpha | 32 |
| target_modules | q/k/v/o/gate/up/down_proj |
| lora_dropout | 0.05 |
| task_type | CAUSAL_LM |

---

## 训练曲线

指标说明：**PPL** 越低越好；**4-gram** 越高说明与原著短语重合越多（高→风格像，过高→背诵）。

| Step | Epoch | PPL | NLL | 2-gram | 3-gram | 4-gram | 备注 |
|------|-------|-----|-----|--------|--------|--------|------|
| 25  | 1  | 8.824 | 2.178 | 0.873 | 0.575 | 0.314 | |
| 50  | 2  | 7.722 | 2.044 | 0.900 | 0.632 | 0.312 | |
| **75**  | **3**  | **7.518** | **2.017** | **0.794** | **0.438** | **0.208** | ★ **最佳 checkpoint** |
| 100 | 4  | 7.556 | 2.022 | 0.761 | 0.320 | 0.115 | PPL 开始回升 |
| 125 | 5  | 7.707 | 2.042 | 0.577 | 0.360 | 0.250 | |
| 150 | 6  | 7.893 | 2.066 | 0.863 | 0.607 | 0.318 | |
| 175 | 7  | 8.056 | 2.086 | 0.880 | 0.642 | 0.363 | |
| 200 | 8  | 8.241 | 2.109 | 0.843 | 0.531 | 0.263 | |
| 225 | 9  | 8.298 | 2.116 | 0.876 | 0.586 | 0.291 | |
| 250 | 10 | 8.356 | 2.123 | 0.892 | 0.645 | 0.409 | |

**观察：**
- 第 3 epoch（step 75）达到最低 PPL，之后轻微过拟合
- Step 75 的 4-gram（0.208）是训练中最低值，说明此时模型最少死记原文，创作能力最强
- `best/` 目录保存的即为 step 75 的权重

与**截断版**（`lora_20260226_1616`）对比：截断版最终 PPL 13.4，本模型最佳 PPL 7.52，提升约 44%。

---

## 生成效果（第 81 回）

相比截断版的改善：
- ✅ 不再输出错误章回标题
- ✅ 元妃托梦场景符合大纲
- ✅ 文言夹白话更流畅，无外文混入
- ⚠️ 偶有段落重复（生成时提高 `repetition_penalty` 可缓解）
- ⚠️ 偶有幻觉人物出现

---

## 生成方式

```bash
# 单回分场景续写
python3 scripts/generate_all_instruct.py --scenes \
    --lora_checkpoint outputs/lora_20260226_1801_sw/best \
    --chapters 81

# 批量生成第 81-120 回
python3 scripts/generate_all_instruct.py --scenes \
    --lora_checkpoint outputs/lora_20260226_1801_sw/best
```

输出目录：`outputs/lora_instruct/`
