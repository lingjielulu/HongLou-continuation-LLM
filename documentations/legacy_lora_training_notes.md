# Legacy LoRA 训练路线说明

> 文档状态：归档说明
> 最后更新：2026-05-14

## 1. 当前定位

本仓库早期主线是基于 Qwen3-8B 的《红楼梦》文风 QLoRA 微调。该路线已经完成过多轮实验，并留下了训练脚本、配置、日志和生成结果。

现在项目主线已经调整为 prompt-only baseline：

```text
大纲 + 上一回结尾 + 人物 card + 世界状态
    -> prompt_baseline
    -> OpenAI-compatible chat model
    -> baseline 续写结果
```

因此旧 LoRA 路线不再作为默认入口，只保留为后续“文风微调模型”的候选资产。

## 2. 为什么暂时不继续训练

当前最主要的问题不是模型是否有红楼文风，而是续写时缺少显式世界状态，导致人物命运和后续情节容易漂移。例如已死人物被写成主动行动，远嫁人物无交代回到现场，婚配关系提前或滞后。

训练可以改善文风，但不能可靠替代以下显式约束：

- 截至本回前的不可逆事件。
- 本回主要人物的状态和允许出现形式。
- 人物之间当前关系。
- 后 40 回大纲中的命运节点。

所以当前策略是先把 prompt baseline 做稳，再决定是否用 LoRA 解决文风。

## 3. 保留的旧资产

旧训练入口：

- `scripts/preprocess.py`
- `scripts/train.py`
- `scripts/train_instruct.py`
- `scripts/build_instruct_data.py`
- `scripts/baseline_eval.py`
- `scripts/evaluate.py`
- `scripts/merge_lora.py`
- `run_baseline.sh`
- `run_train_only.sh`

旧生成入口：

- `scripts/write_chapter.py`
- `scripts/write_chapter_instruct.py`
- `scripts/generate_all_instruct.py`
- `scripts/generate.py`
- `scripts/chat_lora.py`
- `scripts/lora_rewrite.py`

旧配置与日志：

- `configs/training_config.yaml`
- `configs/lora_config.yaml`
- `configs/eval_prompts.txt`
- `TRAINING_LOG.md`
- `TRAINING_LOG_INSTRUCT.md`

旧输出：

- `generations/base_qwen3_8b/`
- `generations/lora_20260226_1801_sw/`
- `outputs/lora_*`

## 4. 使用旧路线前的注意事项

旧脚本和旧 README 曾经以不同阶段的假设为基础，彼此并不完全一致。运行前需要重新核对：

- 本地模型路径是否仍存在。
- conda 环境名是否仍为 `stone`。
- FP8/Qwen3/LoRA checkpoint 路径是否匹配。
- 输出目录是否会覆盖已有实验。
- 当前代码是否需要人物 card 或世界状态注入。

旧路线如果重新启用，应作为一次单独任务处理，不要和 prompt baseline 的生成入口混用。

## 5. 后续回接方式

推荐回接顺序：

1. 先用 prompt baseline 生成一版第 81-120 回。
2. 做规则检查和人工审阅，确认人物命运不漂移。
3. 选择合格章节作为对照集。
4. 训练或选择文风模型。
5. 生成时仍保留 prompt baseline 的大纲、人物 card 和世界状态注入，只把文风模型作为后端。

也就是说，未来 LoRA 只负责“更像红楼梦”，不负责记住“世界当前是什么状态”。
