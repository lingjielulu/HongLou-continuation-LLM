# 脚本入口清单

> 文档状态：整理说明
> 最后更新：2026-05-14

## 当前主线入口

### `scripts/prompt_baseline_generate.py`

用途：生成 prompt baseline。这里的 baseline 指最低可接受续写版本，重点是人物不严重 OOC、情节不出现大 bug，不是最终文风精修版本。

常用命令：

```bash
python3 scripts/prompt_baseline_generate.py --chapter 81 --dry-run
python3 scripts/prompt_baseline_generate.py --chapter 81 --model deepseek-v4-pro
```

依赖：

- `prompt_baseline/`
- `outline/后40回大纲.md`
- `Hongloumeng_card/cards/`
- `data/chapters/`

Card 注入说明：

- 当前不是多人物 agent 系统。
- 脚本会从本回大纲解析主要人物，读取并裁剪对应 card 后注入 prompt。
- 当前 timeline 尚未来自 card，只使用 `outline/后40回大纲.md` 中的人物命运总表作为轻量世界状态。

`.env` 配置：

```env
DEEPSEEK_API_KEY=你的_key
PROMPT_BASELINE_BASE_URL=https://api.deepseek.com
PROMPT_BASELINE_MODEL=deepseek-v4-pro
```

输出：

- `generations/prompt_baseline/chapter_*.txt`
- `generations/prompt_baseline/prompts/chapter_*.md`

## Legacy 训练路线

以下脚本属于旧 LoRA/训练路线，保留但不作为当前默认主线：

- `scripts/preprocess.py`
- `scripts/train.py`
- `scripts/train_instruct.py`
- `scripts/build_instruct_data.py`
- `scripts/baseline_eval.py`
- `scripts/evaluate.py`
- `scripts/merge_lora.py`
- `scripts/plot_metrics.py`

外层 shell 入口：

- `run_baseline.sh`
- `run_train_only.sh`

## Legacy 生成路线

以下脚本属于旧生成实验：

- `scripts/write_chapter.py`
- `scripts/write_chapter_instruct.py`
- `scripts/generate_all_instruct.py`
- `scripts/generate.py`
- `scripts/chat_lora.py`
- `scripts/lora_rewrite.py`
- `scripts/generate_outlines.py`

这些脚本可能仍可运行，但默认 prompt 协议、模型路径、输出目录和当前主线不完全一致。后续如果要继续使用，应先单独检查并重构，而不是直接并入 prompt baseline。

## 整理原则

- 新方案只新增到 `prompt_baseline/` 和明确命名的脚本入口。
- 旧方案不删除，先标注为 legacy。
- 生成结果按方案分目录保存，不复用旧目录。
- README 只描述当前推荐入口，细节说明放到 `documentations/`。
