# HongLou-continuation-LLM

《红楼梦》后 40 回续写实验仓库。

当前主线已经从“先训练 LoRA 文风模型”调整为“先做 prompt baseline，再视效果决定是否训练文风微调模型”。这里的 `prompt_baseline` 不是最终文学质量版本，而是最低可接受续写版本：人物不严重 OOC，情节不出现大 bug。旧训练代码仍保留，但不再作为默认入口。

## 当前阶段

主线目标：

- 基于 `outline/后40回大纲.md` 生成第 81-120 回。
- 使用 `Hongloumeng_card/cards/` 中的人物 card 约束人物关系、说话风格和行为边界。
- 使用后 40 回大纲中的人物命运总表作为轻量世界状态，避免已死、远嫁、出家、失踪人物被错误写成主动出场。
- 先产出人物和情节基本合格的 prompt baseline，再决定是否训练文风 LoRA。

`prompt_baseline` 的验收底线：

- 人物不严重 OOC：核心性格、说话方式、关系立场不能反向或突变。
- 情节不出现大 bug：已死人物不能主动出场，重大命运节点不能提前/滞后到破坏后续叙事，上一回到本回的衔接不能断裂。
- 世界状态可追踪：本回生成能解释人物在场、缺席、被提及或以梦境/回忆出现的理由。
- 不追求一步到位的文风精修：辞藻、句法和章回味可以后续用文风模型或人工修订提高。

暂不作为主线：

- 重新训练 QLoRA。
- 继续扩展旧 Qwen3 instruct 生成脚本。
- 把人物状态更新交给模型自由发挥。

## Card 注入策略

当前没有建立“几个主要人物 agent 按需调用”的多 agent 系统。`prompt_baseline` 现在采用的是确定性的 card 注入：

1. 从本回大纲的 `主要人物` 字段解析人物名单。
2. 按别名表归一化人物名，例如 `宝玉` -> `贾宝玉`、`黛玉` -> `林黛玉`。
3. 到 `Hongloumeng_card/cards/` 中读取对应人物 card。
4. 裁剪为适合 prompt 的摘要，默认保留人物定位、核心深描、关系总览、阶段变化、易错点/生成提醒。
5. 只注入本回相关人物，不把全部 card 一次性塞进 prompt。

因此这里的“按需求调用”指按本回大纲选择并注入 card，不是让每个人物拥有独立 agent 去分别生成或决策。

后续可以扩展人物 agent，但建议先把它定位为“审稿/校验角色”：检查某个人物是否 OOC、是否违反当前 timeline，而不是让多个 agent 分别写正文。

`Hongloumeng_card/` 作为 Git submodule 接入。首次 clone 后需要拉取 cards：

```bash
git submodule update --init --recursive
```

## Timeline 状态

当前 timeline 还没有建立在人物 card 内。现阶段只有一个轻量世界状态来源：

- `outline/后40回大纲.md` 里的“人物命运总表”

它用于判断截至本回前已经兑现的不可逆事件，例如死亡、远嫁、出家、失踪。等后续补充 card 内逐回 timeline 后，可以把 card timeline 作为更细的数据源接入，用来记录每回之后的人物位置、关系变化、允许出现形式和关键状态。

## 快速开始

只生成 prompt，不调用模型：

```bash
python3 scripts/prompt_baseline_generate.py --chapter 81 --dry-run
```

调用 OpenAI-compatible Chat Completions 接口生成正文：

```bash
export DEEPSEEK_API_KEY=你的_key
python3 scripts/prompt_baseline_generate.py \
  --chapter 81 \
  --model deepseek-v4-pro
```

也可以在仓库根目录使用 `.env` 注入配置：

```env
DEEPSEEK_API_KEY=你的_key
PROMPT_BASELINE_BASE_URL=https://api.deepseek.com
PROMPT_BASELINE_MODEL=deepseek-v4-pro
```

DeepSeek V4 当前官方模型名是 `deepseek-v4-pro` 和 `deepseek-v4-flash`。本项目默认使用 `deepseek-v4-pro`；如果希望先跑低成本版本，可以在 `.env` 中改成 `PROMPT_BASELINE_MODEL=deepseek-v4-flash`。接口地址默认使用 `https://api.deepseek.com`，也可以显式传入：

```bash
python3 scripts/prompt_baseline_generate.py \
  --chapter 81 \
  --model deepseek-v4-pro \
  --base-url https://api.deepseek.com
```

## 输出位置

prompt baseline 默认输出到：

```text
generations/prompt_baseline/
```

其中：

- `chapter_081.txt`：生成正文
- `prompts/chapter_081.md`：本次调用使用的完整 prompt，便于人工审阅

默认不覆盖已有正文；需要重跑时加：

```bash
--overwrite
```

## 主要目录

```text
prompt_baseline/                 # 当前主线：prompt-only baseline 逻辑
scripts/prompt_baseline_generate.py

Hongloumeng_card/cards/          # 人物 card 数据源
outline/后40回大纲.md             # 后 40 回大纲与命运总表
data/chapters/                   # 前 80 回分章文本，供第 81 回衔接使用
generations/                     # 各方案生成结果

scripts/train*.py                # legacy：旧 LoRA/训练路线
configs/*.yaml                   # legacy：旧训练配置
outputs/                         # legacy：旧训练与评估输出
```

## 文档索引

当前方案：

- `documentations/prompt_baseline_work_design.md`
- `documentations/DESIGN_CONTEXT_AGENT.md`
- `documentations/script_inventory.md`

旧训练路线归档：

- `documentations/legacy_lora_training_notes.md`
- `TRAINING_LOG.md`
- `TRAINING_LOG_INSTRUCT.md`

背景材料：

- `documentations/project_proposal.md`
- `documentations/classic_generation_review.md`
- `documentations/text_generation_metrics_examples.md`

## 旧方案状态

旧 README 原本描述的是 2026-02 的 QLoRA 文风迁移方案，后来仓库经历了 instruct 生成、后 40 回批量生成、LoRA 输出归档、人物 card 接入等几轮变化，原 README 已不能作为当前操作指南。

旧训练路线不删除，原因是后续仍可能用来训练“文风微调模型”。但它现在只承担 legacy/实验资产角色：

- 可以参考旧训练日志和配置。
- 不应作为生成 baseline 的默认入口。
- 新生成结果不写入旧 `base_qwen3_8b` 或 `lora_*` 目录。

## 推荐工作流

1. 用 `--dry-run` 生成并检查 prompt。
2. 调用目标模型生成第 81 回。
3. 人工检查人物状态、已逝人物出现形式、章回语言和现代说明混入问题。
4. 批量生成第 81-120 回。
5. 与旧 `generations/base_qwen3_8b/`、`generations/lora_20260226_1801_sw/` 对比。
6. 若叙事事实稳定，再训练文风微调模型。

《红楼梦》续书纷纭，众本并存。本项目将在参考前人续作、红学研究与读者反馈的基础上持续修订，希望人物、情节、文体与意蕴不违背原著本意。挂一漏万，谨为芹献。
