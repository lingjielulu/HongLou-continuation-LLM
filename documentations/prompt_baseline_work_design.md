# Prompt Baseline 工作设计文档

> 文档状态：执行设计 v0.1
> 最后更新：2026-05-14
> 关联文档：`DESIGN_CONTEXT_AGENT.md`、`project_proposal.md`

## 1. 当前决策

当前阶段先不训练模型，先做一个可复现、可评估、可替换模型后端的 prompt baseline。这里的 `prompt_baseline` 指最低可接受续写版本：人物不严重 OOC，情节不出现大 bug。它的目标不是一次性解决“红楼文风”，而是先把人物状态、年表约束、后 40 回大纲和已有人物 card 接入生成流程，得到一个人物与情节基本稳住的版本。

基础模型暂按 OpenAI-compatible Chat Completions 接口接入，默认模型名配置为 DeepSeek V4 Pro：`deepseek-v4-pro`。如果需要更低成本或更快生成，可通过 `.env` 或命令行参数切换为 `deepseek-v4-flash`，不改业务逻辑。

## 2. 旧方案与新方案边界

旧方案是训练/LoRA 路线，核心资产包括：

- `scripts/train.py`、`scripts/train_instruct.py`
- `scripts/build_instruct_data.py`
- `scripts/generate_all_instruct.py`
- `configs/lora_config.yaml`、`configs/training_config.yaml`
- `outputs/lora_*`、`generations/lora_*`

新方案是 prompt baseline 路线，核心资产包括：

- `prompt_baseline/`：prompt 构造、人物卡裁剪、年表抽取、模型客户端
- `scripts/prompt_baseline_generate.py`：单回生成入口
- `Hongloumeng_card/cards/`：人物 card 数据源
- `outline/后40回大纲.md`：后 40 回叙事骨架
- `generations/prompt_baseline/`：新 baseline 输出目录

两条路线不共享运行入口，不共享默认输出目录，不互相加载模型权重。以后若训练文风模型，只把 prompt baseline 的“输入上下文协议”和“评估用生成结果”作为训练前后的对照，不把训练逻辑混入 prompt baseline。

## 3. Baseline 目标

第一版 baseline 的质量底线：

- 人物不严重 OOC：核心性格、说话风格、关系立场不反向，不把人物写成与 card 明显冲突的现代角色。
- 情节不出现大 bug：不可逆事件不被撤销，重大命运节点不被随意提前或推迟，上一回结尾到本回开头能自然衔接。
- 世界状态可解释：人物为什么在场、缺席、被提及、梦见或回忆，都能被大纲、上一回结尾、命运总表或人物 card 支撑。
- 允许文风不完美：章回味、句法密度、诗性表达先不作为第一验收门槛。

第一版 baseline 的工程成功标准：

- 能按第 81-120 回大纲生成对应章回。
- 能注入上一回结尾，保证连续性。
- 能注入本回主要人物 cards 的压缩版。
- 能从命运总表抽取已发生的不可逆事件，避免已死、远嫁、出家、失踪人物被错误写成主动行动。
- 能 dry-run 输出完整 prompt，便于人工审阅。
- 能通过统一 Chat Completions 客户端接入 DeepSeek 或其他兼容模型。

不在第一版解决：

- 自动更新动态人物状态卡。
- 自动构造完整逐回年表。
- 自动评价文风优劣。
- LoRA 训练或微调。
- 多模型投票、反思、agent 化修稿。

## 4. Prompt 协议

每次生成第 N 回时，prompt 分为六层：

1. 写作身份与格式约束：章回体、半文半白、只输出小说正文。
2. 硬性叙事约束：不得复活已死人物；远嫁、出家、失踪人物只能按状态出现；不能新增现代解释。
3. 上一回结尾：优先读取 prompt baseline 已生成结果，否则第 81 回回退到前 80 回正文。
4. 本回大纲：标题、核心情节、主要人物、关键场景、情感基调、叙事功能。
5. 世界状态摘要：从命运总表提取截至本回前已经兑现的不可逆事件。
6. 人物 card 摘要：只注入本回主要人物，不整库注入。

人物 card 裁剪原则：

- 保留人物定位、核心深描、关系总览、阶段变化、易错点/生成提醒。
- 控制单人 card 字符数，默认最多 1800 字。
- 优先保障本回主要人物；梦境、述及人物也可注入，但必须标记出现形式。

### 4.1 Card 注入策略

当前实现不是“为几个主要人物建立独立 agent，再按需求调用 agent”。第一版先避免引入多 agent 编排，采用确定性、可审阅的 card 注入流程：

1. 从本回大纲的 `主要人物` 字段解析人物名单。
2. 通过别名表归一化人物名，例如 `宝玉` -> `贾宝玉`、`黛玉` -> `林黛玉`。
3. 在 `Hongloumeng_card/cards/` 中读取对应人物 card。
4. 对 card 进行裁剪，默认保留人物定位、核心深描、关系总览、阶段变化、易错点/生成提醒。
5. 将裁剪后的 card 摘要作为本回 prompt 的上下文块注入。

这里的“按需求调用”指“按本回大纲和叙事需要选择哪些 card 注入”，不是让每个人物拥有独立生成权。正文仍由一个模型调用完成，card 只作为约束上下文。

未来如果引入人物 agent，建议先作为校验层，而不是生成层：

- 生成前：检查本回大纲是否需要补充某个已逝/远嫁/失踪人物的约束。
- 生成后：让某人物校验器检查是否 OOC、是否出现不合理行动或关系突变。
- 修订时：把校验结果转成具体 rewrite 指令，而不是让多个 agent 分段写正文。

### 4.2 Timeline 当前状态

当前 timeline 还没有建立在人物 card 内。第一版只有一个轻量世界状态来源：

- `outline/后40回大纲.md` 中的“人物命运总表”

它只负责不可逆事件的粗粒度约束，例如死亡、远嫁、出家、失踪。它不能替代完整动态 timeline，也不能记录每回之后的人物位置、心理处境、关系变化和允许出现形式。

因此当前 card 的作用主要是提供前 80 回的人物定位、关系、说话风格和易错点；当前 timeline 的作用主要是防止后 40 回重大命运节点被写错。两者暂时是分离的。

后续你补充 card 内 timeline 后，接入顺序建议是：

1. 每张 card 增加“后40回动态状态”或“timeline”字段。
2. 生成第 N 回时，读取截至第 N-1 回的最新状态。
3. 若 card timeline 与大纲命运总表冲突，以大纲命运总表中的不可逆事件为准，并在 dry-run prompt 中显示告警。
4. prompt 中同时注入“人物静态 card 摘要”和“截至本回前动态状态摘要”。

## 5. 数据与输出

输入：

- `outline/后40回大纲.md`
- `data/chapters/chap_080.txt`
- `Hongloumeng_card/cards/*.md`
- 已生成的 `generations/prompt_baseline/chapter_*.txt`

输出：

- `generations/prompt_baseline/chapter_081.txt`
- 可选 prompt 审阅文件：`generations/prompt_baseline/prompts/chapter_081.md`

默认不覆盖已有输出；需要重跑时显式传 `--overwrite`。

## 6. 实现步骤

第一阶段：结构隔离与 prompt dry-run。

- 新建 `prompt_baseline/` 包。
- 新增大纲解析、人物 card 索引、命运总表抽取、prompt builder。
- 新增 CLI，支持 `--dry-run` 输出 prompt。

第二阶段：接入模型调用。

- 使用 OpenAI-compatible `/chat/completions`。
- 从仓库根目录 `.env` 或系统环境变量读取 API key、base URL 和 model id。
- 保存生成正文与 prompt。

第三阶段：批量生成与评估。

- 增加第 81-120 回批量生成脚本。
- 增加规则检查：严重 OOC、重大情节 bug、已逝人物主动说话/动作、结尾套语位置、现代说明混入。
- 与旧 `base_qwen3_8b`、`lora_20260226_1801_sw` 输出做横向对比。

第四阶段：文风微调回接。

- 若 prompt baseline 的叙事事实合格，再训练轻量文风模型。
- 训练目标只负责文风，不负责世界状态记忆。
- 生成时仍保留 prompt baseline 的年表和人物状态注入。

## 7. 近期文件整理建议

先不移动旧代码，避免破坏可复现实验。整理方式采用“标注和隔离”：

- 在文档中明确旧训练路线和新 prompt baseline 路线。
- 新增代码只放入 `prompt_baseline/` 和 `scripts/prompt_baseline_generate.py`。
- 旧训练输出、旧副本 `honglou_style_lora/` 暂不纳入新入口。
- 后续确认无用后，再单独做一次归档提交。

## 8. 风险

- DeepSeek V4 的正式 model id 和上下文长度需要运行前确认；当前实现用命令行参数隔离此不确定性。
- 人物 cards 来自前 80 回分析，不天然包含第 81 回后的动态变化；第一版只用命运总表补不可逆事件。
- 仅靠 prompt 仍可能出现“语义服从失败”，需要后续加规则检查和二次修稿。
