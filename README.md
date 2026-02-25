# 红楼梦文风微调系统设计文档 v1.0

> **项目阶段**：文风迁移（Style Transfer Only）
> **基础模型**：Qwen3-8B
> **训练方式**：QLoRA（4-bit NormalFloat）
> **训练硬件**：RTX 4090（24GB VRAM）

---

## 目录

1. [项目目标说明](#1-项目目标说明)
2. [模型训练策略](#2-模型训练策略)
3. [数据处理流程](#3-数据处理流程)
4. [训练集 / 验证集划分原则](#4-训练集--验证集划分原则)
5. [数据切分策略（Token 滑窗）](#5-数据切分策略token-滑窗)
6. [LoRA 训练参数建议](#6-lora-训练参数建议)
7. [评估指标设计](#7-评估指标设计)
8. [生成阶段使用方法](#8-生成阶段使用方法)
9. [未来扩展方向](#9-未来扩展方向)
10. [项目目录结构](#10-项目目录结构)

---

## 1. 项目目标说明

### 1.1 核心目标

训练一个以 Qwen3-8B 为基座、通过 QLoRA 微调的语言模型，使其在 **zero-shot prompt 下能生成具有《红楼梦》前 80 回文风特征的中文文本**。

文风特征包括：
- 叙事视角：全知叙事、散漫插叙
- 人物对话：半文言口吻，掺杂诗句、谑语
- 景物描写：铺排意象，骈偶句式
- 叙事结构：情节与细节交织，俚俗与典雅并用

### 1.2 当前阶段边界（明确不做）

| 项目 | 是否包含 |
|------|---------|
| 大纲 / 章节结构建模 | ❌ |
| 人物 Persona 建模 | ❌ |
| RAG / 检索增强 | ❌ |
| 对话系统 / 指令微调 | ❌ |
| 文本改写（Rewrite）Pipeline | ❌（留作扩展）|

### 1.3 成功标准

| 指标 | 目标值 |
|------|--------|
| 验证集 PPL | ≤ 训练前基座模型 PPL 的 60% |
| 4-gram 重叠率（vs 训练集） | ≥ 0.15（证明风格习得）|
| 4-gram 重叠率（vs 随机中文语料） | 明显高于基线（风格区分度） |
| 生成流畅度（人工评估 1-5 分） | ≥ 3.5 |

---

## 2. 模型训练策略

### 2.1 基座模型

```
模型：Qwen/Qwen3-8B
量化：bitsandbytes 4-bit NF4（NormalFloat4）
精度：compute_dtype = bfloat16
```

### 2.2 为什么选 QLoRA

RTX 4090（24GB VRAM）全量微调 8B 模型约需 80GB 显存（fp16），无法承载。QLoRA 通过以下方式将显存压缩至 ~14GB：

$$
\text{VRAM} \approx \underbrace{\frac{8B \times 4\text{bit}}{8}}_{\text{量化权重 ≈ 4GB}} + \underbrace{r \times d_{model} \times N_{layers} \times 2 \times 2}_{\text{LoRA 适配器（bf16）}} + \underbrace{\text{Gradient + Optimizer}}_{\approx 4\text{GB}}
$$

其中 $r=16$，适配器参数量约为 80M，bf16 存储约 160MB，总显存 ≈ 14-16GB。

### 2.3 训练目标

标准 Causal LM 自回归目标：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

所有 token 位置均参与损失计算（无 prompt masking），因为此阶段没有指令格式，全文均为语料。

### 2.4 LoRA 挂载层策略

仅对 Transformer 的注意力投影层（Q/K/V/O）和 FFN 门控层（gate/up/down proj）插入 LoRA：

```
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

不对 Embedding 层和 LM Head 插入 LoRA（避免词汇表扩散风险，且文风迁移无需修改 token 语义）。

---

## 3. 数据处理流程

### 3.1 原始数据规格

```
文件：honglou_80.txt
编码：UTF-8
内容：《红楼梦》前 80 回正文（去除现代标注、回目序号保留）
字数：约 70-80 万字
```

### 3.2 处理流水线

```
raw/honglou_80.txt
        │
        ▼
[Step 1] 文本清洗 (scripts/preprocess.py --step clean)
        │  - 去除多余空行（保留段落分隔）
        │  - 去除页码、注释标记、章节标题中的现代语
        │  - 统一标点：全角化，去除非文本符号
        │
        ▼
[Step 2] 回目切分 (scripts/preprocess.py --step split_chapters)
        │  - 按"第X回"切分为 80 个 chunk（保留标题行）
        │  - 输出 data/chapters/chap_{001..080}.txt
        │
        ▼
[Step 3] 全文合并 + Tokenize (scripts/preprocess.py --step tokenize)
        │  - 拼接所有章节文本
        │  - 使用 Qwen3-8B Tokenizer 编码为 token id 序列
        │  - 输出 data/processed/full_token_ids.bin（numpy uint32）
        │
        ▼
[Step 4] 滑窗切分 (scripts/preprocess.py --step chunk)
        │  - 按滑窗策略切分为固定长度 chunk
        │  - 输出 data/processed/train.jsonl / val.jsonl
        │
        ▼
[Step 5] 验证数据集 (scripts/preprocess.py --step verify)
           - 打印统计：总 token 数、chunk 数、平均长度
           - 抽样解码 5 条样本供人工核查
```

### 3.3 清洗规则（正则）

```python
import re

# 去除行内括号注释（现代人注）
text = re.sub(r'【[^】]*】', '', text)
text = re.sub(r'〔[^〕]*〕', '', text)

# 合并多个换行为一个段落分隔
text = re.sub(r'\n{3,}', '\n\n', text)

# 去除章节号前后多余符号（保留"第X回"本体）
text = re.sub(r'[-—=＝]{3,}', '', text)

# 全角标点统一
PUNCT_MAP = {',': '，', '.': '。', '?': '？', '!': '！',
             ';': '；', ':': '：', '"': '"', "'": '''}
for src, dst in PUNCT_MAP.items():
    text = text.replace(src, dst)
```

---

## 4. 训练集 / 验证集划分原则

### 4.1 划分策略：按章节划分（不随机打乱）

**禁止在 token 级别随机划分**。原因：滑窗 chunk 之间存在上下文重叠，随机划分会导致训练集与验证集 chunk 共享大量相同 token，PPL 虚低。

```
训练集：第 1-72 回（约 90%）
验证集：第 73-80 回（约 10%）
```

选择后 8 回作为验证集的理由：
- 后 8 回语言风格密度更高（金陵十二钗命运展开）
- 避免"开头 warmup 语料"对验证集的干扰
- 8 回约 8 万字，验证集 token 量充足（约 10 万 token）

### 4.2 划分执行

```python
TRAIN_CHAPTERS = range(1, 73)   # 第 1-72 回
VAL_CHAPTERS   = range(73, 81)  # 第 73-80 回
```

---

## 5. 数据切分策略（Token 滑窗）

### 5.1 参数设定

| 参数 | 值 | 说明 |
|------|----|------|
| `max_seq_len` | 2048 | 模型上下文长度，单个样本最大长度 |
| `stride` | 512 | 滑窗步长（重叠 = 2048 - 512 = 1536 tokens）|
| `min_chunk_len` | 256 | 丢弃短于此长度的尾部 chunk |

### 5.2 滑窗公式

设全文 token 序列长度为 $N$，窗口大小为 $L$，步长为 $S$：

$$
n_{\text{chunks}} = \left\lfloor \frac{N - L}{S} \right\rfloor + 1
$$

对于约 80 万字 → 约 100 万 token（Qwen3 tokenizer 中文平均压缩比约 1.2-1.5 char/token）：

$$
n_{\text{chunks}} \approx \frac{1{,}000{,}000 - 2048}{512} + 1 \approx 1953 \text{ chunks}
$$

训练集（90%）约 1757 chunks，验证集约 196 chunks。

### 5.3 切分实现

```python
def sliding_window_chunk(token_ids: list[int],
                          max_len: int = 2048,
                          stride: int = 512,
                          min_len: int = 256) -> list[dict]:
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_len, len(token_ids))
        chunk = token_ids[start:end]
        if len(chunk) >= min_len:
            chunks.append({
                "input_ids": chunk,
                "length": len(chunk)
            })
        if end == len(token_ids):
            break
        start += stride
    return chunks
```

### 5.4 输出格式（JSONL）

```jsonl
{"input_ids": [1234, 5678, ...], "length": 2048}
{"input_ids": [2048, 9012, ...], "length": 2048}
```

训练时 `labels = input_ids`（Causal LM 全序列预测）。

---

## 6. LoRA 训练参数建议

### 6.1 LoRA 超参

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lora_r` | 16 | 秩。文风迁移任务复杂度中等，r=16 足够 |
| `lora_alpha` | 32 | scaling = alpha/r = 2.0，偏大以加速收敛 |
| `lora_dropout` | 0.05 | 轻量正则，数据量小时防过拟合 |
| `bias` | `"none"` | 不训练 bias |
| `task_type` | `CAUSAL_LM` | |

### 6.2 训练超参

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `per_device_train_batch_size` | 4 | RTX 4090 / 2048 seq_len 下安全值 |
| `gradient_accumulation_steps` | 8 | 等效 batch_size = 32 |
| `num_train_epochs` | 3 | 数据量约 1800 chunks，3 epoch 约 5400 steps |
| `learning_rate` | 2e-4 | LoRA 标准推荐值 |
| `lr_scheduler_type` | `cosine` | |
| `warmup_ratio` | 0.05 | |
| `max_grad_norm` | 1.0 | |
| `bf16` | `true` | RTX 4090 支持 BF16 |
| `optim` | `paged_adamw_8bit` | bitsandbytes 分页优化器，节省显存 |
| `save_steps` | 200 | |
| `eval_steps` | 200 | |
| `logging_steps` | 10 | |

### 6.3 显存估算（RTX 4090 @ 2048 seq_len）

```
4-bit 量化权重：   ~4.5 GB
LoRA 适配器(bf16)：~0.3 GB
激活值（batch=4）：~6.0 GB
梯度 + 优化器：    ~3.0 GB
─────────────────────────
合计估算：         ~13.8 GB  ✓（24GB 安全）
```

若 OOM，优先调低 `per_device_train_batch_size` 至 2，再增 `gradient_accumulation_steps` 至 16。

### 6.4 量化配置

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # 二次量化，额外节省 ~0.4GB
)
```

---

## 7. 评估指标设计

### 7.1 验证集 PPL（主指标）

Perplexity 定义：

$$
\text{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i}) \right)
$$

**计算方式**：逐 chunk 计算 loss，加权平均（权重为 chunk token 数量），转换为 PPL：

```python
import torch
import math

def compute_ppl(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            # outputs.loss 是每 token 平均 NLL loss
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)
```

**注意**：滑窗 chunk 的首部 tokens 处于无上文状态，导致 loss 偏高。可通过设置 `label_mask_prefix_len=256` 屏蔽每个 chunk 前 256 个 token 的 loss 以获得更准确的 PPL。

### 7.2 N-gram 风格评估

通过计算生成文本与训练集之间的 n-gram 重叠率，量化风格习得程度。

**Self-BLEU 变体（Style Overlap）**：

$$
\text{StyleOverlap}_n = \frac{|\text{ngram}_n(\text{gen}) \cap \text{ngram}_n(\text{train})|}{|\text{ngram}_n(\text{gen})|}
$$

```python
from collections import Counter
from nltk.util import ngrams

def style_overlap(gen_text: str, ref_corpus: str, n: int = 4) -> float:
    """计算生成文本的 n-gram 在参考语料中的覆盖率"""
    gen_tokens  = list(gen_text)   # 字符级 ngram
    ref_tokens  = list(ref_corpus)

    gen_ngrams  = Counter(ngrams(gen_tokens, n))
    ref_ngrams  = set(ngrams(ref_tokens, n))

    overlap = sum(v for k, v in gen_ngrams.items() if k in ref_ngrams)
    total   = sum(gen_ngrams.values())
    return overlap / total if total > 0 else 0.0
```

**评估流程**：

1. 固定 5 个触发 prompt（见 `configs/eval_prompts.txt`）
2. 每个 prompt 生成 512 tokens × 5 次（temperature=0.8）
3. 拼接所有生成文本，计算 2/3/4-gram StyleOverlap
4. 同时计算生成文本 vs 随机现代中文语料的 StyleOverlap 作为基线对比

### 7.3 评估运行频率

```
每 200 steps 运行一次 PPL 评估（~5 分钟）
每 1000 steps 运行一次 n-gram 评估（生成耗时较长）
训练结束后运行完整评估报告
```

### 7.4 评估报告格式

```
Step 2000 Evaluation Report
─────────────────────────────────────
Val PPL:           18.43  (base: 32.17, ratio: 0.572)
2-gram StyleOverlap: 0.423
3-gram StyleOverlap: 0.231
4-gram StyleOverlap: 0.087
4-gram vs Modern:    0.009  (contrast: 9.7x)
─────────────────────────────────────
```

---

## 8. 生成阶段使用方法

### 8.1 加载模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen3-8B"
lora_weights   = "outputs/checkpoint-best"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, lora_weights)
model.eval()
```

### 8.2 生成参数推荐

```python
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,   # 抑制重复，文风微调后尤其重要
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
```

### 8.3 Prompt 格式

当前阶段无指令格式，直接以原文片段作为 prefix：

```python
# 风格延续模式：给定开头，续写
prompt = "话说那日贾宝玉在怡红院中闲坐，忽见"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output_ids = model.generate(**inputs, **generation_config)

generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
print(generated)
```

### 8.4 合并 LoRA 权重（部署用）

```bash
python scripts/merge_lora.py \
    --base_model Qwen/Qwen3-8B \
    --lora_weights outputs/checkpoint-best \
    --output_dir models/honglou_merged
```

合并后可直接用 `AutoModelForCausalLM.from_pretrained("models/honglou_merged")` 加载，无需 PEFT 依赖。

---

## 9. 未来扩展方向

### 9.1 阶段二：大纲 + 章节结构建模

```
架构：Hierarchical Prefix Tuning
─────────────────────────────────────────────────────────────
输入格式：
[OUTLINE] 贾宝玉与林黛玉初见，众人介绍，黛玉拜见贾母 [/OUTLINE]
[CHAPTER] 第三回 [/CHAPTER]
[TEXT] 且说黛玉自那日...
─────────────────────────────────────────────────────────────
训练数据：需构建 (回目摘要, 正文) 对
微调策略：在当前 LoRA 基础上继续 SFT，仅对 [TEXT] 之后的 token 计算损失
```

### 9.2 阶段三：Persona 建模

```
方法：角色标签前缀 + 对话数据集
─────────────────────────────────────────────────────────────
输入格式：
[SPEAKER: 凤姐] 这是说的什么话！[/SPEAKER]
[SPEAKER: 宝玉] 姐姐莫恼，[/SPEAKER]
─────────────────────────────────────────────────────────────
数据构建：从原文抽取对话三元组 (说话人, 语境, 话语)
额外挑战：说话人识别（需 NER + 规则）
```

### 9.3 阶段四：文风改写 Pipeline（Rewrite）

```
目标：将现代白话输入改写为红楼梦文风
架构：
  现代文输入
      │
      ▼
  [Retrieval] → 检索最相似原文段落（FAISS + BGE Embedding）
      │
      ▼
  [Rewrite Prompt]
  [STYLE_REF] {检索段落} [/STYLE_REF]
  [MODERN] {输入} [/MODERN]
  [CLASSIC]
      │
      ▼
  Honglou-LoRA 模型续写 [CLASSIC] 之后
─────────────────────────────────────────────────────────────
训练数据需求：(现代文, 文言风格文) 对齐语料（可用 GPT-4 合成）
```

### 9.4 扩展路线图

```
v1.0  ──► 文风习得（当前）
  │
  ▼
v2.0  ──► 大纲条件生成
  │
  ▼
v3.0  ──► 人物 Persona 对话
  │
  ▼
v4.0  ──► 全流程写作 Agent（大纲→章节→对话→改写）
```

---

## 10. 项目目录结构

```
honglou_style_lora/
│
├── README.md                    # 本文档
│
├── data/
│   ├── README.md                # 数据说明（不上传原始文本）
│   ├── raw/
│   │   └── honglou_80.txt       # 原始文本（手动放入，不入 git）
│   ├── chapters/
│   │   ├── chap_001.txt         # 第 1 回
│   │   └── ...                  # 第 2-80 回
│   └── processed/
│       ├── full_token_ids.bin   # 全文 token id 序列（numpy uint32）
│       ├── train.jsonl          # 训练集（滑窗 chunk）
│       ├── val.jsonl            # 验证集（滑窗 chunk）
│       └── stats.json           # 数据统计
│
├── configs/
│   ├── lora_config.yaml         # LoRA 超参配置
│   ├── training_config.yaml     # 训练超参配置
│   └── eval_prompts.txt         # 固定评估 prompt 列表
│
├── scripts/
│   ├── preprocess.py            # 数据处理全流程
│   ├── train.py                 # QLoRA 训练主脚本
│   ├── evaluate.py              # PPL + n-gram 评估
│   ├── generate.py              # 文本生成推理
│   └── merge_lora.py            # 合并 LoRA 权重
│
├── models/
│   └── honglou_merged/          # 合并后的完整模型（生产用）
│
└── outputs/
    ├── checkpoint-200/          # 训练中间 checkpoint
    ├── checkpoint-best/         # 最佳验证集 PPL checkpoint
    └── eval_reports/
        ├── step_0200.json
        └── ...
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install transformers peft bitsandbytes datasets accelerate nltk

# 2. 放入原始数据
cp /path/to/honglou_80.txt data/raw/

# 3. 数据预处理
python scripts/preprocess.py --all

# 4. 启动训练
python scripts/train.py --config configs/training_config.yaml

# 5. 评估
python scripts/evaluate.py \
    --checkpoint outputs/checkpoint-best \
    --val_data data/processed/val.jsonl

# 6. 生成
python scripts/generate.py \
    --checkpoint outputs/checkpoint-best \
    --prompt "话说那日贾宝玉在怡红院中闲坐，忽见"
```

---

*文档版本：v1.0 | 最后更新：2026-02-24*
