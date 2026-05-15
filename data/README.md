# 数据目录说明

当前仓库已经保留了 `data/chapters/chap_001.txt` 至 `chap_080.txt`，供 prompt baseline 读取上一回结尾、供旧训练路线复用。

原始全文、tokenized 数据和训练中间产物不进入版本控制。

## 当前主线需要的数据

prompt baseline 需要：

```text
data/chapters/chap_080.txt
```

生成第 81 回时会读取第 80 回结尾；生成第 82 回以后优先读取 `generations/prompt_baseline/` 中上一回的生成结果。

## 重新预处理旧训练数据

将《红楼梦》前 80 回纯文本（UTF-8）放置于：

```
data/raw/honglou_80.txt
```

**文件要求**：
- 编码：UTF-8
- 内容：前 80 回正文（可包含回目标题如"第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀"）
- 无需预处理，由 `scripts/preprocess.py` 自动清洗

## 目录结构

```
data/
├── raw/
│   ├── honglou_80.txt               # 原始文本（手动放入，不入 git）
│   └── honglou_80_cleaned.txt       # 清洗后文本（自动生成，不入 git）
│
├── chapters/
│   ├── chap_001.txt                 # 第 1 回（当前保留在仓库中）
│   ├── chap_002.txt                 # 第 2 回
│   └── ...                          # 第 3-80 回
│
└── processed/
    ├── full_token_ids.bin           # 全文 token id 序列（不入 git）
    ├── chapter_boundaries.json      # 各回 token 边界（不入 git）
    ├── train.jsonl                  # 训练集（不入 git）
    ├── val.jsonl                    # 验证集（不入 git）
    └── stats.json                   # 数据统计（不入 git）
```

## 旧训练路线的数据处理命令

```bash
# 一键处理全流程
python scripts/preprocess.py --all

# 或按步骤执行
python scripts/preprocess.py --step clean
python scripts/preprocess.py --step split_chapters
python scripts/preprocess.py --step tokenize
python scripts/preprocess.py --step chunk
python scripts/preprocess.py --step verify
```

这些命令属于 legacy LoRA 训练路线。当前 prompt baseline 不需要重新运行预处理。

## 版本控制边界

进入版本控制：

- `data/README.md`
- `data/chapters/chap_001.txt` 至 `chap_080.txt`

不进入版本控制：

- `data/raw/`
- `data/processed/`
- 模型权重、checkpoint 和训练日志
