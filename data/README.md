# 数据目录说明

原始文本文件不进入版本控制。请手动准备以下文件。

## 准备原始数据

将《红楼梦》前 80 回纯文本（UTF-8）放置于：

```
data/raw/honglou_80.txt
```

**文件要求**：
- 编码：UTF-8
- 内容：前 80 回正文（可包含回目标题如"第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀"）
- 无需预处理，由 `scripts/preprocess.py` 自动清洗

## 目录结构（处理后）

```
data/
├── raw/
│   ├── honglou_80.txt               # 原始文本（手动放入，不入 git）
│   └── honglou_80_cleaned.txt       # 清洗后文本（自动生成）
│
├── chapters/
│   ├── chap_001.txt                 # 第 1 回
│   ├── chap_002.txt                 # 第 2 回
│   └── ...                          # 第 3-80 回
│
└── processed/
    ├── full_token_ids.bin            # 全文 token id 序列（numpy uint32）
    ├── chapter_boundaries.json       # 各回在 token 序列中的边界
    ├── train.jsonl                   # 训练集（第 1-72 回，滑窗 chunk）
    ├── val.jsonl                     # 验证集（第 73-80 回，滑窗 chunk）
    └── stats.json                    # 数据统计
```

## 数据处理命令

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

## .gitignore 建议

```gitignore
data/raw/
data/chapters/
data/processed/
models/
outputs/
```
