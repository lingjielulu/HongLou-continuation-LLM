"""
数据预处理脚本：从原始 txt 到 token chunk
文档参考：README.md §3, §4, §5

原始文件：/home/lulingjie/Stone/红楼梦.txt
  - 共 120 章，格式为"第X章"（阿拉伯数字）
  - 只使用前 80 章（训练+验证），后 40 章忽略
  - 训练集：第 1-72 章，验证集：第 73-80 章

用法：
    python scripts/preprocess.py --all
    python scripts/preprocess.py --step clean
    python scripts/preprocess.py --step split_chapters
    python scripts/preprocess.py --step tokenize
    python scripts/preprocess.py --step chunk
    python scripts/preprocess.py --step verify
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
# 原始文件直接读取项目外路径
RAW_FILE      = Path("/home/lulingjie/Stone/红楼梦.txt")
CHAPTERS_DIR  = ROOT / "data" / "chapters"
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_CHAPTERS = set(range(1, 73))   # 第 1-72 章
VAL_CHAPTERS   = set(range(73, 81))  # 第 73-80 章
USE_CHAPTERS   = set(range(1, 81))   # 只用前 80 章

MAX_SEQ_LEN = 2048
STRIDE      = 512
MIN_LEN     = 256


# ─────────────────────────────────────────────────────────────
# Step 1: 文本清洗
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """清洗原始文本：去注释、统一标点、规范化空白"""
    # 去除行内括号注释（现代人注）
    text = re.sub(r'【[^】]*】', '', text)
    text = re.sub(r'〔[^〕]*〕', '', text)

    # 去除页码行（纯数字行）
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 去除分隔线
    text = re.sub(r'[-—=＝※]{3,}', '', text)

    # 规范化空白：合并多余空行
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def step_clean():
    print("[Step 1] 清洗原始文本...")
    assert RAW_FILE.exists(), f"原始文件不存在：{RAW_FILE}"

    raw_text = RAW_FILE.read_text(encoding="utf-8")

    # 截取前 80 章内容（第81章起始位置之前）
    m81 = re.search(r'\n第81章', raw_text)
    if m81:
        raw_80 = raw_text[:m81.start()]
        print(f"  检测到第81章位置，截取前 80 章")
    else:
        raw_80 = raw_text
        print(f"  ⚠ 未找到第81章边界，使用全文")

    cleaned = clean_text(raw_80)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "honglou_80_cleaned.txt"
    out_path.write_text(cleaned, encoding="utf-8")

    print(f"  原始字符数（前80章）：{len(raw_80):,}")
    print(f"  清洗后字符数：{len(cleaned):,}")
    print(f"  输出：{out_path}")


# ─────────────────────────────────────────────────────────────
# Step 2: 按章节切分
# ─────────────────────────────────────────────────────────────
# 格式："第X章 标题"（X 为阿拉伯数字）
CHAPTER_PATTERN = re.compile(r'(第(\d+)章[^\n]*)')


def step_split_chapters():
    print("[Step 2] 按章节切分...")
    cleaned_file = PROCESSED_DIR / "honglou_80_cleaned.txt"
    assert cleaned_file.exists(), "请先运行 --step clean"

    text = cleaned_file.read_text(encoding="utf-8")
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    # 按"第X章"分割
    parts = CHAPTER_PATTERN.split(text)
    # split 结果：[前置, 全匹配, 数字, 正文, 全匹配, 数字, 正文, ...]

    chapters = {}
    i = 1
    while i + 2 < len(parts):
        full_title = parts[i]         # "第X章 标题"
        chap_no    = int(parts[i + 1])
        content    = parts[i + 2].strip()
        if chap_no in USE_CHAPTERS:
            chapters[chap_no] = f"{full_title}\n\n{content}"
        i += 3

    for chap_no, content in sorted(chapters.items()):
        out_path = CHAPTERS_DIR / f"chap_{chap_no:03d}.txt"
        out_path.write_text(content, encoding="utf-8")

    print(f"  切分出 {len(chapters)} 章（期望 80 章）")
    missing = [i for i in range(1, 81) if i not in chapters]
    if missing:
        print(f"  ⚠ 缺失章节：{missing}")
    else:
        print(f"  ✓ 第 1-80 章全部切分完成")
    print(f"  输出目录：{CHAPTERS_DIR}")


# ─────────────────────────────────────────────────────────────
# Step 3: Tokenize
# ─────────────────────────────────────────────────────────────
LOCAL_MODEL_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)


def step_tokenize(model_id: str | None = None):
    print("[Step 3] Tokenize 全文...")
    from transformers import AutoTokenizer

    model_id = model_id or (
        LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "Qwen/Qwen3-8B-FP8"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_ids: list[int] = []
    chapter_boundaries: dict[int, tuple[int, int]] = {}

    total_chars = 0
    for chap_no in range(1, 81):
        chap_file = CHAPTERS_DIR / f"chap_{chap_no:03d}.txt"
        if not chap_file.exists():
            print(f"  ⚠ 缺失：{chap_file.name}，跳过")
            continue
        text = chap_file.read_text(encoding="utf-8")
        total_chars += len(text)
        ids  = tokenizer.encode(text, add_special_tokens=False)

        start = len(all_ids)
        all_ids.extend(ids)
        end = len(all_ids)
        chapter_boundaries[chap_no] = (start, end)

    arr = np.array(all_ids, dtype=np.uint32)
    arr.tofile(PROCESSED_DIR / "full_token_ids.bin")

    with open(PROCESSED_DIR / "chapter_boundaries.json", "w", encoding="utf-8") as f:
        json.dump(chapter_boundaries, f, ensure_ascii=False, indent=2)

    compression = len(all_ids) / total_chars if total_chars else 0
    print(f"  总字符数：{total_chars:,}")
    print(f"  总 token 数：{len(all_ids):,}")
    print(f"  压缩比：{compression:.2f} token/char")
    print(f"  保存至：{PROCESSED_DIR / 'full_token_ids.bin'}")


# ─────────────────────────────────────────────────────────────
# Step 4: 滑窗切分
# ─────────────────────────────────────────────────────────────
def sliding_window_chunk(
    token_ids: list[int],
    max_len:   int = MAX_SEQ_LEN,
    stride:    int = STRIDE,
    min_len:   int = MIN_LEN,
) -> list[dict]:
    chunks = []
    start  = 0
    while start < len(token_ids):
        end   = min(start + max_len, len(token_ids))
        chunk = token_ids[start:end]
        if len(chunk) >= min_len:
            chunks.append({"input_ids": chunk, "length": len(chunk)})
        if end == len(token_ids):
            break
        start += stride
    return chunks


def step_chunk():
    print("[Step 4] 滑窗切分...")

    token_ids_path  = PROCESSED_DIR / "full_token_ids.bin"
    boundaries_path = PROCESSED_DIR / "chapter_boundaries.json"
    assert token_ids_path.exists(),  "请先运行 --step tokenize"
    assert boundaries_path.exists(), "请先运行 --step tokenize"

    all_ids = np.fromfile(token_ids_path, dtype=np.uint32).tolist()
    with open(boundaries_path, encoding="utf-8") as f:
        boundaries = {int(k): v for k, v in json.load(f).items()}

    train_ids: list[int] = []
    val_ids:   list[int] = []

    for chap_no, (start, end) in sorted(boundaries.items()):
        ids = all_ids[start:end]
        if chap_no in TRAIN_CHAPTERS:
            train_ids.extend(ids)
        elif chap_no in VAL_CHAPTERS:
            val_ids.extend(ids)

    train_chunks = sliding_window_chunk(train_ids)
    val_chunks   = sliding_window_chunk(val_ids)

    def write_jsonl(chunks: list[dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    write_jsonl(train_chunks, PROCESSED_DIR / "train.jsonl")
    write_jsonl(val_chunks,   PROCESSED_DIR / "val.jsonl")

    print(f"  训练集：{len(train_chunks):,} chunks（{len(train_ids):,} tokens，第 1-72 章）")
    print(f"  验证集：{len(val_chunks):,} chunks（{len(val_ids):,} tokens，第 73-80 章）")
    print(f"  滑窗参数：max_len={MAX_SEQ_LEN}, stride={STRIDE}")


# ─────────────────────────────────────────────────────────────
# Step 5: 验证
# ─────────────────────────────────────────────────────────────
def step_verify(model_id: str | None = None):
    print("[Step 5] 验证数据集...")
    from transformers import AutoTokenizer

    model_id = model_id or (
        LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "Qwen/Qwen3-8B-FP8"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    stats = {}
    for split in ["train", "val"]:
        path = PROCESSED_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        chunks  = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        lengths = [c["length"] for c in chunks]
        stats[split] = {
            "num_chunks":   len(chunks),
            "total_tokens": sum(lengths),
            "avg_length":   round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_length":   min(lengths) if lengths else 0,
            "max_length":   max(lengths) if lengths else 0,
        }

    print("\n=== 数据统计 ===")
    for split, s in stats.items():
        print(f"\n[{split}]")
        for k, v in s.items():
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    with open(PROCESSED_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n  统计已保存：{PROCESSED_DIR / 'stats.json'}")

    # 抽样解码
    print("\n=== 训练集抽样（3 条，前 100 tokens 解码）===")
    train_path = PROCESSED_DIR / "train.jsonl"
    if train_path.exists():
        import random
        random.seed(42)
        chunks  = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        samples = random.sample(chunks, min(3, len(chunks)))
        for i, s in enumerate(samples):
            decoded = tokenizer.decode(s["input_ids"][:100], skip_special_tokens=True)
            print(f"\n  [样本 {i+1}] length={s['length']}")
            print(f"  {decoded[:150]}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="红楼梦数据预处理")
    parser.add_argument(
        "--step",
        choices=["clean", "split_chapters", "tokenize", "chunk", "verify"],
    )
    parser.add_argument("--all", action="store_true", help="执行全部步骤")
    args = parser.parse_args()

    if args.all:
        step_clean()
        step_split_chapters()
        step_tokenize()
        step_chunk()
        step_verify()
    elif args.step == "clean":
        step_clean()
    elif args.step == "split_chapters":
        step_split_chapters()
    elif args.step == "tokenize":
        step_tokenize()
    elif args.step == "chunk":
        step_chunk()
    elif args.step == "verify":
        step_verify()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
