"""
构建 instruct 格式训练数据：将大纲 + 上回结尾 → 本回原文 配对

输入：
  - outline/前80回大纲/chap_XXX.json  （由 generate_outlines.py 生成）
  - data/chapters/chap_XXX.txt         （原著章节文本）

输出（tokenized，含 label masking）：
  - data/processed/instruct_train.jsonl
  - data/processed/instruct_val.jsonl
  - data/processed/instruct_stats.json

训练集：第 1-72 回
验证集：第 73-80 回

格式（每条 JSONL）：
{
  "input_ids": [...],      ← 完整对话 token ids
  "labels":    [...],      ← 仅 assistant 部分为真实 token，其余为 -100
  "chapter":   12,
  "length":    1890
}

用法：
    conda run -n stone python3 scripts/build_instruct_data.py
    conda run -n stone python3 scripts/build_instruct_data.py --max_len 2048
    conda run -n stone python3 scripts/build_instruct_data.py --verify
"""

import argparse
import json
import re
from pathlib import Path

ROOT          = Path(__file__).parent.parent
CHAPTERS_DIR  = ROOT / "data" / "chapters"
OUTLINE_DIR   = ROOT / "outline" / "前80回大纲"
PROCESSED_DIR = ROOT / "data" / "processed"

LOCAL_MODEL_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

TRAIN_CHAPTERS = set(range(1, 73))
VAL_CHAPTERS   = set(range(73, 81))

MAX_SEQ_LEN    = 4096   # instruct 样本比 CLM 长，给更大窗口
TAIL_CHARS     = 400    # 取上一回结尾的字符数

# ── 滑窗分片参数 ──────────────────────────────────────────────────────────
WINDOW_TOKENS  = 700    # 每个滑窗的 assistant token 数
STRIDE_TOKENS  = 500    # 相邻窗口的步进量（< WINDOW_TOKENS → 有重叠）
CONTEXT_CHARS  = 200    # 从上一窗口末尾取多少字作为衔接提示

SYSTEM_PROMPT = """你是一位专业的《红楼梦》续写作家，深谙曹雪芹的文风与叙事技法。
【写作要求】
1. 语言风格：文言夹白话的章回体小说文风，典雅而不晦涩
2. 叙事手法：工笔与写意并重，人物对话符合各自身份性格
3. 情节衔接：与上回结尾自然衔接，严格依照大纲展开情节
4. 体量控制：每回约 2000-3000 字
5. 禁止出现任何现代词汇、网络用语
6. 以"话说""却说"等章回体起首句开篇
7. 结尾以"正是：[对仗诗句]\\n欲知后事如何，且听下回分解。"收束"""

# 滑窗三种位置对应的 system prompt
SYSTEM_PROMPT_FIRST = """你是一位专业的《红楼梦》续写作家，深谙曹雪芹的文风与叙事技法。
【写作要求】
1. 语言风格：文言夹白话的章回体小说文风，典雅而不晦涩
2. 叙事手法：工笔与写意并重，人物对话符合各自身份性格
3. 情节衔接：与上回结尾自然衔接，严格依照大纲展开情节
4. 以"话说""却说"等章回体起首句开篇
5. 本段是本回开头，不要写章回结尾套语（后面还有续段）
6. 禁止出现任何现代词汇、网络用语"""

SYSTEM_PROMPT_MIDDLE = """你是一位专业的《红楼梦》续写作家，深谙曹雪芹的文风与叙事技法。
【写作要求】
1. 语言风格：文言夹白话的章回体小说文风，典雅而不晦涩
2. 叙事手法：工笔与写意并重，人物对话符合各自身份性格
3. 情节衔接：从已写内容处自然衔接，继续推进大纲情节
4. 直接续写正文，不要重复章回标题，不要写章回开篇或结尾套语
5. 禁止出现任何现代词汇、网络用语"""

SYSTEM_PROMPT_LAST = """你是一位专业的《红楼梦》续写作家，深谙曹雪芹的文风与叙事技法。
【写作要求】
1. 语言风格：文言夹白话的章回体小说文风，典雅而不晦涩
2. 叙事手法：工笔与写意并重，人物对话符合各自身份性格
3. 情节衔接：从已写内容处自然衔接，将本回情节收尾
4. 结尾以"正是：[对仗诗句]\\n欲知后事如何，且听下回分解。"收束
5. 禁止出现任何现代词汇、网络用语"""


def outline_to_prompt(outline: dict, prev_ending: str = "") -> str:
    """将大纲 JSON 转为 user prompt"""
    title = outline.get("title", f"第{outline.get('chapter', '?')}回")

    # 核心情节
    plot_items = outline.get("核心情节", [])
    if isinstance(plot_items, list):
        plot_str = "\n".join(f"- {p}" for p in plot_items)
    else:
        plot_str = str(plot_items)

    parts = [f"## {title}"]
    parts.append(f"\n**核心情节：**\n{plot_str}")

    for field in ["主要人物", "关键场景", "情感基调", "叙事功能"]:
        val = outline.get(field, "")
        if val:
            parts.append(f"\n**{field}：** {val}")

    if prev_ending:
        parts.append(f"\n**上回结尾（供衔接）：**\n{prev_ending}")

    parts.append("\n请根据以上大纲续写本回正文。")
    return "\n".join(parts)


def outline_to_prompt_window(
    outline: dict,
    prev_ending: str = "",
    context_tail: str = "",
    window_idx: int = 0,
    window_total: int = 1,
) -> str:
    """滑窗模式：将大纲 + 位置信息转为 user prompt"""
    title = outline.get("title", f"第{outline.get('chapter', '?')}回")

    plot_items = outline.get("核心情节", [])
    if isinstance(plot_items, list):
        plot_str = "\n".join(f"- {p}" for p in plot_items)
    else:
        plot_str = str(plot_items)

    parts = [f"## {title}"]
    parts.append(f"\n**核心情节：**\n{plot_str}")

    for field in ["主要人物", "关键场景", "情感基调", "叙事功能"]:
        val = outline.get(field, "")
        if val:
            parts.append(f"\n**{field}：** {val}")

    if window_idx == 0:
        if prev_ending:
            parts.append(f"\n**上回结尾（供衔接）：**\n{prev_ending}")
        if window_total == 1:
            parts.append("\n请根据以上大纲续写本回正文。")
        else:
            parts.append("\n请根据以上大纲续写本回正文第一段，不要写章回结尾套语。")
    else:
        if context_tail:
            parts.append(f"\n**已写内容结尾（续接此处）：**\n{context_tail}")
        if window_idx == window_total - 1:
            parts.append('\n请继续续写本回正文最后一段，以"正是：[对仗诗句]\\n欲知后事如何，且听下回分解。"结束。')
        else:
            parts.append("\n请继续续写本回正文，从以上内容处自然衔接，不要写章回开头或结尾套语。")

    return "\n".join(parts)


def get_prev_ending(chap_no: int, tail_chars: int = TAIL_CHARS) -> str:
    """获取上一回的结尾文本（从原著）"""
    prev_no = chap_no - 1
    if prev_no < 1:
        return ""
    chap_file = CHAPTERS_DIR / f"chap_{prev_no:03d}.txt"
    if not chap_file.exists():
        return ""
    text = chap_file.read_text(encoding="utf-8")
    # 去掉章回结束套语
    text = re.sub(r"要知后事[，,]下回分解[。.]?\s*", "", text)
    return text.strip()[-tail_chars:]


def build_chat_tokens(
    tokenizer,
    outline:      dict,
    chapter_text: str,
    prev_ending:  str = "",
    max_len:      int = MAX_SEQ_LEN,
    truncate:     bool = True,
) -> dict | None:
    """
    构建一条训练样本，返回 {input_ids, labels, chapter, length}
    labels 中 prompt 部分设为 -100（不计算 loss），只计算 assistant 回答部分的 loss
    truncate=True：章节正文超长时截断尾部而非跳过
    """
    user_content = outline_to_prompt(outline, prev_ending)

    # 仅 system + user 的对话（用于确定 prompt 长度）
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # 截断正文以适应 max_len（保留 prompt 空间 + 至少 512 tokens 正文）
    available = max_len - prompt_len - 10  # 10 for EOS/end tokens
    if available < 512:
        return None  # prompt 本身就太长，跳过

    assistant_ids = tokenizer.encode(chapter_text, add_special_tokens=False)
    if len(assistant_ids) > available:
        if not truncate:
            return None
        assistant_ids = assistant_ids[:available]

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": tokenizer.decode(assistant_ids, skip_special_tokens=True)},
    ]

    # 带 assistant 内容的完整对话（用于 labels）
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_len:
        # 最终截断保险
        full_ids = full_ids[:max_len]

    prompt_len = len(prompt_ids)

    # label masking：prompt 部分为 -100，assistant 部分为真实 token
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    assert len(full_ids) == len(labels), \
        f"长度不一致: input={len(full_ids)}, labels={len(labels)}"

    return {
        "input_ids": full_ids,
        "labels":    labels,
        "chapter":   outline.get("chapter", 0),
        "length":    len(full_ids),
        "prompt_len": prompt_len,
    }


def build_sliding_samples(
    tokenizer,
    outline:       dict,
    chapter_text:  str,
    prev_ending:   str = "",
    max_len:       int = MAX_SEQ_LEN,
    window_tokens: int = WINDOW_TOKENS,
    stride_tokens: int = STRIDE_TOKENS,
    context_chars: int = CONTEXT_CHARS,
) -> list[dict]:
    """将一章切分为多个滑窗训练样本，每个样本含完整大纲 + 正文片段"""
    all_ids = tokenizer.encode(chapter_text, add_special_tokens=False)
    if not all_ids:
        return []

    # 计算窗口起止点
    positions = []
    start = 0
    while start < len(all_ids):
        end = min(start + window_tokens, len(all_ids))
        positions.append((start, end))
        if end >= len(all_ids):
            break
        start += stride_tokens

    window_total = len(positions)

    def pick_sys(i):
        if window_total == 1:    return SYSTEM_PROMPT
        if i == 0:               return SYSTEM_PROMPT_FIRST
        if i == window_total - 1: return SYSTEM_PROMPT_LAST
        return SYSTEM_PROMPT_MIDDLE

    samples = []
    for i, (s, e) in enumerate(positions):
        window_ids  = all_ids[s:e]
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)

        # 上一窗口末尾 → 衔接提示
        if i == 0:
            context_tail = ""
        else:
            ps, pe = positions[i - 1]
            prev_text    = tokenizer.decode(all_ids[ps:pe], skip_special_tokens=True)
            context_tail = prev_text[-context_chars:]

        sys_prompt   = pick_sys(i)
        user_content = outline_to_prompt_window(
            outline, prev_ending, context_tail, i, window_total
        )

        # 计算 prompt 长度
        prompt_msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # 若超出 max_len，截断窗口（至少保留 100 tokens）
        available = max_len - prompt_len - 10
        if available < 100:
            continue
        if len(window_ids) > available:
            window_ids  = window_ids[:available]
            window_text = tokenizer.decode(window_ids, skip_special_tokens=True)

        messages = [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": window_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]

        labels = [-100] * prompt_len + full_ids[prompt_len:]
        assert len(full_ids) == len(labels)

        samples.append({
            "input_ids":    full_ids,
            "labels":       labels,
            "chapter":      outline.get("chapter", 0),
            "window_idx":   i,
            "window_total": window_total,
            "length":       len(full_ids),
            "prompt_len":   prompt_len,
        })

    return samples


def build_sliding_dataset(
    tokenizer,
    chapters:      set,
    max_len:       int = MAX_SEQ_LEN,
    window_tokens: int = WINDOW_TOKENS,
    stride_tokens: int = STRIDE_TOKENS,
    split_name:    str = "",
) -> list[dict]:
    samples    = []
    skipped    = 0
    no_outline = 0

    for chap_no in sorted(chapters):
        outline_path = OUTLINE_DIR / f"chap_{chap_no:03d}.json"
        chapter_path = CHAPTERS_DIR / f"chap_{chap_no:03d}.txt"

        if not outline_path.exists():
            no_outline += 1
            continue
        if not chapter_path.exists():
            skipped += 1
            continue

        with open(outline_path, encoding="utf-8") as f:
            outline = json.load(f)

        if outline.get("parse_error"):
            no_outline += 1
            continue

        chapter_text  = chapter_path.read_text(encoding="utf-8").strip()
        prev_ending   = get_prev_ending(chap_no)
        chap_samples  = build_sliding_samples(
            tokenizer, outline, chapter_text, prev_ending,
            max_len, window_tokens, stride_tokens,
        )

        if not chap_samples:
            skipped += 1
            print(f"  跳过第{chap_no}回（无法生成有效样本）")
            continue

        samples.extend(chap_samples)

    label = f"[{split_name}] " if split_name else ""
    print(f"  {label}大纲缺失：{no_outline} 回，跳过：{skipped} 回，有效样本：{len(samples)} 条")
    return samples


def build_dataset(
    tokenizer,
    chapters:  set,
    max_len:   int = MAX_SEQ_LEN,
    split_name: str = "",
) -> list[dict]:
    samples    = []
    skipped    = 0
    no_outline = 0

    for chap_no in sorted(chapters):
        outline_path = OUTLINE_DIR / f"chap_{chap_no:03d}.json"
        chapter_path = CHAPTERS_DIR / f"chap_{chap_no:03d}.txt"

        if not outline_path.exists():
            no_outline += 1
            continue
        if not chapter_path.exists():
            skipped += 1
            continue

        with open(outline_path, encoding="utf-8") as f:
            outline = json.load(f)

        # 跳过解析失败的大纲
        if outline.get("parse_error"):
            no_outline += 1
            continue

        chapter_text = chapter_path.read_text(encoding="utf-8").strip()
        prev_ending  = get_prev_ending(chap_no)

        sample = build_chat_tokens(tokenizer, outline, chapter_text, prev_ending, max_len, truncate=True)
        if sample is None:
            skipped += 1
            print(f"  跳过第{chap_no}回（prompt超限，无法生成有效样本）")
            continue

        samples.append(sample)

    label = f"[{split_name}] " if split_name else ""
    print(f"  {label}大纲缺失：{no_outline} 回，长度跳过：{skipped} 回，有效样本：{len(samples)} 条")
    return samples


def main():
    parser = argparse.ArgumentParser(description="构建 instruct 格式训练数据")
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN,
                        help=f"最大序列长度（默认 {MAX_SEQ_LEN}）")
    parser.add_argument("--verify",  action="store_true",
                        help="只验证数据集，不重新生成")
    parser.add_argument("--sliding_window", action="store_true",
                        help="滑窗分片模式：每章生成多个训练样本，覆盖完整正文")
    parser.add_argument("--window_tokens", type=int, default=WINDOW_TOKENS,
                        help=f"滑窗每片 assistant token 数（默认 {WINDOW_TOKENS}）")
    parser.add_argument("--stride_tokens", type=int, default=STRIDE_TOKENS,
                        help=f"滑窗步进 token 数（默认 {STRIDE_TOKENS}）")
    args = parser.parse_args()

    if args.verify:
        _verify(sw=args.sliding_window)
        return

    # ── 检查大纲文件是否足够 ──────────────────────────────────────────────
    outline_files = list(OUTLINE_DIR.glob("chap_*.json"))
    print(f"找到大纲文件：{len(outline_files)} 个（预期 80 个）")
    if len(outline_files) < 10:
        print("⚠ 大纲文件过少，请先运行：")
        print("   conda run -n stone python3 scripts/generate_outlines.py --all")
        return

    # ── 加载 tokenizer ────────────────────────────────────────────────────
    print("加载 tokenizer...")
    from transformers import AutoTokenizer
    model_id  = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "Qwen/Qwen3-8B-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  tokenizer 加载完成（词表大小 {tokenizer.vocab_size:,}）\n")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 构建训练集 / 验证集 ────────────────────────────────────────────────
    if args.sliding_window:
        print(f"模式：滑窗分片  window={args.window_tokens}  stride={args.stride_tokens}\n")
        print("构建训练集（第 1-72 回）...")
        train_samples = build_sliding_dataset(
            tokenizer, TRAIN_CHAPTERS, args.max_len,
            args.window_tokens, args.stride_tokens, "train",
        )
        print("\n构建验证集（第 73-80 回）...")
        val_samples = build_sliding_dataset(
            tokenizer, VAL_CHAPTERS, args.max_len,
            args.window_tokens, args.stride_tokens, "val",
        )
        train_out = PROCESSED_DIR / "instruct_train_sw.jsonl"
        val_out   = PROCESSED_DIR / "instruct_val_sw.jsonl"
        stats_out = PROCESSED_DIR / "instruct_stats_sw.json"
    else:
        print("构建训练集（第 1-72 回）...")
        train_samples = build_dataset(tokenizer, TRAIN_CHAPTERS, args.max_len, "train")
        print("\n构建验证集（第 73-80 回）...")
        val_samples   = build_dataset(tokenizer, VAL_CHAPTERS,   args.max_len, "val")
        train_out = PROCESSED_DIR / "instruct_train.jsonl"
        val_out   = PROCESSED_DIR / "instruct_val.jsonl"
        stats_out = PROCESSED_DIR / "instruct_stats.json"

    # ── 写入 JSONL ────────────────────────────────────────────────────────
    def write_jsonl(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  已写入：{path}  ({len(samples)} 条)")

    write_jsonl(train_samples, train_out)
    write_jsonl(val_samples,   val_out)

    # ── 统计 ──────────────────────────────────────────────────────────────
    stats = {}
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        lengths = [s["length"] for s in samples]
        plens   = [s["prompt_len"] for s in samples]
        if lengths:
            stats[split] = {
                "num_samples":   len(samples),
                "total_tokens":  sum(lengths),
                "avg_length":    round(sum(lengths) / len(lengths), 1),
                "min_length":    min(lengths),
                "max_length":    max(lengths),
                "avg_prompt_len": round(sum(plens) / len(plens), 1),
                "avg_label_len":  round(
                    sum(l - p for l, p in zip(lengths, plens)) / len(lengths), 1
                ),
            }

    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== 统计摘要 ===")
    for split, s in stats.items():
        print(f"\n[{split}]")
        for k, v in s.items():
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print(f"\n统计已保存：{stats_out}")


def _verify(sw: bool = False):
    """验证已生成的数据集"""
    from transformers import AutoTokenizer
    model_id  = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "Qwen/Qwen3-8B-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    import random
    random.seed(42)

    suffix = "_sw" if sw else ""
    for split in ["train", "val"]:
        path = PROCESSED_DIR / f"instruct_{split}{suffix}.jsonl"
        if not path.exists():
            print(f"文件不存在：{path}")
            continue

        samples = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n=== {split} ({len(samples)} 条) ===")

        # 随机抽 1 条展示
        s = random.choice(samples)
        prompt_len = s["prompt_len"]
        label_ids  = [t for t in s["labels"][prompt_len:] if t != -100]

        print(f"  第{s['chapter']}回  总长={s['length']}  prompt={prompt_len}  label={len(label_ids)}")
        print(f"\n  --- prompt 结尾（前100字）---")
        print(f"  {tokenizer.decode(s['input_ids'][:200], skip_special_tokens=True)[:200]}")
        print(f"\n  --- assistant 开头（前100字）---")
        print(f"  {tokenizer.decode(label_ids[:100], skip_special_tokens=True)[:150]}")

        # 检查 label masking 正确性
        masked = sum(1 for t in s["labels"] if t == -100)
        assert masked == prompt_len, f"label masking 异常：-100 count={masked}, prompt_len={prompt_len}"
    print("\n✓ 验证通过")


if __name__ == "__main__":
    main()
