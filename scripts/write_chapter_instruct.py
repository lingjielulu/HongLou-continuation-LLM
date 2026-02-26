"""指令模式章回续写脚本 — 直接用 Qwen3-8B chat 模式按大纲生成章回

用法：
    python scripts/write_chapter_instruct.py \\
        --chapter 81 \\
        --output outputs/chapter_081_instruct.txt

    # 分场景多轮生成（更长、更可控）
    python scripts/write_chapter_instruct.py \\
        --chapter 81 \\
        --scenes \\
        --output outputs/chapter_081_instruct.txt

与 write_chapter.py 的区别：
- 不加载 LoRA，直接使用 Qwen3-8B-FP8 的 chat/instruct 能力
- 用 system prompt 规定文风，user prompt 注入大纲与情节要求
- 可选分场景多轮：按大纲核心情节逐条生成，再拼接
"""

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent

LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

# ─────────────────────────────────────────────────────────────
# 模型加载（仅基座，无 LoRA）
# ─────────────────────────────────────────────────────────────
def load_base_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"加载基座模型：{LOCAL_FP8_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_FP8_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_FP8_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("模型加载完成")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# 大纲解析（复用 write_chapter.py 的逻辑）
# ─────────────────────────────────────────────────────────────
def parse_outline(chapter_num: int) -> dict:
    outline_path = ROOT / "outline" / "后40回大纲.md"
    text = outline_path.read_text(encoding="utf-8")

    cn_map = {
        81: "八十一", 82: "八十二", 83: "八十三", 84: "八十四", 85: "八十五",
        86: "八十六", 87: "八十七", 88: "八十八", 89: "八十九", 90: "九十",
        91: "九十一", 92: "九十二", 93: "九十三", 94: "九十四", 95: "九十五",
        96: "九十六", 97: "九十七", 98: "九十八", 99: "九十九", 100: "一百",
        101: "一百零一", 102: "一百零二", 103: "一百零三", 104: "一百零四",
        105: "一百零五", 106: "一百零六", 107: "一百零七", 108: "一百零八",
        109: "一百零九", 110: "一百一十", 111: "一百一十一", 112: "一百一十二",
        113: "一百一十三", 114: "一百一十四", 115: "一百一十五", 116: "一百一十六",
        117: "一百一十七", 118: "一百一十八", 119: "一百一十九", 120: "一百二十",
    }
    cn_num = cn_map.get(chapter_num, str(chapter_num))
    pattern = rf"###\s*第{cn_num}回[^\n]*\n(.*?)(?=###|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        raise ValueError(f"未找到第{chapter_num}回的大纲")

    block = m.group(0)
    title_line = block.splitlines()[0].replace("###", "").strip()

    def extract_field(label: str) -> str:
        fm = re.search(rf"\*\*{label}[：:]\*\*\s*\n(.*?)(?=\n\*\*|\Z)", block, re.DOTALL)
        if fm:
            return fm.group(1).strip()
        fm = re.search(rf"\*\*{label}[：:]\*\*\s*(.+)", block)
        if fm:
            return fm.group(1).strip()
        fm = re.search(rf"{label}[：:]\s*(.+)", block)
        return fm.group(1).strip() if fm else ""

    # 将核心情节拆成列表
    plot_raw = extract_field("核心情节")
    plot_items = [
        line.lstrip("- ").strip()
        for line in plot_raw.splitlines()
        if line.strip().startswith("-")
    ] or [plot_raw]

    return {
        "title":      title_line,
        "plot_items": plot_items,
        "plot_raw":   plot_raw,
        "characters": extract_field("主要人物"),
        "scenes":     extract_field("关键场景"),
        "tone":       extract_field("情感基调"),
        "function":   extract_field("叙事功能"),
    }


def get_prev_chapter_ending(chapter_num: int, tail_chars: int = 400) -> str:
    prev_num = chapter_num - 1
    path = ROOT / "data" / "chapters" / f"chap_{prev_num:03d}.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"要知后事[，,]下回分解[。.]?\s*(\(本章完\))?", "", text)
    text = re.sub(r"\(本章完\)", "", text)
    return text.strip()[-tail_chars:]


# ─────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位专业的《红楼梦》续写作家，深谙曹雪芹的文风与叙事技法。

【写作要求】
1. 语言风格：文言夹白话的章回体小说文风，与曹雪芹原著前80回保持一致。多用"话说""却说""只见""正是"等章回体习语。
2. 人物性格：严格遵循原著中各人物已有的性格、语气、习惯，不得出现性格突变。
3. 叙事节奏：有场景描写、人物对话、心理刻画，不可单纯概述情节。
4. 格式要求：每个自然段开头用全角空格"　　"缩进两格。
5. 禁止：不得输出现代白话解释、注释、分析或任何非小说正文的内容。
6. 【重要】"欲知后事如何，且听下回分解。"是章回结尾专用套语，只在被明确要求写结尾时才使用，其他任何时候绝对不得出现。"""


# ─────────────────────────────────────────────────────────────
# Chat 生成
# ─────────────────────────────────────────────────────────────
def chat_generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1200,
    temperature: float = 0.75,
    top_p: float = 0.92,
    repetition_penalty: float = 1.1,
) -> str:
    """单次 chat 生成"""
    # Qwen3 支持 enable_thinking 参数，创意写作关闭思考模式提升速度
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,      # 关闭 <think> 块，直接输出正文
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# 整章一次性生成
# ─────────────────────────────────────────────────────────────
def generate_whole_chapter(model, tokenizer, outline: dict,
                           prev_ending: str, chapter_num: int,
                           max_new_tokens: int) -> str:
    title = outline["title"]
    plot_text = outline["plot_raw"]
    user_prompt = f"""请续写《红楼梦》{title}。

【上一回（第{chapter_num-1}回）结尾】
{prev_ending}

【本回大纲】
章回标题：{title}
核心情节：
{plot_text}
主要人物：{outline['characters']}
关键场景：{outline['scenes']}
情感基调：{outline['tone']}

请按章回体格式，从章回标题开始，写出完整的一回正文（约2000字）。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    print("\n[生成整章...]")
    result = chat_generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)
    return result


# ─────────────────────────────────────────────────────────────
# 分场景多轮生成（更长、更可控）
# ─────────────────────────────────────────────────────────────
def generate_by_scenes(model, tokenizer, outline: dict,
                       prev_ending: str, chapter_num: int,
                       tokens_per_scene: int) -> str:
    title = outline["title"]
    plot_items = outline["plot_items"]
    n = len(plot_items)

    print(f"\n[分场景生成，共 {n} 个场景]")

    # 第一场景：给出标题 + 上回结尾 + 第1条情节
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    full_text = ""

    for i, scene in enumerate(plot_items):
        is_first = (i == 0)
        is_last  = (i == n - 1)

        if is_first:
            user_content = f"""请续写《红楼梦》{title}。

【上一回（第{chapter_num-1}回）结尾】
{prev_ending}

【本回标题】{title}
【本回情感基调】{outline['tone']}
【主要人物】{outline['characters']}

请从章回标题开始，写出第一个场景的正文（约500字）。
注意：这只是本回的第一段，不要写章回结尾套语，下面还有后续场景。
场景：{scene}"""
        else:
            last_note = "（这是本回最后一个场景，结尾处用「欲知后事如何，且听下回分解。」收束即可，不必另起章节）" if is_last else "（场景正文即可，不需要写章回结尾套语，下一段会继续）"
            user_content = f"""继续写下一个场景（约500字）：
场景：{scene}
{last_note}"""

        messages.append({"role": "user", "content": user_content})
        print(f"\n  [场景 {i+1}/{n}] {scene[:30]}...")

        scene_text = chat_generate(
            model, tokenizer, messages,
            max_new_tokens=tokens_per_scene,
        )
        print(f"  生成 {len(scene_text)} 字")

        # 非最后场景：去掉模型误生成的章回结尾套语（含各种变体）
        # 策略：从"正是："或"[欲未]知后事"首次出现处向前找，截断整段尾巴
        if not is_last:
            # 找"欲/未知后事"所在行的行首
            end_m = re.search(r"\n[　 ]*[欲未]知后事", scene_text)
            if end_m:
                # 向上找"正是："所在行，一并截掉
                prefix = scene_text[:end_m.start()]
                zhengshi_m = re.search(r"\n[　 ]*正是[：:]", prefix)
                if zhengshi_m:
                    scene_text = prefix[:zhengshi_m.start()].rstrip()
                else:
                    scene_text = prefix.rstrip()

        # 将助手回复加入对话历史，保持连贯
        messages.append({"role": "assistant", "content": scene_text})
        full_text += scene_text + "\n"

    return full_text.strip()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="红楼梦指令模式章回续写")
    parser.add_argument("--chapter",  type=int, required=True, help="章回号（如 81）")
    parser.add_argument("--scenes",   action="store_true",
                        help="分场景多轮生成（更长更可控，默认：整章一次生成）")
    parser.add_argument("--max_new_tokens", type=int, default=2000,
                        help="整章模式最大 token 数（默认 2000）")
    parser.add_argument("--tokens_per_scene", type=int, default=600,
                        help="分场景模式每场景 token 数（默认 600）")
    parser.add_argument("--output",   default=None, help="输出文件路径")
    parser.add_argument("--show_outline", action="store_true", help="只显示大纲，不生成")
    args = parser.parse_args()

    # 1. 解析大纲
    print(f"\n解析第 {args.chapter} 回大纲...")
    outline = parse_outline(args.chapter)
    print(f"标题：{outline['title']}")
    print(f"情节：")
    for item in outline["plot_items"]:
        print(f"  - {item}")
    print(f"人物：{outline['characters']}")
    print(f"基调：{outline['tone']}")

    if args.show_outline:
        return

    # 2. 上一回结尾
    prev_ending = get_prev_chapter_ending(args.chapter)
    if prev_ending:
        print(f"\n上一回结尾（后 {len(prev_ending)} 字）：\n{'-'*50}")
        print(prev_ending[-200:])   # 只打印最后200字
        print("-" * 50)

    # 3. 加载模型
    model, tokenizer = load_base_model()

    # 4. 生成
    if args.scenes:
        chapter_text = generate_by_scenes(
            model, tokenizer, outline, prev_ending,
            chapter_num=args.chapter,
            tokens_per_scene=args.tokens_per_scene,
        )
    else:
        chapter_text = generate_whole_chapter(
            model, tokenizer, outline, prev_ending,
            chapter_num=args.chapter,
            max_new_tokens=args.max_new_tokens,
        )

    # 5. 输出
    print(f"\n{'='*60}")
    print(f"生成完成，共 {len(chapter_text)} 字")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(chapter_text, encoding="utf-8")
        print(f"已保存至：{out_path}")
    else:
        print("\n" + chapter_text)


if __name__ == "__main__":
    main()
