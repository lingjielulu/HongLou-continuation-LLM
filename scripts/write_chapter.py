"""章回续写脚本 — 结合大纲与模型生成一个章回草稿

用法：
    python scripts/write_chapter.py \\
        --chapter 81 \\
        --checkpoint outputs/checkpoint-best \\
        --segments 10 \\
        --output outputs/chapter_081.txt

逻辑：
1. 从 outline/后40回大纲.md 提取目标章回大纲
2. 从 data/chapters/chap_0XX.txt 读取上一回末尾作为叙事衔接参考
3. 构造章回体开篇 prompt（模仿训练数据格式）
4. 多段循环生成，滑动窗口衔接上下文
5. 输出保存至 --output 文件
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────
# 复用 generate.py 的模型加载逻辑
# ─────────────────────────────────────────────────────────────
LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)


def _replace_fp8_recursive(module, seen_ids):
    from transformers.integrations import FP8Linear
    replaced = 0
    for child_name in list(module._modules.keys()):
        child = module._modules[child_name]
        if child is None:
            continue
        if isinstance(child, FP8Linear) and id(child) not in seen_ids:
            seen_ids.add(id(child))
            dev = child.weight.device
            out_f, in_f = child.weight.shape
            w_cpu  = child.weight.data.to("cpu").to(torch.float32)
            si_cpu = child.weight_scale_inv.data.to("cpu").float()
            if child.block_size is None:
                w_bf16 = (w_cpu * si_cpu.item()).to(torch.bfloat16)
            else:
                bh, bw = child.block_size
                n_out = (out_f + bh - 1) // bh
                n_in  = (in_f  + bw - 1) // bw
                if n_out * bh != out_f or n_in * bw != in_f:
                    w_cpu = nn.functional.pad(
                        w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f))
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(
                    n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)
            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                dtype=torch.bfloat16)
            new_lin.weight.data = w_bf16
            if child.bias is not None:
                new_lin.bias.data = child.bias.data.to("cpu").to(torch.bfloat16)
            del child, w_cpu, si_cpu, w_bf16
            setattr(module, child_name, new_lin)
            new_lin = new_lin.to(dev)
            setattr(module, child_name, new_lin)
            torch.cuda.empty_cache()
            replaced += 1
        else:
            replaced += _replace_fp8_recursive(child, seen_ids)
    return replaced


def load_model(checkpoint_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    peft_cfg = PeftConfig.from_pretrained(checkpoint_path)
    base_model_id = (
        LOCAL_FP8_PATH
        if Path(LOCAL_FP8_PATH).exists()
        else peft_cfg.base_model_name_or_path
    )

    print(f"加载 FP8 基座模型：{base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    n = _replace_fp8_recursive(base_model, set())
    print(f"[FP8→BF16] 已将 {n} 个 FP8Linear 反量化为 BF16")
    print(f"加载 LoRA 权重：{checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    print("[Merge] 合并 LoRA...")
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)
    print("[Merge] 完成，模型为纯 BF16")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# 生成函数（单段）
# ─────────────────────────────────────────────────────────────
DEFAULT_GEN_CONFIG = {
    "max_new_tokens":     400,
    "temperature":        0.85,
    "top_p":              0.92,
    "top_k":              50,
    "repetition_penalty": 1.15,
    "do_sample":          True,
}


def generate_segment(model, tokenizer, prompt: str, device: str = "cuda",
                     max_new_tokens: int = 400) -> str:
    cfg = dict(DEFAULT_GEN_CONFIG)
    cfg["max_new_tokens"] = max_new_tokens

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **cfg,
        )

    gen_ids  = output_ids[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text


# ─────────────────────────────────────────────────────────────
# 大纲解析
# ─────────────────────────────────────────────────────────────
def parse_outline(chapter_num: int) -> dict:
    """从 outline/后40回大纲.md 提取指定章回的大纲信息"""
    outline_path = ROOT / "outline" / "后40回大纲.md"
    text = outline_path.read_text(encoding="utf-8")

    # 将数字转为中文以匹配标题（如 81 → 八十一）
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

    # 匹配章回标题块
    pattern = rf"###\s*第{cn_num}回[^\n]*\n(.*?)(?=###|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        raise ValueError(f"未找到第{chapter_num}回的大纲，请检查 outline/后40回大纲.md")

    block = m.group(0)
    title_line = block.splitlines()[0].replace("###", "").strip()

    # 提取子字段（兼容 **标签：** 多行 和 **标签：** 单行 两种格式）
    def extract_field(label: str) -> str:
        # 多行：**标签：**\n- 内容\n- 内容
        fm = re.search(
            rf"\*\*{label}[：:]\*\*\s*\n(.*?)(?=\n\*\*|\Z)", block, re.DOTALL)
        if fm:
            return fm.group(1).strip()
        # 单行：**标签：** 内容
        fm = re.search(rf"\*\*{label}[：:]\*\*\s*(.+)", block)
        if fm:
            return fm.group(1).strip()
        # 无星号单行
        fm = re.search(rf"{label}[：:]\s*(.+)", block)
        return fm.group(1).strip() if fm else ""

    return {
        "title":      title_line,
        "plot":       extract_field("核心情节"),
        "characters": extract_field("主要人物"),
        "scenes":     extract_field("关键场景"),
        "tone":       extract_field("情感基调"),
    }


# ─────────────────────────────────────────────────────────────
# 读取上一回结尾（用于叙事衔接参考，不放入 prompt）
# ─────────────────────────────────────────────────────────────
def get_prev_chapter_ending(chapter_num: int, tail_chars: int = 300) -> str:
    prev_num = chapter_num - 1
    path = ROOT / "data" / "chapters" / f"chap_{prev_num:03d}.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    # 去掉"要知后事"和"(本章完)"等结尾套语
    text = re.sub(r"要知后事[，,]下回分解[。.]?\s*(\(本章完\))?", "", text)
    text = re.sub(r"\(本章完\)", "", text)
    return text.strip()[-tail_chars:]


# ─────────────────────────────────────────────────────────────
# 构造开篇 prompt
# ─────────────────────────────────────────────────────────────
# 手写第81回开篇引子（仿红楼梦章回体风格，"却说/话说"衔接）
CHAPTER_OPENERS = {
    81: (
        "第八十一回　哀元妃托梦示忧思　骇贾府闻讯各惊心\n\n"
        "　　却说迎春去后，众姊妹各自散去，心中皆有戚戚之感。"
        "宝玉独自徘徊于园中，见花木萧索，秋意渐浓，不觉神思恍惚。"
        "是夜，忽闻宫中有信传来，说元妃娘娘圣体违和，"
    ),
}

# 通用开篇模板（当没有手写开篇时使用）
def make_generic_opener(outline: dict, chapter_num: int) -> str:
    title = outline["title"]
    # 从标题提取两半
    parts = re.split(r"[　，,]", title.strip())
    half1 = parts[1] if len(parts) > 1 else ""
    return (
        f"{title}\n\n"
        f"　　却说荣宁二府，{half1[:4]}之事尚未平息，"
        f"忽又生出一段变故来。"
    )


def build_opening_prompt(chapter_num: int, outline: dict) -> str:
    if chapter_num in CHAPTER_OPENERS:
        return CHAPTER_OPENERS[chapter_num]
    return make_generic_opener(outline, chapter_num)


# ─────────────────────────────────────────────────────────────
# 多段生成主逻辑
# ─────────────────────────────────────────────────────────────
CONTEXT_WINDOW_CHARS = 800   # 每段生成前，保留的历史文本字符数

# 章回结束信号（检测到后截断并停止继续生成）
_CHAPTER_END_RE = re.compile(
    r"(欲知后事|且听下回|请看下文分解|下回分解)"
)

# 非叙事内容信号（Qwen3 base 预训练中"章回后跟注释"的模式）
_NON_NARRATIVE_LINE_RE = re.compile(
    r"^(>|#{1,4}\s|注释[：:]|章节简析|赏析[：：]|第\d+章\s)"
)


def sanitize_segment(text: str) -> tuple[str, bool]:
    """
    清理一段生成文本：
    1. 遇到章回结束套语 → 截断到该句末尾，返回 (截断文本, chapter_ended=True)
    2. 遇到注释/Markdown标题等非叙事行 → 截断到该行之前，返回 (截断文本, False)
    3. 无异常 → 返回 (原文本, False)
    """
    # 优先处理章回结束套语
    m = _CHAPTER_END_RE.search(text)
    if m:
        # 找到该句末尾（句号/感叹号/换行）
        end = m.end()
        tail = text[end:end + 20]
        extra = re.match(r"[^。！？\n]*[。！？]?", tail)
        cut = end + (len(extra.group()) if extra else 0)
        return text[:cut], True

    # 逐行扫描非叙事标记
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if _NON_NARRATIVE_LINE_RE.match(line.lstrip()):
            break
        clean_lines.append(line)

    cleaned = "\n".join(clean_lines)
    # 去掉末尾孤立的不完整括号/Markdown符号
    cleaned = re.sub(r"\n?[>#*`]+\s*$", "", cleaned)
    return cleaned, False


def write_chapter(
    model,
    tokenizer,
    chapter_num: int,
    outline: dict,
    segments: int = 10,
    max_new_tokens: int = 400,
    device: str = "cuda",
) -> str:
    """多段循环生成一个章回"""
    opening = build_opening_prompt(chapter_num, outline)
    print(f"\n开篇 prompt（{len(opening)} 字）：\n{'-'*60}")
    print(opening)
    print("-" * 60)

    full_text = opening
    context   = opening       # 当前送入模型的上下文

    for seg_idx in range(segments):
        print(f"\n[第 {seg_idx+1}/{segments} 段生成中...]")
        new_text = generate_segment(model, tokenizer, context,
                                    device=device, max_new_tokens=max_new_tokens)
        if not new_text.strip():
            print("  ⚠ 本段无输出，提前结束")
            break

        clean_text, chapter_ended = sanitize_segment(new_text)
        if len(clean_text) < len(new_text):
            print(f"  [净化] 截去 {len(new_text)-len(clean_text)} 字非叙事内容")

        full_text += clean_text
        print(f"  生成 {len(clean_text)} 字 | 累计 {len(full_text)} 字")

        if chapter_ended:
            print("  [检测到章回结束套语，停止生成]")
            break

        # 滑动窗口：取最新 CONTEXT_WINDOW_CHARS 字，末尾加段落缩进锚点
        # 强制模型下一句以叙事段落格式续写，防止跳入注释模式
        window = full_text[-CONTEXT_WINDOW_CHARS:]
        # 若窗口末尾已在句中（不以标点结尾），直接续写；否则新起段落
        if window.rstrip() and window.rstrip()[-1] in '。！？…」\u201d\u2019':
            context = window + "\n　　"
        else:
            context = window

    # 结尾套语（如已由模型生成则不重复追加）
    if not _CHAPTER_END_RE.search(full_text[-50:]):
        full_text += "\n　　欲知后事如何，且听下回分解。\n"
    return full_text


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="红楼梦章回续写")
    parser.add_argument("--chapter",    type=int, required=True, help="续写章回号（如 81）")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint 路径")
    parser.add_argument("--segments",   type=int, default=10, help="生成段数（默认 10）")
    parser.add_argument("--max_new_tokens", type=int, default=400,
                        help="每段最大生成 token 数（默认 400）")
    parser.add_argument("--output",     default=None,
                        help="输出文件路径（默认打印到终端）")
    parser.add_argument("--show_outline", action="store_true",
                        help="打印章回大纲后退出，不生成文本")
    args = parser.parse_args()

    # 1. 解析大纲
    print(f"\n解析第 {args.chapter} 回大纲...")
    outline = parse_outline(args.chapter)
    print(f"标题：{outline['title']}")
    print(f"核心情节：\n{outline['plot']}")
    print(f"主要人物：{outline['characters']}")
    print(f"情感基调：{outline['tone']}")

    if args.show_outline:
        return

    # 2. 读取上一回结尾（仅供参考，打印但不放入 prompt）
    prev_ending = get_prev_chapter_ending(args.chapter)
    if prev_ending:
        print(f"\n上一回结尾（后 {len(prev_ending)} 字，仅供参考）：\n{'-'*60}")
        print(prev_ending)
        print("-" * 60)

    # 3. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint)

    # 4. 多段生成
    chapter_text = write_chapter(
        model, tokenizer,
        chapter_num=args.chapter,
        outline=outline,
        segments=args.segments,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    # 5. 输出
    print(f"\n{'='*60}")
    print(f"生成完成，总计 {len(chapter_text)} 字")
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
