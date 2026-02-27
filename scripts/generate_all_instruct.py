"""批量生成后40回（第81-120回）— 大纲+instruct 模式

输出目录：
    outputs/instruct/          — 纯基座模型（默认）
    outputs/lora_instruct/     — LoRA + chat 模式（--lora_checkpoint 指定）

用法：
    # 基座模型，分场景多轮生成
    python scripts/generate_all_instruct.py --scenes

    # LoRA 接入 chat 模式（推荐：风格更接近红楼梦）
    python scripts/generate_all_instruct.py --scenes \\
        --lora_checkpoint outputs/checkpoint-best

    # 断点续跑
    python scripts/generate_all_instruct.py --scenes --start 85

    # 只生成指定几回
    python scripts/generate_all_instruct.py --scenes --chapters 89 90 91

特性：
- 模型只加载一次，循环生成全部章回
- 自动跳过已存在的输出文件（支持断点续跑）
- 每回生成后立即保存，不会因中断丢失前面的进度
- 上一回结尾优先从当前模式的输出目录读取，否则回退到 data/chapters/
- LoRA 模式：FP8→BF16 反量化后挂载 LoRA，全程走 chat template（不退回 causal LM）
"""

import argparse
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent

LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

CN_MAP = {
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


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────
def load_base_model():
    """加载纯基座 FP8 模型（无 LoRA）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n{'='*60}")
    print(f"加载基座模型：{LOCAL_FP8_PATH}")
    print(f"{'='*60}")
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
    print("基座模型加载完成\n")
    return model, tokenizer


def _dequantize_fp8(module, seen_ids: set) -> int:
    """递归将 FP8Linear 反量化为标准 BF16 Linear（LoRA 挂载前必须做）"""
    import torch.nn as nn
    try:
        from transformers.integrations import FP8Linear
    except ImportError:
        return 0
    replaced = 0
    for name in list(module._modules.keys()):
        child = module._modules[name]
        if child is None:
            continue
        if isinstance(child, FP8Linear) and id(child) not in seen_ids:
            seen_ids.add(id(child))
            dev = child.weight.device
            out_f, in_f = child.weight.shape
            w_cpu  = child.weight.data.to("cpu").float()
            si_cpu = child.weight_scale_inv.data.to("cpu").float()
            if child.block_size is None:
                w_bf16 = (w_cpu * si_cpu.item()).to(torch.bfloat16)
            else:
                bh, bw = child.block_size
                n_out = (out_f + bh - 1) // bh
                n_in  = (in_f  + bw - 1) // bw
                if n_out * bh != out_f or n_in * bw != in_f:
                    import torch.nn.functional as F
                    w_cpu = F.pad(w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f))
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)
            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None, dtype=torch.bfloat16)
            new_lin.weight.data = w_bf16
            if child.bias is not None:
                new_lin.bias.data = child.bias.data.to("cpu").bfloat16()
            del child, w_cpu, si_cpu, w_bf16
            setattr(module, name, new_lin.to(dev))
            torch.cuda.empty_cache()
            replaced += 1
        else:
            replaced += _dequantize_fp8(child, seen_ids)
    return replaced


def load_lora_model(checkpoint_path: str):
    """加载 LoRA checkpoint，走 chat 模式生成（不退回 causal LM）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    import transformers.trainer as _t
    _t.validate_quantization_for_training = lambda m: None   # 绕过量化检查

    cfg = PeftConfig.from_pretrained(checkpoint_path)
    base_id = LOCAL_FP8_PATH if Path(LOCAL_FP8_PATH).exists() else cfg.base_model_name_or_path

    print(f"\n{'='*60}")
    print(f"加载 FP8 基座：{base_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n = _dequantize_fp8(base, set())
    print(f"[FP8→BF16] {n} 层反量化完成")

    model = PeftModel.from_pretrained(base, checkpoint_path)
    model = model.merge_and_unload().to(torch.bfloat16)
    print(f"[LoRA Merge] 完成，checkpoint: {checkpoint_path}")
    model.eval()
    print(f"{'='*60}\n")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# 大纲解析
# ─────────────────────────────────────────────────────────────
def parse_outline(chapter_num: int) -> dict:
    outline_path = ROOT / "outline" / "后40回大纲.md"
    text = outline_path.read_text(encoding="utf-8")
    cn_num = CN_MAP.get(chapter_num, str(chapter_num))
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


# ─────────────────────────────────────────────────────────────
# 获取上一回结尾（优先读当前模式的输出目录，其次读原著数据）
# ─────────────────────────────────────────────────────────────
def get_prev_ending(chapter_num: int, tail_chars: int = 400,
                    instruct_dir: Path | None = None) -> str:
    prev_num = chapter_num - 1
    _dir = instruct_dir or (ROOT / "outputs" / "instruct")
    # 优先：当前模式的输出目录
    gen_path = _dir / f"chapter_{prev_num:03d}.txt"
    if gen_path.exists():
        text = gen_path.read_text(encoding="utf-8")
        text = re.sub(r"要知后事[，,]下回分解[。.]?\s*(\(本章完\))?", "", text)
        return text.strip()[-tail_chars:]
    # 回退：data/chapters/chap_0XX.txt（仅前80回有）
    orig_path = ROOT / "data" / "chapters" / f"chap_{prev_num:03d}.txt"
    if orig_path.exists():
        text = orig_path.read_text(encoding="utf-8")
        text = re.sub(r"要知后事[，,]下回分解[。.]?\s*(\(本章完\))?", "", text)
        return text.strip()[-tail_chars:]
    return ""


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
6. 【严禁】"欲知后事如何，且听下回分解""且听下回""正是：""要知后事"等一切章回结尾套语，只在被明确要求写"本回最后一个场景"时才可使用。写中间场景时，绝对不得出现任何结尾套语，直接以情节正文收住即可。"""


# ─────────────────────────────────────────────────────────────
# 章回结尾套语清理（用于非最后场景）
# ─────────────────────────────────────────────────────────────
# 匹配所有常见变体，从第一个触发词所在行的行首截断
_ENDING_TRIGGERS = re.compile(
    r"\n[　 ]*(?:"
    r"[欲未]知后事|"          # 欲知后事 / 未知后事
    r"且[听看]下回|"           # 且听下回 / 且看下回
    r"正是[：:]|"              # 正是：（对联引导）
    r"欲[知晓]详情|"           # 其他变体
    r"要知后事"                # 要知后事
    r")"
)

def strip_chapter_ending(text: str) -> str:
    """截掉场景末尾误生成的章回结尾套语（含'正是：'对联块）。"""
    m = _ENDING_TRIGGERS.search(text)
    if not m:
        return text.rstrip()
    cut = m.start()
    # 如果"正是："出现在"欲知后事"之前，也一并截掉
    earlier = _ENDING_TRIGGERS.search(text[:cut]) if cut > 0 else None
    if earlier:
        cut = earlier.start()
    return text[:cut].rstrip()


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
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
def generate_whole(model, tokenizer, outline: dict, prev_ending: str,
                   chapter_num: int, max_new_tokens: int) -> str:
    title = outline["title"]
    user_prompt = f"""请续写《红楼梦》{title}。

【上一回（第{chapter_num-1}回）结尾】
{prev_ending}

【本回大纲】
章回标题：{title}
核心情节：
{outline['plot_raw']}
主要人物：{outline['characters']}
关键场景：{outline['scenes']}
情感基调：{outline['tone']}

请按章回体格式，从章回标题开始，写出完整的一回正文（约2000字）。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    print("  [整章生成...]")
    return chat_generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)


# ─────────────────────────────────────────────────────────────
# 分场景多轮生成
# ─────────────────────────────────────────────────────────────
def generate_by_scenes(model, tokenizer, outline: dict, prev_ending: str,
                       chapter_num: int, tokens_per_scene: int) -> str:
    title = outline["title"]
    plot_items = outline["plot_items"]
    n = len(plot_items)
    print(f"  [分场景生成，共 {n} 个场景]")

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
            last_note = (
                "（这是本回最后一个场景，结尾处用「欲知后事如何，且听下回分解。」收束）"
                if is_last else
                "（场景正文即可，不需要写章回结尾套语，下一段会继续）"
            )
            user_content = f"""继续写下一个场景（约500字）：
场景：{scene}
{last_note}"""

        messages.append({"role": "user", "content": user_content})
        print(f"    场景 {i+1}/{n}：{scene[:40]}...")

        scene_text = chat_generate(model, tokenizer, messages, max_new_tokens=tokens_per_scene)
        print(f"    → 生成 {len(scene_text)} 字")

        # 非最后场景去掉误生成的章回结尾套语（各种变体）
        if not is_last:
            scene_text = strip_chapter_ending(scene_text)

        messages.append({"role": "assistant", "content": scene_text})
        full_text += scene_text + "\n"

    return full_text.strip()


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="批量生成后40回（大纲+instruct模式）")
    parser.add_argument("--scenes",   action="store_true",
                        help="分场景多轮生成（默认：整章一次生成）")
    parser.add_argument("--lora_checkpoint", default=None,
                        help="LoRA checkpoint 路径（如 outputs/checkpoint-best）；"
                             "不传则使用纯基座模型")
    parser.add_argument("--start",    type=int, default=81,
                        help="从第几回开始生成（默认81）")
    parser.add_argument("--end",      type=int, default=120,
                        help="生成到第几回（默认120）")
    parser.add_argument("--chapters", type=int, nargs="+",
                        help="只生成指定章回（覆盖 --start/--end）")
    parser.add_argument("--overwrite", action="store_true",
                        help="强制重新生成（默认跳过已存在文件）")
    parser.add_argument("--max_new_tokens",   type=int, default=2000,
                        help="整章模式最大 token（默认2000）")
    parser.add_argument("--tokens_per_scene", type=int, default=600,
                        help="分场景每场景 token（默认600）")
    args = parser.parse_args()

    # 根据是否使用 LoRA 决定输出目录（统一放在 generations/）
    if args.lora_checkpoint:
        # 用 checkpoint 上两级目录名作为标识（如 lora_20260226_1801_sw）
        ckpt = Path(args.lora_checkpoint)
        # best/ 或 checkpoint-N/ 都往上一级取运行目录名
        if ckpt.name == "best" or ckpt.name.startswith("checkpoint-"):
            run_name = ckpt.parent.name
        else:
            run_name = ckpt.name
        output_dir = ROOT / "generations" / run_name
    else:
        output_dir = ROOT / "generations" / "base_qwen3_8b"
    output_dir.mkdir(parents=True, exist_ok=True)

    chapters = args.chapters if args.chapters else list(range(args.start, args.end + 1))
    mode = "分场景" if args.scenes else "整章"
    backend = f"LoRA({Path(args.lora_checkpoint).name})" if args.lora_checkpoint else "基座"
    print(f"\n{'='*60}")
    print(f"后40回批量生成  模式：{mode}  后端：{backend}  共 {len(chapters)} 回")
    print(f"输出目录：{output_dir}")
    print(f"{'='*60}")

    # 检查哪些已存在
    to_generate = []
    for ch in chapters:
        out_path = output_dir / f"chapter_{ch:03d}.txt"
        if out_path.exists() and not args.overwrite:
            print(f"  第{ch:3d}回 → 已存在，跳过（用 --overwrite 强制重新生成）")
        else:
            to_generate.append(ch)

    if not to_generate:
        print("\n所有章回均已生成完毕。")
        return

    print(f"\n待生成：{len(to_generate)} 回  {to_generate[0]}-{to_generate[-1]}")

    # 加载模型
    if args.lora_checkpoint:
        model, tokenizer = load_lora_model(args.lora_checkpoint)
    else:
        model, tokenizer = load_base_model()

    total = len(to_generate)
    t_start_all = time.time()

    for idx, ch in enumerate(to_generate, 1):
        out_path = output_dir / f"chapter_{ch:03d}.txt"
        print(f"\n{'─'*60}")
        print(f"[{idx}/{total}] 第 {ch} 回")

        # 解析大纲
        try:
            outline = parse_outline(ch)
        except ValueError as e:
            print(f"  !! 大纲解析失败：{e}，跳过")
            continue

        print(f"  标题：{outline['title']}")
        print(f"  情节：{len(outline['plot_items'])} 条")

        # 上一回结尾（优先读当前输出目录的生成结果）
        prev_ending = get_prev_ending(ch, instruct_dir=output_dir)
        source = "生成结果" if (output_dir / f"chapter_{ch-1:03d}.txt").exists() else "原著数据"
        print(f"  上一回结尾来源：{source}（{len(prev_ending)} 字）")

        # 生成
        t0 = time.time()
        if args.scenes:
            text = generate_by_scenes(
                model, tokenizer, outline, prev_ending, ch,
                tokens_per_scene=args.tokens_per_scene,
            )
        else:
            text = generate_whole(
                model, tokenizer, outline, prev_ending, ch,
                max_new_tokens=args.max_new_tokens,
            )
        elapsed = time.time() - t0

        # 保存
        out_path.write_text(text, encoding="utf-8")
        char_count = len(text)
        print(f"  完成：{char_count} 字  用时 {elapsed:.0f}s  → {out_path.name}")

    total_time = time.time() - t_start_all
    print(f"\n{'='*60}")
    print(f"全部完成！共 {len(to_generate)} 回，总用时 {total_time/60:.1f} 分钟")
    print(f"输出目录：{output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
