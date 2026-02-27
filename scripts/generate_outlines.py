"""
为前 80 回原著生成结构化大纲（用于后续 instruct 格式训练）

输出：outline/前80回大纲/chap_001.json ... chap_080.json
格式：
{
  "chapter": 1,
  "title": "第一回　甄士隐梦幻识通灵，贾雨村风尘怀闺秀",
  "outline_text": "...(markdown格式大纲)...",
  "核心情节": ["...", "..."],
  "主要人物": "...",
  "关键场景": "...",
  "情感基调": "...",
  "叙事功能": "..."
}

用法：
    conda run -n stone python3 scripts/generate_outlines.py --all
    conda run -n stone python3 scripts/generate_outlines.py --chapters 1-10
    conda run -n stone python3 scripts/generate_outlines.py --chapters 5,8,12
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT          = Path(__file__).parent.parent
CHAPTERS_DIR  = ROOT / "data" / "chapters"
OUTLINE_DIR   = ROOT / "outline" / "前80回大纲"

LOCAL_MODEL_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

SYSTEM_PROMPT = """你是《红楼梦》研究专家，精通曹雪芹的叙事结构与写作手法。
你的任务是：阅读给定的《红楼梦》某回原文，提取其结构化大纲。

请严格按照以下 JSON 格式输出，不要输出其他任何内容：
{
  "title": "第X回　回目标题（保持原文）",
  "核心情节": [
    "情节要点1（一句话）",
    "情节要点2（一句话）",
    "情节要点3（一句话）"
  ],
  "主要人物": "人物1、人物2、人物3（逗号分隔）",
  "关键场景": "场景1、场景2（逗号分隔）",
  "情感基调": "一句话描述本回的情感氛围",
  "叙事功能": "本回在全书中的叙事作用（伏笔/回收/推进情节等）"
}

要求：
- 核心情节：3-6 个要点，每条15-40字，涵盖本回主要事件
- 主要人物：列出本回出场并有实质戏份的人物
- 关键场景：列出本回最重要的2-4个场景地点或情境
- 情感基调：用8-20字精准概括（如"喜中带忧，热闹背后的寒意"）
- 叙事功能：用15-40字说明本回在全书结构中的作用
- 只输出 JSON，不要有任何前缀或后缀文字"""

EXTRACT_PROMPT_TEMPLATE = """以下是《红楼梦》第{chap_no}回原文（节选约前3000字），请提取结构化大纲：

{chapter_text}

请严格按要求输出 JSON 格式大纲。"""


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "Qwen/Qwen3-8B-FP8"
    print(f"  加载模型：{model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_json(text: str) -> dict | None:
    """从模型输出中提取 JSON 对象"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 找到第一个 { ... } 块
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    snippet = None
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        snippet = text[brace_start:brace_end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # 尝试清理后再解析（去除控制字符等）
    source = snippet if snippet is not None else text
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", source)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def generate_outline(model, tokenizer, chap_no: int) -> dict | None:
    """为指定章节生成大纲"""
    import torch

    chap_file = CHAPTERS_DIR / f"chap_{chap_no:03d}.txt"
    if not chap_file.exists():
        print(f"  ⚠ 章节文件不存在：{chap_file.name}，跳过")
        return None

    chapter_text = chap_file.read_text(encoding="utf-8")
    # 只取前 ~3000 字（约 1500 tokens），避免 context 过长
    chapter_excerpt = chapter_text[:3000]

    user_content = EXTRACT_PROMPT_TEMPLATE.format(
        chap_no=chap_no,
        chapter_text=chapter_excerpt,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=800,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    result = extract_json(raw_output)
    if result is None:
        print(f"  ⚠ 第{chap_no}回：JSON 解析失败，保存原始输出")
        print(f"  原始输出（前300字）：{raw_output[:300]}")
        return {"chapter": chap_no, "raw_output": raw_output, "parse_error": True}

    result["chapter"] = chap_no
    return result


def parse_chapter_range(spec: str) -> list[int]:
    """解析章节范围规格，支持 '1-10' 或 '5,8,12' 或 '1-5,10,15-20'"""
    chapters = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            chapters.extend(range(int(a.strip()), int(b.strip()) + 1))
        else:
            chapters.append(int(part))
    return sorted(set(chapters))


def main():
    parser = argparse.ArgumentParser(description="为前80回原著生成结构化大纲")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",      action="store_true", help="处理全部第1-80回")
    group.add_argument("--chapters", type=str,            help="指定章节，如 '1-10' 或 '5,8,12'")
    parser.add_argument("--force",   action="store_true", help="强制重新生成（覆盖已有文件）")
    args = parser.parse_args()

    if args.all:
        chapters = list(range(1, 81))
    else:
        chapters = parse_chapter_range(args.chapters)
        chapters = [c for c in chapters if 1 <= c <= 80]
        if not chapters:
            print("错误：指定的章节范围超出 1-80 范围")
            sys.exit(1)

    OUTLINE_DIR.mkdir(parents=True, exist_ok=True)

    # 检查跳过
    to_process = []
    skipped = 0
    for c in chapters:
        out_path = OUTLINE_DIR / f"chap_{c:03d}.json"
        if out_path.exists() and not args.force:
            skipped += 1
        else:
            to_process.append(c)

    print(f"计划处理：{len(chapters)} 回")
    if skipped:
        print(f"  已跳过（已有文件）：{skipped} 回，使用 --force 强制重新生成")
    print(f"  待生成：{len(to_process)} 回")

    if not to_process:
        print("无需生成，退出。")
        return

    print("\n加载模型...")
    model, tokenizer = load_model()
    print("模型加载完成\n")

    ok_count    = 0
    err_count   = 0
    parse_errs  = []

    for i, chap_no in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] 第{chap_no}回...", end=" ", flush=True)
        try:
            result = generate_outline(model, tokenizer, chap_no)
            if result is None:
                err_count += 1
                continue

            out_path = OUTLINE_DIR / f"chap_{chap_no:03d}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if result.get("parse_error"):
                parse_errs.append(chap_no)
                print(f"⚠ JSON解析失败（已保存原始）")
            else:
                title = result.get("title", "")
                print(f"✓  {title[:30]}")
                ok_count += 1

        except Exception as e:
            print(f"✗ 错误：{e}")
            err_count += 1

    print(f"\n=== 完成 ===")
    print(f"  成功：{ok_count} 回")
    if parse_errs:
        print(f"  JSON解析失败（已保存原始）：{len(parse_errs)} 回 → {parse_errs}")
    if err_count:
        print(f"  错误跳过：{err_count} 回")
    print(f"  输出目录：{OUTLINE_DIR}")


if __name__ == "__main__":
    main()
