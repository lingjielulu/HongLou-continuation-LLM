#!/usr/bin/env python3
"""Generate a prompt-only baseline chapter.

Examples:
    python scripts/prompt_baseline_generate.py --chapter 81 --dry-run
    python scripts/prompt_baseline_generate.py --chapter 81 --model deepseek-chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompt_baseline.cards import load_cards
from prompt_baseline.client import chat_completion, default_base_url, default_model
from prompt_baseline.env import load_project_env
from prompt_baseline.outline import active_fate_events, parse_outline
from prompt_baseline.prompt import (
    PromptContext,
    build_messages,
    read_previous_ending,
    render_prompt_markdown,
)


def main() -> None:
    load_project_env(ROOT)

    parser = argparse.ArgumentParser(
        description="Prompt-only 红楼梦章回 baseline 生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chapter", type=int, required=True, help="章回号，如 81")
    parser.add_argument("--model", default=default_model(), help="Chat Completions 模型名")
    parser.add_argument("--outline", type=Path, default=ROOT / "outline" / "后40回大纲.md")
    parser.add_argument("--card-dir", type=Path, default=ROOT / "Hongloumeng_card" / "cards")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "chapters")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generations" / "prompt_baseline")
    parser.add_argument("--tail-chars", type=int, default=800, help="上一回结尾注入字数")
    parser.add_argument("--card-limit", type=int, default=1800, help="每个人物 card 最大字符数")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--dry-run", action="store_true", help="只生成 prompt，不调用模型")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有正文输出")
    args = parser.parse_args()

    outline = parse_outline(args.outline, args.chapter)
    cards = load_cards(args.card_dir)
    previous_ending = read_previous_ending(
        args.chapter,
        baseline_dir=args.output_dir,
        data_dir=args.data_dir,
        tail_chars=args.tail_chars,
    )
    context = PromptContext(
        outline=outline,
        previous_ending=previous_ending,
        fate_events=active_fate_events(args.outline, args.chapter),
        cards=cards,
        card_limit=args.card_limit,
    )
    messages = build_messages(context)

    prompt_dir = args.output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"chapter_{args.chapter:03d}.md"
    prompt_path.write_text(render_prompt_markdown(messages), encoding="utf-8")
    print(f"Prompt 已保存：{prompt_path}")

    if args.dry_run:
        print("Dry-run 模式：不调用模型。")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"chapter_{args.chapter:03d}.txt"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"输出已存在：{output_path}；如需重写请加 --overwrite")

    print(f"调用模型：{args.model}")
    print(f"接口地址：{(args.base_url or default_base_url()).rstrip('/')}/chat/completions")
    content = chat_completion(
        messages,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"正文已保存：{output_path}（{len(content)} 字）")


if __name__ == "__main__":
    main()
