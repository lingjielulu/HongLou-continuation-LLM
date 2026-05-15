"""Prompt construction for the prompt-only baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .cards import CharacterCard, compact_card, names_from_outline
from .outline import ChapterOutline, FateEvent


SYSTEM_PROMPT = """你是一位严谨的《红楼梦》续写作家。你必须只输出小说正文，不输出解释、提纲、注释、分析或创作说明。

写作硬约束：
1. 使用章回体小说语言，半文半白，尽量贴近《红楼梦》前八十回叙述方式。
2. 以给定大纲为叙事骨架，不擅自改写重大命运节点。
3. 严格遵守世界状态：已逝人物不得主动说话、行动或参与新事件；远嫁、出家、失踪人物不得无交代回到现场。
4. 已逝人物只能以他人回忆、梦兆、诗词意象、旧物联想、祭奠文字等形式出现。
5. 人物说话、举止、关系必须服从人物 card；不要把人物写成现代心理分析。
6. 本次生成的最低合格线是人物不严重 OOC、情节不出现大 bug；文风可以含蓄铺叙，但不能牺牲人物与情节一致性。
7. 每个自然段开头用两个全角空格缩进。
8. 除非明确写到本回最后收束，不得使用“欲知后事如何，且听下回分解”等结尾套语。"""


@dataclass(frozen=True)
class PromptContext:
    outline: ChapterOutline
    previous_ending: str
    fate_events: list[FateEvent]
    cards: dict[str, CharacterCard]
    card_limit: int = 1800


def clean_previous_ending(text: str) -> str:
    text = re.sub(r"要知后事[，,]下回分解[。.]?\s*(\(本章完\))?", "", text)
    text = re.sub(r"\(本章完\)", "", text)
    return text.strip()


def read_previous_ending(
    chapter: int,
    baseline_dir: Path,
    data_dir: Path,
    tail_chars: int = 800,
) -> str:
    prev = chapter - 1
    generated = baseline_dir / f"chapter_{prev:03d}.txt"
    source = generated if generated.exists() else data_dir / f"chap_{prev:03d}.txt"
    if not source.exists():
        return ""
    return clean_previous_ending(source.read_text(encoding="utf-8"))[-tail_chars:]


def _format_fate_events(events: list[FateEvent]) -> str:
    if not events:
        return "截至本回前，命运总表中暂无已兑现的不可逆结局。"
    lines = []
    for event in events:
        lines.append(
            f"- {event.character}：{event.outcome}（{event.chapter_text}；判词：{event.verdict}）"
        )
    return "\n".join(lines)


def _format_cards(outline: ChapterOutline, cards: dict[str, CharacterCard], limit: int) -> str:
    blocks: list[str] = []
    missing: list[str] = []
    for name in names_from_outline(outline.characters):
        card = cards.get(name)
        if card is None:
            missing.append(name)
            continue
        blocks.append(compact_card(card, limit=limit))
    if missing:
        blocks.append("## 未找到人物 card\n" + "、".join(missing))
    return "\n\n---\n\n".join(blocks) if blocks else "本回未解析出可注入的人物 card。"


def build_user_prompt(context: PromptContext) -> str:
    outline = context.outline
    plot_items = "\n".join(f"- {item}" for item in outline.plot_items)
    return f"""请续写《红楼梦》{outline.title}。

【上一回结尾】
{context.previous_ending or "（未提供上一回结尾）"}

【本回大纲】
章回标题：{outline.title}
核心情节：
{plot_items or outline.plot_raw}
主要人物：{outline.characters}
关键场景：{outline.scenes}
情感基调：{outline.tone}
叙事功能：{outline.function}

【截至本回前的世界状态】
{_format_fate_events(context.fate_events)}

【本回人物 card 摘要】
{_format_cards(outline, context.cards, context.card_limit)}

【写作任务】
从章回标题开始，写出第 {outline.chapter} 回正文。最低目标是人物不严重 OOC、情节不出现大 bug；重点是让情节、人物状态、人物关系符合上面的世界状态与人物 card。正文可以保留古典章回小说的铺叙和含蓄，但不得输出任何现代说明或分析。"""


def build_messages(context: PromptContext) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context)},
    ]


def render_prompt_markdown(messages: list[dict[str, str]]) -> str:
    rendered = []
    for message in messages:
        rendered.append(f"## {message['role']}\n\n{message['content']}")
    return "\n\n---\n\n".join(rendered).strip() + "\n"
