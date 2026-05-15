"""Outline parsing helpers for the prompt-only baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


CN_MAP = {
    81: "八十一",
    82: "八十二",
    83: "八十三",
    84: "八十四",
    85: "八十五",
    86: "八十六",
    87: "八十七",
    88: "八十八",
    89: "八十九",
    90: "九十",
    91: "九十一",
    92: "九十二",
    93: "九十三",
    94: "九十四",
    95: "九十五",
    96: "九十六",
    97: "九十七",
    98: "九十八",
    99: "九十九",
    100: "一百",
    101: "一百零一",
    102: "一百零二",
    103: "一百零三",
    104: "一百零四",
    105: "一百零五",
    106: "一百零六",
    107: "一百零七",
    108: "一百零八",
    109: "一百零九",
    110: "一百一十",
    111: "一百一十一",
    112: "一百一十二",
    113: "一百一十三",
    114: "一百一十四",
    115: "一百一十五",
    116: "一百一十六",
    117: "一百一十七",
    118: "一百一十八",
    119: "一百一十九",
    120: "一百二十",
}


@dataclass(frozen=True)
class ChapterOutline:
    chapter: int
    title: str
    plot_raw: str
    plot_items: list[str]
    characters: str
    scenes: str
    tone: str
    function: str


@dataclass(frozen=True)
class FateEvent:
    character: str
    verdict: str
    outcome: str
    chapter_text: str
    start_chapter: int | None
    end_chapter: int | None


def _extract_field(block: str, label: str) -> str:
    fm = re.search(rf"\*\*{label}[：:]\*\*\s*\n(.*?)(?=\n\*\*|\Z)", block, re.DOTALL)
    if fm:
        return fm.group(1).strip()
    fm = re.search(rf"\*\*{label}[：:]\*\*\s*(.+)", block)
    if fm:
        return fm.group(1).strip()
    fm = re.search(rf"{label}[：:]\s*(.+)", block)
    return fm.group(1).strip() if fm else ""


def parse_outline(outline_path: Path, chapter: int) -> ChapterOutline:
    text = outline_path.read_text(encoding="utf-8")
    cn_num = CN_MAP.get(chapter, str(chapter))
    pattern = rf"###\s*第{cn_num}回[^\n]*\n(.*?)(?=###|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"未找到第 {chapter} 回大纲：{outline_path}")

    block = match.group(0)
    title = block.splitlines()[0].replace("###", "").strip()
    plot_raw = _extract_field(block, "核心情节")
    plot_items = [
        line.lstrip("- ").strip()
        for line in plot_raw.splitlines()
        if line.strip().startswith("-")
    ] or ([plot_raw] if plot_raw else [])

    return ChapterOutline(
        chapter=chapter,
        title=title,
        plot_raw=plot_raw,
        plot_items=plot_items,
        characters=_extract_field(block, "主要人物"),
        scenes=_extract_field(block, "关键场景"),
        tone=_extract_field(block, "情感基调"),
        function=_extract_field(block, "叙事功能"),
    )


def parse_fate_events(outline_path: Path) -> list[FateEvent]:
    text = outline_path.read_text(encoding="utf-8")
    events: list[FateEvent] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| 人物 | 判词关键句 | 结局 | 对应章回 |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if events:
                break
            continue
        if set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        chapter_numbers = [int(value) for value in re.findall(r"\d+", cells[3])]
        start_chapter = chapter_numbers[0] if chapter_numbers else None
        end_chapter = chapter_numbers[-1] if chapter_numbers else None
        events.append(
            FateEvent(
                character=cells[0],
                verdict=cells[1],
                outcome=cells[2],
                chapter_text=cells[3],
                start_chapter=start_chapter,
                end_chapter=end_chapter,
            )
        )
    return events


def active_fate_events(outline_path: Path, chapter: int) -> list[FateEvent]:
    return [
        event
        for event in parse_fate_events(outline_path)
        if event.end_chapter is not None and event.end_chapter < chapter
    ]
