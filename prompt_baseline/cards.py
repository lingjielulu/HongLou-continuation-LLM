"""Character card loading and compression."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ALIASES = {
    "宝玉": "贾宝玉",
    "黛玉": "林黛玉",
    "宝钗": "薛宝钗",
    "凤姐": "王熙凤",
    "熙凤": "王熙凤",
    "探春": "贾探春",
    "迎春": "贾迎春",
    "惜春": "贾惜春",
    "元春": "贾元春",
    "湘云": "史湘云",
    "李纨": "李纨",
    "贾母": "贾母",
    "王夫人": "王夫人",
    "薛姨妈": "薛姨妈",
    "紫鹃": "紫鹃",
    "平儿": "平儿",
    "袭人": "袭人",
    "妙玉": "妙玉",
    "巧姐": "巧姐",
    "香菱": "香菱",
    "贾政": "贾政",
    "贾琏": "贾琏",
    "贾赦": "贾赦",
    "邢夫人": "邢夫人",
}


@dataclass(frozen=True)
class CharacterCard:
    name: str
    character_id: str
    path: Path
    text: str


def _extract_character_id(text: str) -> str:
    match = re.search(r"- `character_id`:\s*`([^`]+)`", text)
    return match.group(1) if match else ""


def load_cards(card_dir: Path) -> dict[str, CharacterCard]:
    cards: dict[str, CharacterCard] = {}
    for path in sorted(card_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        heading = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), "")
        if not heading:
            continue
        cards[heading] = CharacterCard(
            name=heading,
            character_id=_extract_character_id(text),
            path=path,
            text=text,
        )
    return cards


def normalize_character_name(raw_name: str) -> str:
    name = re.sub(r"[（(].*?[）)]", "", raw_name).strip()
    name = name.strip(" 、，,。；;")
    return ALIASES.get(name, name)


def names_from_outline(characters: str) -> list[str]:
    names: list[str] = []
    for part in re.split(r"[、,，；;]\s*", characters):
        name = normalize_character_name(part)
        if name and name not in names:
            names.append(name)
    return names


def _section(text: str, title: str) -> str:
    pattern = rf"(^## {re.escape(title)}\n.*?)(?=^## |\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def compact_card(card: CharacterCard, limit: int = 1800) -> str:
    parts = [f"## {card.name}"]
    for title in ("人物定位", "核心深描", "关系总览", "阶段变化", "易错点与生成提醒"):
        section = _section(card.text, title)
        if section:
            parts.append(section)
    compact = "\n\n".join(parts)
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "\n...[已截断]"
