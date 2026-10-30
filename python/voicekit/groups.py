"""Word groups an LLM writes for a wake phrase.

A group is a named prompt plus the label its items get in the dataset. New
groups are added to ``GROUPS``; the UI lists whatever this module reports.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from .base import LlmEngine
from .common import VoiceError, log
from .languages import language_name, matches_script

MAX_ITEM_CHARS = 60
MAX_ITEM_WORDS = 8
# Ask for small batches: short answers stay varied and fast on small models.
BATCH_SIZE = 15
TOKENS_PER_ITEM = 16


@dataclass(frozen=True)
class WordGroup:
    id: str
    title: str
    description: str
    label: str  # "negative" | "positive"
    default_count: int
    instructions: str


GROUPS: tuple[WordGroup, ...] = (
    WordGroup(
        id="confusable",
        title="Similar-sounding",
        description="Real words and short phrases that sound close to the wake phrase.",
        label="negative",
        default_count=30,
        instructions=(
            "List real words and short phrases (1-3 words) that SOUND similar to the wake "
            "phrase: rhymes, shared syllables, the same stressed vowel, similar consonants, "
            "other grammatical forms of its words. Each must be something a person could "
            "actually say, and none may be the wake phrase itself."
        ),
    ),
    WordGroup(
        id="partial",
        title="Partial overlap",
        description="Phrases that reuse a part of the wake phrase but never all of it.",
        label="negative",
        default_count=20,
        instructions=(
            "List short phrases (1-4 words) that reuse only a PART of the wake phrase: one of "
            "its words, its first or last syllables, or a word that starts or ends the same "
            "way, combined with other words. Never include the complete wake phrase."
        ),
    ),
    WordGroup(
        id="everyday",
        title="Everyday speech",
        description="Ordinary things people say at home, unrelated to the wake phrase.",
        label="negative",
        default_count=30,
        instructions=(
            "List varied everyday phrases (2-6 words) people say at home or at work: "
            "questions, requests, remarks about weather, food, time, devices, family. They "
            "must not contain or resemble the wake phrase."
        ),
    ),
)

BY_ID = {group.id: group for group in GROUPS}

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    "required": ["items"],
    "additionalProperties": False,
}


def groups_payload() -> list[dict]:
    return [
        {
            "id": group.id,
            "title": group.title,
            "description": group.description,
            "label": group.label,
            "defaultCount": group.default_count,
        }
        for group in GROUPS
    ]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(item: str, phrase: str) -> bool:
    """True when the wake phrase appears in ``item`` as a whole-word sequence."""
    item_tokens = normalize(item).split()
    phrase_tokens = normalize(phrase).split()
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(item_tokens[i : i + width] == phrase_tokens for i in range(len(item_tokens) - width + 1))


def clean_items(raw: list[str], phrase: str, language: str, seen: set[str]) -> tuple[list[str], dict[str, int]]:
    """Filter an LLM answer. Returns kept items and counts of dropped ones by reason."""
    kept: list[str] = []
    dropped = {"duplicate": 0, "contains_phrase": 0, "wrong_script": 0, "too_long": 0}
    for value in raw:
        if not isinstance(value, str):
            continue
        item = re.sub(r"\s+", " ", value).strip().strip("\"'«»„“”.,;:!?-").strip()
        if not item:
            continue
        key = normalize(item)
        if not key or key in seen:
            dropped["duplicate"] += 1
            continue
        if len(item) > MAX_ITEM_CHARS or len(key.split()) > MAX_ITEM_WORDS:
            dropped["too_long"] += 1
            continue
        if contains_phrase(item, phrase):
            # Speech that contains the whole wake phrase would legitimately
            # trigger a sliding-window detector: it is not a negative.
            dropped["contains_phrase"] += 1
            continue
        if not matches_script(item, language):
            dropped["wrong_script"] += 1
            continue
        seen.add(key)
        kept.append(item)
    return kept, dropped


def _parse_items(text: str) -> list:
    """Read the items list; salvage complete strings from a truncated answer."""
    text = text.strip()
    # Tolerate code fences or prose around the JSON object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except ValueError:
        start = text.find("[")
        if start < 0:
            raise VoiceError(f"The language model did not return JSON: {text[:200]!r}") from None
        # An answer cut off by max_tokens still holds valid leading items.
        items = [json.loads(f'"{value}"') for value in re.findall(r'"((?:[^"\\]|\\.)*)"', text[start:])]
        log(f"Recovered {len(items)} items from a truncated answer")
        return items
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise VoiceError("The language model returned JSON without an 'items' list.")
    return items


def _messages(group: WordGroup, phrase: str, language: str, count: int, avoid: list[str]) -> list[dict]:
    language_label = language_name(language)
    avoid_text = ", ".join(avoid[:60]) if avoid else "none"
    system = (
        "You write training data for an on-device wake-word detector. The detector must "
        "react ONLY to the exact wake phrase. Your lists become negative examples that teach "
        "it what to ignore. Answer with JSON only."
    )
    user = (
        f"Wake phrase: \"{phrase}\"\n"
        f"Language: {language_label} ({language}). Write every item in {language_label}, in its "
        f"normal script, with natural spelling.\n"
        f"Task: {group.instructions}\n"
        f"Return {count} distinct items as {{\"items\": [...]}}.\n"
        f"Do not repeat any of these: {avoid_text}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate(
    llm: LlmEngine,
    phrase: str,
    language: str,
    requests: dict[str, int],
    existing: dict[str, list[str]] | None = None,
    on_group=None,
) -> dict[str, dict]:
    """Generate every requested group. ``existing`` items are kept out of new answers."""
    existing = existing or {}
    seen = {normalize(item) for items in existing.values() for item in items}
    seen.add(normalize(phrase))
    results: dict[str, dict] = {}
    for index, (group_id, count) in enumerate(requests.items()):
        group = BY_ID.get(group_id)
        if group is None:
            raise VoiceError(f"Unknown word group '{group_id}'.")
        if on_group:
            on_group(index, group, 0, count)
        kept: list[str] = []
        dropped_total: dict[str, int] = {}
        # Small batches until the count is reached; give up after a few rounds
        # that add nothing new (the model ran out of ideas).
        max_rounds = -(-count // BATCH_SIZE) + 3
        empty_rounds = 0
        for round_index in range(max_rounds):
            missing = count - len(kept)
            if missing <= 0 or empty_rounds >= 2:
                break
            ask = min(BATCH_SIZE, missing) + 3
            answer = llm.chat(
                _messages(group, phrase, language, ask, kept + existing.get(group_id, [])),
                schema=SCHEMA,
                temperature=min(1.2, 0.9 + 0.1 * round_index),
                max_tokens=ask * TOKENS_PER_ITEM + 64,
            )
            items, dropped = clean_items(_parse_items(answer), phrase, language, seen)
            kept.extend(items[:missing])
            empty_rounds = empty_rounds + 1 if not items else 0
            for reason, value in dropped.items():
                dropped_total[reason] = dropped_total.get(reason, 0) + value
            if on_group:
                on_group(index, group, len(kept), count)
        log(f"{group.title}: {len(kept)} items (dropped {json.dumps(dropped_total)})")
        results[group_id] = {"items": kept, "dropped": dropped_total}
    return results
