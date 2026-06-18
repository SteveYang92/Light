"""LLM merge-hint review — second pass per translation batch."""

from __future__ import annotations

import json
import re

from light_models import Segment

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient
from ...llm.prompts import render_prompt

_MERGE_REVIEW_RETRIES = 3
_SNIP_LIMIT = 20
_CLOSURE_GAP_FALSE_MS = 800
_CLOSURE_ENDS = "。！？"
_SEVERELY_DANGLING_ENDS = "的地得"
_OPEN_END_SUFFIXES = ("……", "…", "——", "—")

MergeHint = tuple[Segment, Segment, str, str]


def _render_merge_review_prompt() -> str:
    return render_prompt("merge_review.j2", gap_closure_false_ms=_CLOSURE_GAP_FALSE_MS)


def _ends_with_closure(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _CLOSURE_ENDS


def _is_severely_dangling(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in _SEVERELY_DANGLING_ENDS:
        return True
    return any(stripped.endswith(suffix) for suffix in _OPEN_END_SUFFIXES)


def _apply_gap_filters(
    flags: dict[int, bool],
    segments: list[Segment],
    parsed_texts: dict[int, str],
) -> dict[int, bool]:
    """Force ``false`` when closure or long gap rules apply."""
    result = dict(flags)
    for idx, merge in result.items():
        if not merge:
            continue
        text = parsed_texts[idx]
        if _ends_with_closure(text):
            result[idx] = False
            logger.info(f"  Merge skipped (closure): {segments[idx].unit_id} | '{_snip_text(text)}'")
            continue
        gap = _gap_to_next_ms(segments, idx)
        if gap is None or gap < _CLOSURE_GAP_FALSE_MS:
            continue
        if not _is_severely_dangling(text):
            result[idx] = False
    return result


def _snip_text(text: str, limit: int = _SNIP_LIMIT) -> str:
    flat = text.replace("\n", "").strip()
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _parse_merge_review_response(response: str, n: int) -> dict[int, bool]:
    """Parse merge review JSON into ``batch_index → merge_with_next``."""
    json_match = re.search(r"\[([\s\S]*)\]", response)
    if json_match:
        data = json.loads(json_match.group(0))
    else:
        data = json.loads(response)

    if not isinstance(data, list):
        raise ValueError("Merge review response is not a JSON array")

    expected = set(range(n))
    by_index: dict[int, bool] = {}
    for item in data:
        if not isinstance(item, dict) or "batch_index" not in item:
            continue
        try:
            idx = int(item["batch_index"])
        except (TypeError, ValueError):
            continue
        if idx not in expected:
            continue
        if idx in by_index:
            raise ValueError(f"Duplicate batch_index in merge review: {idx}")
        by_index[idx] = bool(item.get("merge_with_next", False))

    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        if missing:
            for idx in missing:
                by_index[idx] = False
            logger.warning(f"  Merge review incomplete: missing {missing}, defaulting to merge_with_next=false")
        if extra:
            raise ValueError(f"Merge review incomplete: missing {missing}, unexpected {extra}")

    return by_index


def _gap_to_next_ms(segments: list[Segment], idx: int) -> int | None:
    """Milliseconds from ``segments[idx].end`` to ``segments[idx + 1].start``."""
    if idx >= len(segments) - 1:
        return None
    return round((segments[idx + 1].start - segments[idx].end) * 1000)


def _build_review_units(segments: list[Segment], parsed_texts: dict[int, str]) -> list[dict[str, int | str | None]]:
    units: list[dict[str, int | str | None]] = []
    for idx in range(len(segments)):
        unit: dict[str, int | str | None] = {
            "batch_index": idx,
            "text": parsed_texts[idx],
        }
        gap = _gap_to_next_ms(segments, idx)
        if gap is not None:
            unit["gap_to_next_ms"] = gap
        units.append(unit)
    return units


def _hints_from_flags(
    segments: list[Segment],
    parsed_texts: dict[int, str],
    flags: dict[int, bool],
) -> list[MergeHint]:
    hints: list[MergeHint] = []
    for idx in range(len(segments) - 1):
        if not flags.get(idx, False):
            continue
        hints.append(
            (
                segments[idx],
                segments[idx + 1],
                parsed_texts[idx],
                parsed_texts[idx + 1],
            )
        )
    return hints


def review_merge_hints(
    client: OpenAIClient,
    segments: list[Segment],
    parsed_texts: dict[int, str],
    config: SubtitleConfig,
) -> list[MergeHint]:
    """Run LLM merge review on one batch of translated texts."""
    if len(segments) <= 1:
        return []

    system_prompt = _render_merge_review_prompt()
    max_true = max(1, (len(segments) + 23) // 24)
    payload = {
        "max_merge_true": max_true,
        "gap_closure_false_ms": _CLOSURE_GAP_FALSE_MS,
        "units": _build_review_units(segments, parsed_texts),
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    last_error: Exception | None = None
    for attempt in range(_MERGE_REVIEW_RETRIES):
        try:
            response, _usage = client.chat(messages, temperature=config.llm_temperature)
            flags = _parse_merge_review_response(response, len(segments))
            flags = _apply_gap_filters(flags, segments, parsed_texts)
            return _hints_from_flags(segments, parsed_texts, flags)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < _MERGE_REVIEW_RETRIES - 1:
                logger.warning(f"    Merge review retry {attempt + 1}/{_MERGE_REVIEW_RETRIES}: {e}")

    logger.warning(f"    Merge review failed, skipping hints: {last_error}")
    return []


def _duration_ms(segment: Segment) -> int:
    return round((segment.end - segment.start) * 1000)


def log_merge_hints(hints: list[MergeHint]) -> None:
    """Log merge review suggestions (observation only — no layout action)."""
    for curr, nxt, text, next_text in hints:
        gap_ms = _gap_to_next_ms([curr, nxt], 0)
        curr_dur_ms = _duration_ms(curr)
        next_dur_ms = _duration_ms(nxt)
        logger.info(
            f"  Layout merge hint: {curr.unit_id} → merge_with_next | "
            f"gap={gap_ms}ms curr_dur={curr_dur_ms}ms next_dur={next_dur_ms}ms | "
            f"'{_snip_text(text)}' + next '{_snip_text(next_text)}' "
            f"({nxt.unit_id})"
        )
