"""Adjacent-overlap and duplicate-term dedup rules for merged cues."""

from __future__ import annotations

import re

from light_core import logger

# (start, end, dialogue-fields-of-one-display-cue)
_BilingualGroup = tuple[float, float, list[list[str]]]


def _dedup_srt_overlaps(cues: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Remove adjacent overlapping SRT cues, keeping the later one in each pair."""
    if len(cues) < 2:
        return cues
    deduped: list[tuple[float, float, str]] = [cues[0]]
    for cue in cues[1:]:
        if deduped[-1][1] > cue[0] + 0.001:
            deduped[-1] = cue
        else:
            deduped.append(cue)
    return deduped


def _dedup_vtt_overlaps(cues: list[tuple[float, float, str, str]]) -> list[tuple[float, float, str, str]]:
    """Remove adjacent overlapping VTT cues, keeping the later one in each pair."""
    if len(cues) < 2:
        return cues
    deduped: list[tuple[float, float, str, str]] = [cues[0]]
    for cue in cues[1:]:
        if deduped[-1][1] > cue[0] + 0.001:
            deduped[-1] = cue
        else:
            deduped.append(cue)
    return deduped


def _dedup_json_overlaps(cues: list[dict]) -> list[dict]:
    """Remove adjacent overlapping JSON cues, keeping the later one in each pair."""
    if len(cues) < 2:
        return cues
    deduped: list[dict] = [cues[0]]
    for cue in cues[1:]:
        if deduped[-1].get("end", 0) > cue.get("start", 0) + 0.001:
            deduped[-1] = cue
        else:
            deduped.append(cue)
    return deduped


def _dedup_bilingual_ass_overlaps(groups: list[_BilingualGroup]) -> list[_BilingualGroup]:
    """Remove adjacent overlapping bilingual cue groups (main subtitle stream).

    Mirrors ``_dedup_srt_overlaps``: when two adjacent cue groups overlap
    (prev.end > cur.start + tol), keep the later one — bilingual main cues
    from overlapping segments should not double-display.

    Operates on whole cue groups (box + text events sharing one start/end),
    never on individual Dialogue events: event-level dedup would treat the
    box/EN/ZH events of a single cue as overlapping duplicates and strip all
    but the last one.
    """
    if len(groups) < 2:
        return groups
    deduped: list[_BilingualGroup] = [groups[0]]
    for group in groups[1:]:
        if deduped[-1][1] > group[0] + 0.001:
            deduped[-1] = group
        else:
            deduped.append(group)
    return deduped


# ── Annotation term dedup ───────────────────────────────

_ANNOTATION_MARKER_RE = re.compile(r"^\s*(?:※\s*)+")


def _strip_annotation_marker(text: str) -> str:
    """Remove leading ※ markers from annotation body text."""
    return _ANNOTATION_MARKER_RE.sub("", text).strip()


def _extract_annotation_term(text: str) -> str:
    """Extract normalized term from formatted annotation text.

    "※ RL训练：强化学习的方法" → "rl训练"
    """
    body = _strip_annotation_marker(text)
    if "：" in body:
        return body.split("：")[0].strip().lower()
    if ":" in body:
        return body.split(":")[0].strip().lower()
    return body.strip().lower()


def _dedup_annotation_terms(cues: list[tuple]) -> list[tuple]:
    """Remove duplicate annotations by normalized term, keeping first occurrence."""
    seen: set[str] = set()
    removed = 0
    deduped: list = []
    for cue in cues:
        key = _extract_annotation_term(cue[2])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(cue)
    if removed:
        logger.info(f"    Deduplicated {removed} annotation(s) by term")
    return deduped
