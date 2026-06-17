"""Apply merge-review hints to translated cues (display merge at translate time)."""

from __future__ import annotations

from light_models import SubtitleCue

from ... import logger
from ...config import SubtitleConfig
from .merge_review import MergeHint

_MAX_DURATION_MULTIPLIER = 2.0


def _join_cue_text(lang: str, parts: list[str]) -> str:
    stripped = [p.strip() for p in parts if p.strip()]
    if not stripped:
        return ""
    if lang == "zh":
        return "".join(stripped)
    return " ".join(stripped)


def _build_merge_next(hints: list[MergeHint]) -> dict[str, str]:
    return {curr.unit_id: nxt.unit_id for curr, nxt, _, _ in hints}


def _collect_chain(cues: list[SubtitleCue], start: int, merge_next: dict[str, str]) -> list[SubtitleCue]:
    chain = [cues[start]]
    while chain[-1].unit_id in merge_next:
        expected = merge_next[chain[-1].unit_id]
        next_idx = start + len(chain)
        if next_idx >= len(cues) or cues[next_idx].unit_id != expected:
            break
        chain.append(cues[next_idx])
    return chain


def _merge_chain(chain: list[SubtitleCue]) -> SubtitleCue:
    head = chain[0]
    return SubtitleCue(
        cue_id=head.cue_id,
        unit_id=head.unit_id,
        start=head.start,
        end=chain[-1].end,
        text=_join_cue_text(head.lang, [c.text for c in chain]),
        lang=head.lang,
        speaker=head.speaker,
        words=[w for c in chain for w in c.words],
        qc=head.qc,
        annotation=head.annotation,
        merged_from=[c.unit_id for c in chain[1:]],
    )


def covered_unit_ids(cues: list[SubtitleCue]) -> set[str]:
    """Return all unit_ids present or absorbed via display merge."""
    ids: set[str] = set()
    for cue in cues:
        ids.add(cue.unit_id)
        ids.update(cue.merged_from)
    return ids


def apply_display_merges(
    cues: list[SubtitleCue],
    hints: list[MergeHint],
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    """Merge consecutive cues per review hints; skip when merged duration exceeds limit."""
    if not hints:
        return cues

    merge_next = _build_merge_next(hints)
    max_allowed = config.max_duration * _MAX_DURATION_MULTIPLIER
    result: list[SubtitleCue] = []
    i = 0
    while i < len(cues):
        chain = _collect_chain(cues, i, merge_next)
        if len(chain) == 1:
            result.append(chain[0])
            i += 1
            continue

        merged_duration = chain[-1].end - chain[0].start
        unit_ids = " → ".join(c.unit_id for c in chain)
        if merged_duration > max_allowed:
            logger.warning(f"  Merge skipped (duration {merged_duration:.2f}s > {max_allowed:.2f}s): {unit_ids}")
            result.extend(chain)
            i += len(chain)
            continue

        merged = _merge_chain(chain)
        logger.info(
            f"  Merge applied: {unit_ids} | duration={merged_duration:.2f}s | "
            f"'{merged.text[:40]}{'…' if len(merged.text) > 40 else ''}'"
        )
        result.append(merged)
        i += len(chain)

    return result
