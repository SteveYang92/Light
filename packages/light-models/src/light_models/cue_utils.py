"""Helpers for display-merged subtitle cues (merged_from chains)."""

from __future__ import annotations

from .cue import SubtitleCue
from .utils import is_cjk


def effective_unit_ids(cue: SubtitleCue) -> set[str]:
    """Return unit_ids covered by a cue, including absorbed merge chain members."""
    return {cue.unit_id, *cue.merged_from}


def covered_source_text(cue: SubtitleCue, source_map: dict[str, str]) -> str:
    """Concatenate source text for head unit + merged_from in chain order."""
    parts = [source_map[uid].strip() for uid in [cue.unit_id, *cue.merged_from] if source_map.get(uid, "").strip()]
    if not parts:
        return source_map.get(cue.unit_id, "")
    if is_cjk(parts[0][0]):
        return "".join(parts)
    return " ".join(parts)


def covered_time_window(
    cue: SubtitleCue,
    unit_times: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """Min start / max end across head unit and merged_from in unit_times."""
    bounds = [unit_times[uid] for uid in [cue.unit_id, *cue.merged_from] if uid in unit_times]
    if not bounds:
        return None
    return min(start for start, _ in bounds), max(end for _, end in bounds)
