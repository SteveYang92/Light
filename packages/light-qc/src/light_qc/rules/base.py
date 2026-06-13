"""Rule base classes and shared helpers."""

from __future__ import annotations

from light_models import QCIssue, SubtitleCue

from ..config import QCConfig


class HardRule:
    name: str = ""
    default_severity: str = "error"
    # Explicit language scope.  None = all languages.
    # Set to {"zh"} or {"en"} to restrict to one language.
    languages: set[str] | None = None

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        return []


class SoftRule:
    name: str = ""
    default_severity: str = "suggestion"
    languages: set[str] | None = None

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        return []


# ── Helpers for rules that only operate on a single language ──


def _iter_cues(cues: dict[str, list[SubtitleCue]]):
    """Yield (lang, cue_list) pairs. Most simple rules iterate over just one."""
    yield from cues.items()


def _all_cues(cues: dict[str, list[SubtitleCue]]) -> list[SubtitleCue]:
    """Flatten all cues across languages into a single list."""
    result = []
    for cue_list in cues.values():
        result.extend(cue_list)
    return result


def pair_bilingual(cues: dict[str, list[SubtitleCue]], source_lang: str):
    """Yield (source_idx, source_cue, target_idx, target_cue) for
    time-overlapping bilingual pairs."""
    source_cues = cues.get(source_lang, [])
    for t_lang, t_cues in cues.items():
        if t_lang == source_lang:
            continue
        for si, sc in enumerate(source_cues):
            for ti, tc in enumerate(t_cues):
                overlap_start = max(sc.start, tc.start)
                overlap_end = min(sc.end, tc.end)
                if overlap_start < overlap_end - 0.05:
                    yield si, sc, ti, tc
