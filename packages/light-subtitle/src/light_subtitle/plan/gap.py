"""Tiered semantic gap-split for overlong units — deterministic, no LLM."""

from __future__ import annotations

from light_models import Word

from .lexicon import CONJ, FUNC_TAIL

_SENTENCE_ENDS = frozenset({".", "!", "?"})
_CLAUSE_ENDS = frozenset({",", ";", ":", "—"})


def gap_split(words: list[Word], max_duration: float) -> list[tuple[int, int]]:
    """Split an overlong word span into [start, end) ranges using tiered break priority.

    Priority: sentence_final > clause_punct > before conjunction > largest gap.
    Stub parts (< 3 words) are folded into neighbours.
    """
    return _split_recursive(words, 0, len(words), max_duration)


def _split_recursive(words: list[Word], start: int, end: int, max_dur: float) -> list[tuple[int, int]]:
    dur = words[end - 1].end - words[start].start
    if dur <= max_dur or end - start <= 1:
        return [(start, end)]

    cut = _find_best_cut(words, start, end)
    if cut <= start or cut >= end:
        return [(start, end)]

    left = _split_recursive(words, start, cut, max_dur)
    right = _split_recursive(words, cut, end, max_dur)
    parts = left + right
    return _merge_stubs(parts, words)


def _find_best_cut(words: list[Word], start: int, end: int) -> int:
    """Find the best cut point using tiered priority, biased towards the midpoint."""
    mid = (start + end) // 2
    candidates: list[tuple[int, int, float]] = []  # (cut_pos, priority, gap)

    for i in range(start + 1, end):
        gap = words[i].start - words[i - 1].end
        if gap < 0:
            gap = 0.0
        left_text = words[i - 1].text.strip()
        right_text = words[i].text.strip()

        if _is_legal_tail(left_text) and end - i >= 3 and i - start >= 3:
            priority = _cut_priority(left_text, right_text, gap)
            candidates.append((i, priority, gap))

    if not candidates:
        return -1

    # Sort by priority (lower is better), then by distance to midpoint, then by larger gap
    candidates.sort(key=lambda x: (x[1], abs(x[0] - mid), -x[2]))
    return candidates[0][0]


def _cut_priority(left: str, right: str, gap: float) -> int:
    """Lower = better cut."""
    tail = left.rstrip()
    if tail and tail[-1] in _SENTENCE_ENDS:
        return 1
    if tail and tail[-1] in _CLAUSE_ENDS:
        return 2
    right_core = right.strip(".,!?;:—\"'()[]“”‘’").lower()
    if right_core in CONJ:
        return 3
    if gap >= 0.50:
        return 4
    if gap >= 0.25:
        return 5
    return 6


def _is_legal_tail(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if any(s.endswith(p) for p in (".", ",", "!", "?", ";", ":", "—", "…")):
        return True
    core = s.strip(".,!?;:—\"'()[]“”‘’").lower()
    return core not in FUNC_TAIL


def _merge_stubs(parts: list[tuple[int, int]], words: list[Word]) -> list[tuple[int, int]]:
    if len(parts) <= 1:
        return parts
    result = list(parts)
    changed = True
    while changed and len(result) > 1:
        changed = False
        for i in range(len(result) - 1, -1, -1):
            s, e = result[i]
            if e - s >= 3:
                continue
            if i > 0:
                result[i - 1] = (result[i - 1][0], e)
            elif i + 1 < len(result):
                result[i + 1] = (s, result[i + 1][1])
            del result[i]
            changed = True
            break
    return result
