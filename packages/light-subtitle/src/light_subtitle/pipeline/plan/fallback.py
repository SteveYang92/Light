"""Deterministic insurance plans, used only when the LLM planner fails.

These are deliberately simple — sentence-ending punctuation plus silence
gaps, no word lists and no scoring.  They guarantee the pipeline always
produces *a* valid plan; the LLM planner remains the main path.
"""

from __future__ import annotations

from light_models import Segment, Word

from ...language import is_sentence_end

# A silence longer than this separates two thoughts regardless of punctuation.
_MERGE_GAP_MAX = 3.0  # seconds

# Split parts smaller than this read as stubs on screen (QC TinyCue).
_MIN_PART_WORDS = 3


def merge_fragments(segments: list[Segment]) -> list[list[int]]:
    """Group consecutive segment indices into sentence-complete groups.

    A fragment (text not ending with sentence punctuation) always merges
    forward; merging stops at a long silence or a speaker change.
    """
    if not segments:
        return []
    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(segments)):
        prev, cand = segments[current[-1]], segments[i]
        gap = cand.start - prev.end
        speaker_change = bool(prev.speaker and cand.speaker and prev.speaker != cand.speaker)
        if is_sentence_end(prev.source_text) or gap > _MERGE_GAP_MAX or speaker_change:
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    groups.append(current)
    return groups


def split_at_gaps(words: list[Word], max_duration: float) -> list[tuple[int, int]]:
    """Split a word span at the largest silences until every part fits
    ``max_duration``.  Returns [start, end) word-index ranges."""
    parts: list[tuple[int, int]] = []
    stack = [(0, len(words))]
    while stack:
        s, e = stack.pop()
        if e - s > 1 and words[e - 1].end - words[s].start > max_duration:
            cut = _best_cut(words, s, e)
            if s < cut < e:
                stack.append((cut, e))
                stack.append((s, cut))
                continue
        parts.append((s, e))
    parts.sort()
    return _merge_stub_parts(parts, words)


def _merge_stub_parts(parts: list[tuple[int, int]], words: list[Word]) -> list[tuple[int, int]]:
    """Fold parts with fewer than ``_MIN_PART_WORDS`` words back into the
    cheaper neighbour — a small duration overflow beats a stub cue."""
    merged = list(parts)
    while len(merged) > 1:
        stub = next((i for i, (s, e) in enumerate(merged) if e - s < _MIN_PART_WORDS), None)
        if stub is None:
            break
        if stub == 0:
            merged[1] = (merged[0][0], merged[1][1])
        elif stub == len(merged) - 1:
            merged[stub - 1] = (merged[stub - 1][0], merged[stub][1])
        else:
            left = (merged[stub - 1][0], merged[stub][1])
            right = (merged[stub][0], merged[stub + 1][1])
            left_dur = words[left[1] - 1].end - words[left[0]].start
            right_dur = words[right[1] - 1].end - words[right[0]].start
            if left_dur <= right_dur:
                merged[stub - 1] = left
            else:
                merged[stub + 1] = right
            del merged[stub]
            continue
        del merged[stub]
    return merged


def _best_cut(words: list[Word], start: int, end: int) -> int:
    """Cut point: the largest inter-word silence, else the midpoint word."""
    best_i, best_gap = -1, -1.0
    for i in range(start + 1, end):
        gap = words[i].start - words[i - 1].end
        if gap > best_gap:
            best_i, best_gap = i, gap
    return best_i if best_i != -1 else (start + end) // 2
