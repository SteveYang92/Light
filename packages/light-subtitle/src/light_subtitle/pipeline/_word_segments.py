"""Shared gap-based word segmentation for LLM word-level stages.

Used by punct_restore, transcript_correct, and similar modules that batch
words into pause-based segments before sending them to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from light_models import Word
from light_models.punctuation import SENTENCE_ENDS

# Gap threshold for splitting words into segments (same as segment.py).
GAP_THRESHOLD = 0.5  # seconds


@dataclass
class WordSegment:
    """A pause-based segment referencing original Word objects."""

    index: int
    words: list[Word]
    text: str


def join_word_text(words: list[Word]) -> str:
    """Join word texts preserving whisper's leading-space convention."""
    if not words:
        return ""
    return "".join(w.text for w in words)


def group_words_by_gap(words: list[Word], gap_threshold: float = GAP_THRESHOLD) -> list[WordSegment]:
    """Split *words* into segments wherever inter-word gap exceeds *gap_threshold*."""
    if not words:
        return []

    segments: list[WordSegment] = []
    current: list[Word] = [words[0]]

    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap > gap_threshold:
            segments.append(WordSegment(index=len(segments), words=current, text=join_word_text(current)))
            current = [words[i]]
        else:
            current.append(words[i])

    if current:
        segments.append(WordSegment(index=len(segments), words=current, text=join_word_text(current)))

    return segments


def merge_short_segments(segments: list[WordSegment]) -> list[WordSegment]:
    """Merge very short segments with neighbors for better LLM context.

    Segments with ≤ 3 words and no sentence-ending punctuation are merged
    into the preceding segment when the gap is ≤ 0.8 s and the combined
    duration stays within 15 s.
    """
    if len(segments) < 2:
        return segments

    merged: list[WordSegment] = [segments[0]]

    for i in range(1, len(segments)):
        prev = merged[-1]
        curr = segments[i]

        gap = curr.words[0].start - prev.words[-1].end

        prev_text = prev.text.rstrip()
        curr_text = curr.text.rstrip()
        prev_short = len(prev.words) <= 3 and (not prev_text or prev_text[-1] not in SENTENCE_ENDS)
        curr_short = len(curr.words) <= 3 and (not curr_text or curr_text[-1] not in SENTENCE_ENDS)

        combined_dur = curr.words[-1].end - prev.words[0].start

        if (prev_short or curr_short) and gap <= 0.8 and combined_dur <= 15.0:
            merged[-1] = WordSegment(
                index=prev.index,
                words=prev.words + curr.words,
                text=join_word_text(prev.words + curr.words),
            )
        else:
            merged.append(curr)

    for new_idx, seg in enumerate(merged):
        seg.index = new_idx

    return merged
