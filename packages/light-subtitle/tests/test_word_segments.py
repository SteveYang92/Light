"""Tests for shared gap-based word segmentation."""

from __future__ import annotations

from light_models import Word
from light_subtitle.pipeline._word_segments import (
    WordSegment,
    group_words_by_gap,
    join_word_text,
    merge_short_segments,
)


def _w(text: str, start: float = 0.0, end: float = 1.0) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


def _seg(index: int, words: list[Word]) -> WordSegment:
    return WordSegment(index=index, words=words, text=join_word_text(words))


class TestGroupWordsByGap:
    def test_single_word(self):
        words = [_w(" hello")]
        segs = group_words_by_gap(words)
        assert len(segs) == 1
        assert len(segs[0].words) == 1

    def test_splits_on_large_gap(self):
        words = [
            _w(" a", start=0.0, end=0.5),
            _w(" b", start=1.5, end=2.0),
        ]
        segs = group_words_by_gap(words)
        assert len(segs) == 2


class TestMergeShortSegments:
    def test_short_segment_merged(self):
        segs = [
            _seg(0, [_w(" hello"), _w(" world"), _w(" this")]),
            _seg(1, [_w(" is"), _w(" short")]),
        ]
        segs[0].words[-1].end = 1.0
        segs[1].words[0].start = 1.3
        result = merge_short_segments(segs)
        assert len(result) == 1
        assert len(result[0].words) == 5
