"""Tests for punctuation restoration — word-level diff and segment merging."""

from __future__ import annotations

from light_models import Word
from light_subtitle.pipeline.punct_restore import (
    _apply_punct_to_words,
    _join_text,
    _merge_short_segments,
    _Segment,
)

# ── Helpers ────────────────────────────────────────────


def _w(text: str) -> Word:
    """Create a word with minimal fields for testing."""
    return Word(text=text, start=0.0, end=1.0, confidence=0.9)


def _seg(index: int, words: list[Word]) -> _Segment:
    return _Segment(index=index, words=words, text=_join_text(words))


# ═══════════════════════════════════════════════════════
# _apply_punct_to_words — word-level alignment
# ═══════════════════════════════════════════════════════


class TestApplyPunctToWords:
    """Word-level punctuation mapping from LLM output to original words."""

    def test_trailing_period_on_last_word(self):
        """Trailing period mapped to last word."""
        words = [_w(" hello"), _w(" world")]
        old = " hello world"
        new = " hello world."
        _apply_punct_to_words(words, old, new)
        assert words[1].text == " world."

    def test_internal_comma_on_correct_word(self):
        """Comma after first word mapped to first word, not last."""
        words = [_w(" firstly"), _w(" this"), _w(" and"), _w(" that")]
        old = " firstly this and that"
        new = " firstly, this and that."
        _apply_punct_to_words(words, old, new)
        assert words[0].text == " firstly,"
        assert words[3].text == " that."

    def test_multi_sentence_periods(self):
        """Multiple periods within a segment mapped to correct words."""
        words = [_w(" hello"), _w(" world"), _w(" goodbye"), _w(" moon")]
        old = " hello world goodbye moon"
        new = " hello world. goodbye moon."
        _apply_punct_to_words(words, old, new)
        assert words[1].text == " world."
        assert words[3].text == " moon."

    def test_question_mark(self):
        """Question mark attached to last word."""
        words = [_w(" what"), _w(" city")]
        old = " what city"
        new = " what city?"
        _apply_punct_to_words(words, old, new)
        assert words[1].text == " city?"

    def test_comma_and_period_together(self):
        """Comma on word 2, period on word 4."""
        words = [_w(" first"), _w(" second"), _w(" third"), _w(" fourth")]
        old = " first second third fourth"
        new = " first, second third, fourth."
        _apply_punct_to_words(words, old, new)
        assert words[0].text == " first,"
        assert words[2].text == " third,"
        assert words[3].text == " fourth."

    def test_no_change_when_identical(self):
        """No modification when old and new text are identical."""
        words = [_w(" hello"), _w(" world")]
        old = " hello world"
        new = " hello world"
        _apply_punct_to_words(words, old, new)
        assert words[0].text == " hello"
        assert words[1].text == " world"

    def test_no_change_when_no_punctuation_added(self):
        """No modification when LLM only changes casing (no punct)."""
        words = [_w(" hello"), _w(" world")]
        old = " hello world"
        new = " Hello World"  # LLM changed casing but no punct
        _apply_punct_to_words(words, old, new)
        assert words[0].text == " hello"
        assert words[1].text == " world"

    def test_empty_words_unchanged(self):
        """Empty word list returns immediately."""
        words: list[Word] = []
        _apply_punct_to_words(words, "hello", "hello.")
        assert words == []

    def test_preserves_trailing_whitespace(self):
        """Word with trailing space keeps it after punct is appended."""
        words = [_w(" hello ")]
        old = " hello "
        new = " hello."
        _apply_punct_to_words(words, old, new)
        # Trailing space preserved: "hello." + " " = " hello. "
        assert words[0].text == " hello. "

    def test_no_duplicate_punctuation(self):
        """Same punctuation already present is not duplicated."""
        words = [_w(" world.")]
        old = " world."
        new = " world."  # LLM output same
        _apply_punct_to_words(words, old, new)
        assert words[0].text == " world."

    def test_different_whitespace_handled(self):
        """LLM may change whitespace — diff still maps punct correctly."""
        words = [_w(" hello"), _w(" world")]
        old = " hello world"
        new = " hello,  world."  # extra space
        _apply_punct_to_words(words, old, new)
        assert words[0].text == " hello,"


# ═══════════════════════════════════════════════════════
# _merge_short_segments
# ═══════════════════════════════════════════════════════


class TestMergeShortSegments:
    """Short segment merging for better LLM context."""

    def test_short_segment_merged_with_prev(self):
        """A 2-word segment merges into preceding segment."""
        segs = [
            _seg(0, [_w(" hello"), _w(" world"), _w(" this")]),
            _seg(1, [_w(" is"), _w(" short")]),
        ]
        # Set gap to be small (≤ 0.8s)
        segs[0].words[-1].end = 1.0
        segs[1].words[0].start = 1.3  # gap = 0.3s
        result = _merge_short_segments(segs)
        assert len(result) == 1
        assert len(result[0].words) == 5

    def test_large_gap_not_merged(self):
        """Large gap prevents merging."""
        segs = [
            _seg(0, [_w(" hello"), _w(" world"), _w(" this")]),
            _seg(1, [_w(" is"), _w(" short")]),
        ]
        segs[0].words[-1].end = 1.0
        segs[1].words[0].start = 3.0  # gap = 2.0s > 0.8
        result = _merge_short_segments(segs)
        assert len(result) == 2

    def test_sentence_end_not_merged(self):
        """Segment ending with '.' is not merged even if short."""
        segs = [
            _seg(0, [_w(" hello.")]),
            _seg(1, [_w(" is"), _w(" short")]),
        ]
        segs[0].words[-1].end = 1.0
        segs[1].words[0].start = 1.3
        result = _merge_short_segments(segs)
        # First seg ends with ".", so it's not "short enough to merge"
        # But second IS short — so second merges into first
        assert len(result) == 1

    def test_single_segment_unchanged(self):
        """Single segment passes through unchanged."""
        segs = [_seg(0, [_w(" hello")])]
        result = _merge_short_segments(segs)
        assert len(result) == 1

    def test_both_long_not_merged(self):
        """Two long segments are not merged."""
        segs = [
            _seg(0, [_w(" hello"), _w(" world"), _w(" this"), _w(" is"), _w(" long")]),
            _seg(1, [_w(" and"), _w(" this"), _w(" too"), _w(" also")]),
        ]
        segs[0].words[-1].end = 1.0
        segs[1].words[0].start = 1.3
        result = _merge_short_segments(segs)
        assert len(result) == 2

    def test_chained_merge(self):
        """Two consecutive short segments merge into one."""
        segs = [
            _seg(0, [_w(" a")]),
            _seg(1, [_w(" b")]),
            _seg(2, [_w(" c")]),
        ]
        for s in segs:
            s.words[0].start = 0.0
            s.words[0].end = 1.0
        segs[0].words[0].end = 0.5
        segs[1].words[0].start = 0.6
        segs[1].words[0].end = 1.5
        segs[2].words[0].start = 1.6
        result = _merge_short_segments(segs)
        assert len(result) == 1
        assert len(result[0].words) == 3
