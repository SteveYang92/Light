"""Tests for punctuation restoration — word-level diff and segment merging."""

from __future__ import annotations

from light_asr_polish.punct import (
    _apply_punct_to_words,
    _has_sufficient_punctuation,
    _join_text,
    _merge_short_segments,
    _parse_llm_response,
    _Segment,
)
from light_models import Word

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


# ═══════════════════════════════════════════════════════
# Merged from root tests/test_punct_restore.py
# ═══════════════════════════════════════════════════════


def test_diff_trailing_exclamation():
    words = [Word(text=" great", start=0.0, end=0.5, confidence=0.9)]
    _apply_punct_to_words(words, " great", " great!")
    assert words[0].text == " great!"


def test_diff_llm_strips_whitespace():
    words = [Word(text=" okay", start=0.0, end=0.5, confidence=0.9)]
    _apply_punct_to_words(words, " okay", "okay.")  # LLM stripped leading space
    assert words[0].text == " okay."  # Should still append to original


# ── _parse_llm_response ───────────────────────────────────────────


def test_parse_json_array():
    resp = '[{"index": 0, "text": "hello."}, {"index": 1, "text": "world?"}]'
    result = _parse_llm_response(resp)
    assert len(result) == 2
    assert result[0]["text"] == "hello."
    assert result[1]["index"] == 1


def test_parse_json_extract_from_markdown():
    resp = '```json\n[{"index": 0, "text": "hi."}]\n```'
    result = _parse_llm_response(resp)
    assert len(result) == 1
    assert result[0]["text"] == "hi."


def test_parse_invalid():
    assert _parse_llm_response("not json") == []


# ── _has_sufficient_punctuation ───────────────────────────────────


def test_has_sufficient_punct_above_threshold():
    segs = [
        _Segment(index=0, words=[], text=" hello world."),
        _Segment(index=1, words=[], text=" how are you?"),
        _Segment(index=2, words=[], text=" i'm fine."),
        _Segment(index=3, words=[], text=" good to hear"),  # no punct
    ]
    assert _has_sufficient_punctuation(segs, threshold=0.3) is True


def test_has_sufficient_punct_below_threshold():
    segs = [
        _Segment(index=0, words=[], text=" hello world"),
        _Segment(index=1, words=[], text=" how are you"),
        _Segment(index=2, words=[], text=" i'm fine"),
        _Segment(index=3, words=[], text=" good to hear"),
    ]
    assert _has_sufficient_punctuation(segs, threshold=0.3) is False


def test_has_sufficient_punct_empty():
    assert _has_sufficient_punctuation([]) is False


def test_has_sufficient_punct_all_punctuated():
    segs = [_Segment(index=0, words=[], text="hello.")]
    assert _has_sufficient_punctuation(segs, threshold=1.0) is True


def test_has_sufficient_punct_mixed_edge():
    # Exactly at threshold
    segs = [
        _Segment(index=0, words=[], text="one."),
        _Segment(index=1, words=[], text="two."),
        _Segment(index=2, words=[], text="three"),
        _Segment(index=3, words=[], text="four."),
        _Segment(index=4, words=[], text="five."),
        _Segment(index=5, words=[], text="six"),
        _Segment(index=6, words=[], text="seven."),
        _Segment(index=7, words=[], text="eight"),
        _Segment(index=8, words=[], text="nine"),
        _Segment(index=9, words=[], text="ten"),
    ]
    # 4/10 = 0.4 >= 0.3
    assert _has_sufficient_punctuation(segs) is True


# ── _join_text ─────────────────────────────────────────────────────


def test_join_text():
    words = [
        Word(text=" hello", start=0.0, end=0.5, confidence=0.9),
        Word(text=" world", start=0.6, end=1.0, confidence=0.9),
    ]
    assert _join_text(words) == " hello world"
