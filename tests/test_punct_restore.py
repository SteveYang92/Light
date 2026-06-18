"""Tests for asr.punct_restore — LLM-based punctuation restoration."""

from __future__ import annotations

from light_models import Word
from light_subtitle.pipeline.punct_restore import (
    _diff_punct_to_words,
    _group_words,
    _has_sufficient_punctuation,
    _join_text,
    _parse_llm_response,
    _Segment,
)

# ── _group_words ───────────────────────────────────────────────────


def test_group_words_empty():
    assert _group_words([]) == []


def test_group_words_single():
    words = [Word(text=" hello", start=0.0, end=1.0, confidence=0.9)]
    segs = _group_words(words)
    assert len(segs) == 1
    assert segs[0].text == " hello"


def test_group_words_by_gap():
    words = [
        Word(text=" hello", start=0.0, end=0.5, confidence=0.9),
        Word(text=" world", start=1.2, end=1.7, confidence=0.9),  # gap=0.7 > 0.5
    ]
    segs = _group_words(words)
    assert len(segs) == 2


def test_group_words_no_gap():
    words = [
        Word(text=" hello", start=0.0, end=0.5, confidence=0.9),
        Word(text=" world", start=0.6, end=1.0, confidence=0.9),  # gap=0.1 < 0.5
    ]
    segs = _group_words(words)
    assert len(segs) == 1


# ── _diff_punct_to_words ──────────────────────────────────────────


def test_diff_trailing_question():
    words = [Word(text=" hello", start=0.0, end=0.5, confidence=0.9)]
    _diff_punct_to_words(words, " hello", " hello?")
    assert words[0].text == " hello?"


def test_diff_trailing_period():
    words = [Word(text=" world", start=0.0, end=0.5, confidence=0.9)]
    _diff_punct_to_words(words, " world", " world.")
    assert words[0].text == " world."


def test_diff_trailing_exclamation():
    words = [Word(text=" great", start=0.0, end=0.5, confidence=0.9)]
    _diff_punct_to_words(words, " great", " great!")
    assert words[0].text == " great!"


def test_diff_no_change():
    words = [Word(text=" hello", start=0.0, end=0.5, confidence=0.9)]
    _diff_punct_to_words(words, " hello", " hello")
    assert words[0].text == " hello"


def test_diff_multiple_words():
    words = [
        Word(text=" what", start=0.0, end=0.3, confidence=0.9),
        Word(text=" is", start=0.3, end=0.5, confidence=0.9),
        Word(text=" that", start=0.5, end=0.8, confidence=0.9),
    ]
    _diff_punct_to_words(words, " what is that", " what is that?")
    # Only last word gets trailing punctuation
    assert words[0].text == " what"
    assert words[1].text == " is"
    assert words[2].text == " that?"


def test_diff_empty_text():
    words = []
    _diff_punct_to_words(words, "", "")
    assert words == []


def test_diff_llm_strips_whitespace():
    words = [Word(text=" okay", start=0.0, end=0.5, confidence=0.9)]
    _diff_punct_to_words(words, " okay", "okay.")  # LLM stripped leading space
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


# ── _join_text ────────────────────────────────────────────────────


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
