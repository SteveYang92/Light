"""Tests for transcript correction — 1:1 word mapping and grammar fixes."""

from __future__ import annotations

import pytest
from light_models import Word
from light_subtitle.pipeline.transcript_correct import (
    _apply_word_corrections,
    _redistribute_timing,
    apply_word_corrections,
    preserve_leading_space,
)


def _w(text: str, start: float = 0.0, end: float = 1.0, confidence: float = 0.9, speaker: str | None = None) -> Word:
    return Word(text=text, start=start, end=end, confidence=confidence, speaker=speaker)


class TestPreserveLeadingSpace:
    def test_adds_leading_space(self):
        assert preserve_leading_space(" hello", "world") == " world"

    def test_keeps_existing_space(self):
        assert preserve_leading_space(" hello", " world") == " world"


class TestApplyWordCorrections:
    def test_one_to_one_mapping(self):
        words = [_w(" there"), _w(" model")]
        assert apply_word_corrections(words, [" their", " model"])
        assert words[0].text == " their"
        assert words[1].text == " model"

    def test_rejects_count_mismatch(self):
        words = [_w(" hello"), _w(" world")]
        assert not apply_word_corrections(words, [" hello"])
        assert words[0].text == " hello"

    def test_preserves_timestamps(self):
        words = [Word(text=" was", start=1.5, end=2.0, confidence=0.8)]
        apply_word_corrections(words, [" were"])
        assert words[0].start == 1.5
        assert words[0].end == 2.0
        assert words[0].confidence == 0.8


class TestRedistributeTiming:
    def test_equal_count_no_change(self):
        words = [_w(" a", 0.0, 0.5), _w(" b", 0.5, 1.0)]
        result = _redistribute_timing(words, [" a", " b"])
        assert len(result) == 2
        assert result[0].start == 0.0
        assert result[1].end == 1.0

    def test_adds_one_word_proportional(self):
        words = [_w(" a", 0.0, 2.0), _w(" b", 2.0, 4.0), _w(" c", 4.0, 6.0)]
        result = _redistribute_timing(words, [" a", " new", " b", " c"])
        assert len(result) == 4
        assert result[0].start == 0.0
        assert result[3].end == 6.0
        assert result[1].start == 1.5  # 2/4 * 3 → 1.5 * 2s → 3.0 → wait
        assert result[1].end == 3.0

    def test_removes_one_word_proportional(self):
        words = [_w(" a", 0.0, 1.0), _w(" b", 1.0, 3.0), _w(" c", 3.0, 4.0), _w(" d", 4.0, 6.0)]
        result = _redistribute_timing(words, [" a", " b", " c"])
        assert len(result) == 3
        assert result[0].start == 0.0
        assert result[2].end == 6.0

    def test_preserves_speaker_from_first_half(self):
        words = [_w(" a", speaker="A"), _w(" b", speaker="B")]
        result = _redistribute_timing(words, [" new", " a", " b"])
        assert len(result) == 3
        assert result[0].speaker == "A"
        assert result[1].speaker == "A"
        assert result[2].speaker == "B"

    def test_default_confidence(self):
        words = [_w(" a", confidence=0.8), _w(" b", confidence=0.6)]
        result = _redistribute_timing(words, [" new", " a", " b"])
        assert len(result) == 3
        assert result[0].confidence == 0.7


class TestPrivateApplyWordCorrections:
    def test_identical_count_matches_public(self):
        words = [_w(" was"), _w(" good")]
        assert _apply_word_corrections(words, [" was", " great"])
        assert words[0].text == " was"
        assert words[1].text == " great"

    def test_skip_delta_greater_than_one(self):
        words = [_w(" a"), _w(" b")]
        assert not _apply_word_corrections(words, [" x", " y", " z"])
        assert words[0].text == " a"

    def test_skip_delta_negative_greater_than_one(self):
        words = [_w(" a"), _w(" b"), _w(" c")]
        assert not _apply_word_corrections(words, [" x"])

    def test_grammar_fix_adds_word_at_start(self):
        words = [
            _w(" is", 0.0, 0.5),
            _w(" a", 0.5, 1.0),
            _w(" way", 1.0, 1.5),
            _w(" to", 1.5, 2.0),
            _w(" think", 2.0, 2.5),
        ]
        assert _apply_word_corrections(words, ["There", " is", " a", " way", " to", " think"])
        assert len(words) == 6
        assert words[0].text == "There"
        assert words[1].text == " is"
        assert words[0].start == 0.0
        assert words[-1].end == 2.5
        assert words[0].confidence == pytest.approx(0.9)

    def test_grammar_fix_rejects_below_min_words(self):
        words = [_w(" is"), _w(" good")]
        assert not _apply_word_corrections(words, ["It", " is", " good"])
        assert len(words) == 2

    def test_grammar_fix_keeps_unchanged_when_no_fix_needed(self):
        words = [
            _w(" the", 0.0, 0.5),
            _w(" model", 0.5, 1.0),
            _w(" is", 1.0, 1.5),
            _w(" good", 1.5, 2.0),
        ]
        assert _apply_word_corrections(words, [" the", " model", " is", " good"])
        assert len(words) == 4
        assert words[0].text == " the"
