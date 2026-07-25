"""Tests for plan word-level split validation and grammar-aware fallback."""

from __future__ import annotations

from light_models import Word
from light_subtitle.plan.fallback import split_at_gaps
from light_subtitle.plan.planner import _break_problems, _dangling_tail


def _words(texts: list[str], word_dur: float = 0.4, gap: float = 0.1) -> list[Word]:
    words = []
    t = 0.0
    for text in texts:
        words.append(Word(text=text, start=t, end=t + word_dur, confidence=0.9))
        t += word_dur + gap
    return words


# ── _dangling_tail ──────────────────────────────────────────────────


def test_dangling_tail_flags_function_words():
    assert _dangling_tail(Word(text=" are", start=0, end=1, confidence=0.9)) == "are"
    assert _dangling_tail(Word(text=" if", start=0, end=1, confidence=0.9)) == "if"
    assert _dangling_tail(Word(text=" you", start=0, end=1, confidence=0.9)) == "you"
    assert _dangling_tail(Word(text=" The", start=0, end=1, confidence=0.9)) == "the"


def test_dangling_tail_allows_content_words():
    assert _dangling_tail(Word(text=" stage", start=0, end=1, confidence=0.9)) is None
    assert _dangling_tail(Word(text=" experience", start=0, end=1, confidence=0.9)) is None


def test_dangling_tail_allows_object_pronouns():
    assert _dangling_tail(Word(text=" it", start=0, end=1, confidence=0.9)) is None
    assert _dangling_tail(Word(text=" them", start=0, end=1, confidence=0.9)) is None


def test_dangling_tail_exempts_trailing_punctuation():
    assert _dangling_tail(Word(text=" you,", start=0, end=1, confidence=0.9)) is None
    assert _dangling_tail(Word(text=" are.", start=0, end=1, confidence=0.9)) is None


# ── _break_problems ─────────────────────────────────────────────────


def test_break_after_function_word_rejected():
    # "…or if you are | working in a bank" — cut strands the auxiliary.
    words = _words(["or", "if", "you", "are", "working", "in", "a", "bank"])
    problems = _break_problems([3], words, max_duration=7.0)
    assert any('"are"' in p for p in problems)


def test_break_after_content_word_accepted():
    words = _words(["people", "do", "it", "in", "this", "stage", "is", "that", "they", "experience"])
    assert _break_problems([5], words, max_duration=7.0) == []


def test_break_after_punctuated_word_accepted():
    words = _words(["I", "know", "him,", "and", "he", "knows", "me", "too"])
    assert _break_problems([2], words, max_duration=7.0) == []


def test_break_validation_keeps_existing_hard_checks():
    words = _words(["a", "b", "c", "d", "e", "f"])
    assert _break_problems([], words, max_duration=7.0)  # no breaks
    assert _break_problems([9], words, max_duration=7.0)  # out of range
    assert _break_problems([3, 3], words, max_duration=7.0)  # duplicates


# ── fallback.split_at_gaps ──────────────────────────────────────────


def _timed_words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(text=t, start=s, end=e, confidence=0.9) for t, s, e in spec]


def test_fallback_avoids_dangling_gap_boundary():
    # Biggest silence sits right after "and" (0.5s); a clean boundary after
    # "ourselves" only has 0.2s.  The grammar-aware cut must skip "and".
    words = _timed_words(
        [
            ("we", 0.0, 0.4),
            ("experience", 0.5, 1.2),
            ("ourselves", 1.3, 2.0),
            ("and", 2.2, 2.4),
            ("interact", 2.9, 3.5),
            ("with", 3.6, 3.8),
            ("others", 3.9, 4.5),
            ("daily", 4.6, 6.2),
        ]
    )
    ranges = split_at_gaps(words, max_duration=4.0)
    boundaries = [words[e - 1].text for s, e in ranges[:-1]]
    assert "and" not in boundaries
    assert "with" not in boundaries


def test_fallback_prefers_punctuated_boundary():
    # "us," carries a comma: preferred over the bigger silence after "we".
    words = _timed_words(
        [
            ("this", 0.0, 0.3),
            ("is", 0.4, 0.6),
            ("us,", 0.7, 1.4),
            ("and", 1.5, 1.7),
            ("we", 1.8, 2.0),
            ("keep", 2.6, 3.0),
            ("going", 3.1, 3.6),
            ("today", 3.7, 5.4),
        ]
    )
    ranges = split_at_gaps(words, max_duration=4.0)
    boundaries = [words[e - 1].text for s, e in ranges[:-1]]
    assert boundaries == ["us,"]
