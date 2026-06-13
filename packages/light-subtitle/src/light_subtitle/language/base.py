"""Language-agnostic text utilities shared by English and CJK processing.

Sentence-ending detection, abbreviation handling, language identification,
and the BreakFinder abstraction used by both EnglishBreakFinder and
ChineseBreakFinder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from light_models import Word, is_cjk
from light_models.punctuation import CLAUSE_PUNCT, SENTENCE_ENDS

# Sentence-ending punctuation character set (excluding abbreviation dots).
SENTENCE_END = set(SENTENCE_ENDS)

# ── BreakFinder priority constants ───────────────────────────

BREAK_SENTENCE_END = 100  # .!?。！？
BREAK_CLAUSE = 80  # ,;:，；：
BREAK_CONJUNCTION = 60  # and/but/然而/所以...
BREAK_PREP_ARTICLE = 40  # in/to/the/a...
BREAK_WORD = 20  # space
BREAK_FALLBACK = 0  # no good break found

# ── BreakFinder ABC ──────────────────────────────────────────


class BreakFinder(ABC):
    """Find optimal break positions in text, respecting language constraints.

    Strategy: scan [low, high] → score each position → return position
    with highest score that is not forbidden.  The scoring priority
    chain ensures sentence-ending punctuation always wins over commas,
    which win over conjunctions, etc.
    """

    def __init__(self, text: str):
        self.text = text

    @abstractmethod
    def score(self, pos: int) -> int:
        """Return the break-quality score for splitting at *pos*.

        *pos* is the index of the last character on the first line
        (i.e. the break goes after text[pos]).
        """
        ...

    @abstractmethod
    def is_forbidden(self, pos: int) -> bool:
        """Return True if splitting at *pos* is never allowed."""
        ...

    def find(self, low: int, high: int) -> int | None:
        """Find the highest-scoring break in [low, high]."""
        high = min(high, len(self.text) - 1)
        if low > high:
            return None
        best_pos = None
        best_score = -1
        for pos in range(low, high + 1):
            if self.is_forbidden(pos):
                continue
            s = self.score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos
        return best_pos if best_score > 0 else None

    def find_balanced(self, low: int, high: int, fallback: int) -> int:
        """Find best break position balancing semantics and line symmetry.

        Scans [low, high] from right to left, combining semantic score
        with a balance bonus.  Returns *fallback* (adjusted for forbidden
        positions) when no scored position exists in the range.
        """
        high = min(high, len(self.text) - 1)
        if low > high:
            return self._adjust_fallback(fallback)
        total = len(self.text)
        best_pos = high
        best_combined = -1
        for pos in range(high, low - 1, -1):
            if self.is_forbidden(pos):
                continue
            semantic = self.score(pos)
            if semantic == 0:
                continue
            left = pos
            right = total - pos
            bal = min(left, right) / max(left, right) if max(left, right) > 0 else 0
            combined = semantic + bal * 100
            if combined > best_combined:
                best_combined = combined
                best_pos = pos
        if best_combined > 0:
            return best_pos
        return self._adjust_fallback(fallback)

    def _adjust_fallback(self, pos: int) -> int:
        """Walk forward/backward from *pos* to find nearest safe position."""
        if not self.is_forbidden(pos):
            return pos
        for d in range(1, min(len(self.text) - pos, 15)):
            if not self.is_forbidden(pos + d):
                return pos + d
        for d in range(1, min(pos, 15)):
            if not self.is_forbidden(pos - d):
                return pos - d
        return pos


# Known abbreviations / address forms — a trailing "." alone does
# not make these sentence-ending.  Used by is_abbreviation_dot() and
# is_sentence_end().
_KNOWN_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "st",
    "jr",
    "sr",
    "capt",
    "gen",
    "e.g",
    "i.e",
    "etc",
    "vs",
    "inc",
    "ltd",
    "dept",
    "est",
    "ph.d",
    "ll.b",
    "ll.m",
    "b.a",
    "b.s",
    "m.a",
    "m.s",
}

# Short discourse fillers — trailing "." ends the utterance, not an abbreviation.
# Without this exclusion, the len≤4 uppercase heuristic treats "Hmm." like "Mr."
# and compose chains them into the next segment.
_DISCOURSE_FILLERS = {"hmm", "um", "uh", "mm", "er", "ah", "oh"}

# ── Abbreviation detection ──


def _is_known_abbreviation_word(word: str) -> bool:
    """True if *word* (with trailing dot) is a known abbreviation.

    Handles Mr., Dr., U.S., Ph.D., etc.
    """
    if not word.endswith("."):
        return False
    no_dot = word[:-1]
    if no_dot.lower() in _DISCOURSE_FILLERS:
        return False
    if no_dot.lower() in _KNOWN_ABBREVIATIONS:
        return True
    if "." in no_dot and no_dot.replace(".", "").isupper():
        return True
    if len(word) <= 4 and no_dot and no_dot[0].isupper():
        return True
    return False


def is_abbreviation_dot(text: str, pos: int) -> bool:
    """Check if a '.' at *pos* is part of an abbreviation (e.g. 'U.S.').

    A dot is an abbreviation marker when:
    - The preceding character is a letter.
    - One of:
      a) There IS more text after the dot, and the next non-space
         character is uppercase (e.g. "U.S. These" → abbreviation).
      b) The dot is mid-word in a multi-dot abbreviation (e.g. "U.S.").
      c) The word ending at this dot is a known abbreviation
         (Mr., Dr., e.g., U.S., etc.).
    """
    if text[pos] != "." or pos <= 0:
        return False
    if not text[pos - 1].isalpha():
        return False

    rest = text[pos + 1 :].lstrip()
    if rest and rest[0].isupper():
        return True

    word_start = pos
    while word_start > 0 and text[word_start - 1].isalpha():
        word_start -= 1
    while word_start >= 2 and text[word_start - 1] == "." and text[word_start - 2].isalpha():
        word_start -= 2
        while word_start > 0 and text[word_start - 1].isalpha():
            word_start -= 1

    word_full = text[word_start : pos + 1]
    return _is_known_abbreviation_word(word_full)


def is_sentence_end(text: str) -> bool:
    """True if *text* ends with sentence-ending punctuation
    (ignoring abbreviation dots like 'U.S.', 'Mr.')."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[-1] in SENTENCE_END:
        if stripped[-1] == "." and is_abbreviation_dot(stripped, len(stripped) - 1):
            return False
        return True
    return False


# ── Language detection ──


def detect_source_lang(words: list[Word]) -> str:
    """Detect source language by CJK vs alphabetic character ratio.

    Returns 'zh' when ≥ 40% of characters are CJK, 'en' otherwise.
    """
    zh_count = 0
    total = 0
    for w in words:
        for ch in w.text:
            if is_cjk(ch):
                zh_count += 1
                total += 1
            elif ch.isalpha():
                total += 1
    if total == 0:
        return "en"
    return "zh" if zh_count / total >= 0.4 else "en"


# ── Character-level split for oversized cues ───────────────


def best_split_position(text: str) -> int:
    """Find best character-level split point near the midpoint."""
    if len(text) <= 1:
        return -1

    mid = len(text) // 2

    for pos in _scan_outward(mid, 30, len(text)):
        if text[pos] in SENTENCE_ENDS:
            return pos + 1

    for pos in _scan_outward(mid, 30, len(text)):
        if text[pos] == "\n":
            return pos + 1

    for pos in _scan_outward(mid, 30, len(text)):
        if text[pos] in CLAUSE_PUNCT:
            return pos + 1

    for pos in _scan_outward(mid, 30, len(text)):
        if text[pos] == " ":
            return pos + 1

    return -1


def _scan_outward(center: int, radius: int, length: int):
    for i in range(radius):
        p = center + i
        if 0 < p < length:
            yield p
        if i > 0:
            p = center - i
            if 0 < p < length:
                yield p
