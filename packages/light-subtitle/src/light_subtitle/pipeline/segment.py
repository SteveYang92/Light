"""Segment — split word sequences into semantically-aware audio segments.

This module splits contiguous ASR word streams into ``Segment`` objects
using a six-tier scoring system rather than a hard gap threshold.

Scoring hierarchy (strictly ordered, higher tier signals dominate):

    Tier 1 (~100): Sentence boundaries  — sentence-ending punctuation
    Tier 2 (~80):  Semantic boundaries   — uppercase start, discourse
                  markers, long fragment + substantial pause
    Tier 3 (10-60): Pause signals        — gap duration (supporting only)
    Tier 4 (~50):  Clause boundaries     — comma / semicolon
    Tier 5 (~30):  Phrase boundaries     — conjunction before
    Tier 6 (forced): Constraint overflow — duration / character limits

Lower-tier scores do NOT stack with higher-tier: a long pause (Tier 3, 60)
never outweighs a period (Tier 1, 100).  Speaker change always splits.
Netflix §4 word pairs (article→noun, etc.) are forbidden.
"""

from __future__ import annotations

from light_models import Segment, Word, is_cjk
from light_models.punctuation import CLAUSE_PUNCT, EN_TRAILING_PUNCT, SENTENCE_ENDS

from ..language import is_sentence_end
from ..language.english import (
    ARTICLES,
    CONJUNCTIONS,
    DISCOURSE_MARKERS,
    PREPOSITIONS,
    _is_forbidden_split,
)

SPLIT_THRESHOLD = 40

# ═══════════════════════════════════════════════════════════════════
# Tier score constants
# ═══════════════════════════════════════════════════════════════════

TIER_SENTENCE = 100  # sentence-ending .!?。！？
TIER_SEMANTIC = 80  # uppercase start / discourse marker / long fragment + pause
TIER_PAUSE_LONG = 60  # gap > 0.8s
TIER_PAUSE_MEDIUM = 45  # gap > 0.5s
TIER_PAUSE_SHORT = 25  # gap > 0.3s
TIER_PAUSE_BRIEF = 10  # gap > 0.15s
TIER_CLAUSE = 50  # comma / semicolon
TIER_PHRASE = 30  # conjunction before
TIER_FORCE = 100  # max duration / char overflow

# Negative (continuation) signals
CONTINUATION_LOWERCASE = -30  # next word starts lowercase → mid-sentence
CONTINUATION_PREP_ARTICLE = -25  # next word is preposition / article
CONTINUATION_SHORT_FRAGMENT = -40  # buffer ≤ 3 words, no sentence-ending punct

# ═══════════════════════════════════════════════════════════════════
# Tier score constants
# ═══════════════════════════════════════════════════════════════════


def _score_split(
    last_word: Word,
    next_word: Word,
    gap: float,
    buffer_words: list[Word],
    max_duration: float,
    max_chars: int,
) -> tuple[int, bool]:  # (score, is_forbidden)
    """Score a potential split between *last_word* and *next_word*.

    Returns (score, is_forbidden).  The caller splits when score >=
    SPLIT_THRESHOLD and is_forbidden is False.

    Only the single highest positive tier contributes its score
    (no stacking across tiers), ensuring the hierarchy holds:
    a long pause (Tier 3, 60) never outweighs a period (Tier 1, 100).
    """
    last_text = last_word.text.strip()
    next_text = next_word.text.strip()
    next_clean = next_text.lower().rstrip(EN_TRAILING_PUNCT)

    # ── Netflix §4: never split forbidden pairs ──
    if _is_forbidden_split(last_text, next_text):
        return (0, True)

    score = 0

    # ── Tier 1: Sentence boundary (~100) ──
    if is_sentence_end(last_text):
        score = TIER_SENTENCE

    # ── Tier 6: Max constraint overflow (forced split) ──
    if score == 0:
        buffer_dur = buffer_words[-1].end - buffer_words[0].start if buffer_words else 0
        if buffer_dur + gap > max_duration:
            score = TIER_FORCE
        else:
            total_chars = sum(len(w.text.strip()) for w in buffer_words) + len(next_text)
            if total_chars > max_chars * 3 and gap > 0.15:
                score = 70  # below TIER_FORCE but above TIER_SEMANTIC

    # ── Tier 2: Semantic boundary (~80) ──
    if score == 0:
        if next_text and next_text[0].isupper() and gap > 0.3:
            score = TIER_SEMANTIC
        elif next_clean in DISCOURSE_MARKERS and gap > 0.3:
            score = 75
        elif len(buffer_words) >= 5:
            has_any_punct = any(any(ch in CLAUSE_PUNCT + SENTENCE_ENDS for ch in w.text) for w in buffer_words)
            if not has_any_punct and gap > 0.5:
                score = 70

    # ── Tier 4: Clause boundary (~50) ──
    if score == 0:
        if any(ch in CLAUSE_PUNCT for ch in last_text):
            score = TIER_CLAUSE

    # ── Tier 3: Pause signals (10-60) ──
    if score == 0:
        if gap > 0.8:
            score = TIER_PAUSE_LONG
        elif gap > 0.5:
            score = TIER_PAUSE_MEDIUM
        elif gap > 0.3:
            score = TIER_PAUSE_SHORT
        elif gap > 0.15:
            score = TIER_PAUSE_BRIEF

    # ── Tier 5: Phrase boundary (~30) ──
    if score == 0:
        if next_clean in CONJUNCTIONS:
            score = TIER_PHRASE

    # ── Negative signals (continuation) ──
    negative = 0
    if next_text and next_text[0].islower():
        negative += CONTINUATION_LOWERCASE
    if next_clean in PREPOSITIONS | ARTICLES:
        negative += CONTINUATION_PREP_ARTICLE
    if len(buffer_words) <= 3:
        has_punct = any(any(ch in CLAUSE_PUNCT + SENTENCE_ENDS for ch in w.text) for w in buffer_words)
        if not has_punct:
            negative += CONTINUATION_SHORT_FRAGMENT

    return (score + negative, False)


# ═══════════════════════════════════════════════════════════════════
# Text joining
# ═══════════════════════════════════════════════════════════════════


def _is_cjk_or_kana(ch: str) -> bool:
    return is_cjk(ch) or "\u3040" <= ch <= "\u30ff"


def _join_text(words: list[Word]) -> str:
    if not words:
        return ""
    sample = "".join(w.text for w in words[:10])
    cjk_count = sum(1 for ch in sample if _is_cjk_or_kana(ch))
    if cjk_count > len(sample) * 0.3:
        return "".join(w.text.strip() for w in words)
    return " ".join(t for w in words if (t := w.text.strip()))


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════


def run(
    words: list[Word],
    max_duration: float = 7.0,
    max_chars_per_line: int = 42,
) -> list[Segment]:
    """Split word stream into semantically-aware segments.

    Uses a six-tier scoring system: sentence boundary > semantic >
    pause > clause > phrase > constraint overflow.  Gap duration
    serves as a supporting signal rather than the primary criterion.

    Speaker changes always split.  Netflix §4 word pairs (article →
    noun, auxiliary → verb, etc.) are never split.
    """
    if not words:
        return []

    segments: list[Segment] = []
    current_words: list[Word] = []
    unit_idx = 0

    for i, w in enumerate(words):
        current_words.append(w)

        split = False

        # Speaker change always triggers a split.
        if i < len(words) - 1 and w.speaker and words[i + 1].speaker and w.speaker != words[i + 1].speaker:
            split = True

        elif i < len(words) - 1:
            gap = words[i + 1].start - w.end
            next_word = words[i + 1]

            score, forbidden = _score_split(
                w,
                next_word,
                gap,
                current_words,
                max_duration,
                max_chars_per_line,
            )

            if not forbidden and score >= SPLIT_THRESHOLD:
                split = True

        # End of word list — always emit final segment.
        if split or i == len(words) - 1:
            segment = Segment(
                unit_id=f"u{unit_idx:04d}",
                start=current_words[0].start,
                end=current_words[-1].end,
                speaker=current_words[0].speaker or "",
                source_text=_join_text(current_words),
                words=list(current_words),
            )
            segments.append(segment)
            current_words = []
            unit_idx += 1

    return segments
