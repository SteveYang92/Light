"""Compose — merge fragmentary audio segments into complete translation units.

``segment.run()`` splits at silence gaps > 0.5 s and sentence-ending
punctuation.  The resulting segments are driven by English speech rhythm,
not by semantic completeness.  For example::

    u0041: "These are not just"              (mid-sentence pause → fragment)
    u0042: "coding assists now where you…"   (continuation)

When translated independently, these produce broken Chinese like
"这不再是" / "那种你让它解决特定问题…".

This module merges such fragments into complete sentences so the LLM
translates whole thoughts.

Usage::
    from .compose import compose_segments
    composed = compose_segments(segments)
"""

from __future__ import annotations

from light_models import Segment, Word

from ...language import is_sentence_end

# If the gap between two segments exceeds this, they are separate
# thoughts and should not be merged regardless of fragment status.
_MERGE_GAP_MAX = 3.0  # seconds

# Segments ≤ this many words that don't end with sentence-ending
# punctuation are always merged forward (e.g. "Well,", "So,").
_MIN_WORDS_FOR_AUTO_MERGE = 3


# ── Fragment detection ──────────────────────────────────────────────


def _is_fragment(segment: Segment) -> bool:
    """True when *segment* does not end with sentence-ending punctuation
    — the speaker was mid-sentence."""
    return not is_sentence_end(segment.source_text)


# ── Merge eligibility ──────────────────────────────────────────────


def _should_merge(buffer: list[Segment], candidate: Segment) -> bool:
    """Decide whether *candidate* should be appended to *buffer*."""
    if not buffer:
        return False

    # Gap too large — separate thought.
    gap = candidate.start - buffer[-1].end
    if gap > _MERGE_GAP_MAX:
        return False

    # Speaker change — keep separate.
    if candidate.speaker and buffer[-1].speaker and candidate.speaker != buffer[-1].speaker:
        return False

    # Buffer's last segment is a fragment (no sentence-ending punct).
    if _is_fragment(buffer[-1]):
        return True

    # Candidate is very short and doesn't end a sentence (e.g. "Well,"
    # as a standalone segment).  Merge it forward.
    if len(candidate.source_text.split()) <= _MIN_WORDS_FOR_AUTO_MERGE and not is_sentence_end(candidate.source_text):
        return True

    return False


# ── Combining ──────────────────────────────────────────────────────


def _combine(segments: list[Segment]) -> Segment:
    """Merge a list of consecutive segments into one."""
    all_words: list[Word] = []
    parts: list[str] = []

    for s in segments:
        all_words.extend(s.words)
        parts.append(s.source_text.strip())

    # Build combined text with a single space between segments.
    combined = " ".join(p for p in parts if p)

    return Segment(
        unit_id=f"m{segments[0].unit_id}_{segments[-1].unit_id}",
        start=segments[0].start,
        end=segments[-1].end,
        speaker=segments[0].speaker or "",
        source_text=combined,
        words=all_words,
    )


# ── Public API ─────────────────────────────────────────────────────


def compose_segments(segments: list[Segment]) -> list[Segment]:
    """Merge fragmentary audio segments into complete translation units.

    Input segments are produced by ``segment.run()`` — they are driven
    by audio pauses and punctuation, not by semantic completeness.

    Unlike the previous algorithm that stopped merging at ``max_duration``
    (producing mid-sentence cuts), this version accumumulates fragments
    until a semantically complete sentence is formed.  Duration constraints
    are handled downstream by ``_split_overlong_units()`` which uses the
    LLM to split overlong semantic units at natural break points.

    Algorithm:
      1. Walk left-to-right, accumulating fragments into a buffer.
      2. A fragment (no sentence-ending punctuation) always merges forward.
      3. A complete sentence stops accumulation — emit the buffer.
    Example::

        [u0039 "Well,", u0040 "the agents are really working."]
        → [m0039_0040 "Well, the agents are really working."]

    Returns a new list of ``Segment`` objects with merged ``unit_id`` IDs
    (e.g. ``"m0039_0040"``) and combined word-level timestamps.
    Original segments are not modified.
    """
    if not segments:
        return []

    result: list[Segment] = []
    buffer: list[Segment] = [segments[0]]

    for i in range(1, len(segments)):
        if _should_merge(buffer, segments[i]):
            buffer.append(segments[i])
        else:
            result.append(_combine(buffer))
            buffer = [segments[i]]

    if buffer:
        result.append(_combine(buffer))

    return result
