"""Segment — pause-based audio segment from ASR output.

This is NOT a semantic unit.  It is produced by ``segment.run()`` which
splits word sequences at silence gaps > 0.5 s and sentence-ending
punctuation.  For translation purposes these segments are often too
fine-grained; use ``compose_units()`` to merge them into complete
translation units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .word import Word


@dataclass
class Segment:
    """A contiguous block of words delimited by audio pauses or punctuation.

    Attributes:
        unit_id:    Unique identifier (e.g. ``"u0001"``).
        start:      Start time of the first word (seconds).
        end:        End time of the last word (seconds).
        speaker:    Speaker label if available.
        source_text: Joined text of all words in this segment.
        words:      Individual word-level ASR output.
        source_cue_ids: References to source subtitle cues (used in QC).
    """

    unit_id: str
    start: float
    end: float
    speaker: str
    source_text: str
    words: list[Word]
    source_cue_ids: list[str] = field(default_factory=list)
