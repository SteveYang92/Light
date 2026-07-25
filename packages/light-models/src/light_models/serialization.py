"""Word (de)serialization — single 5-field JSON schema.

Shared by every writer (ASR checkpoints, transcript exports, segment words,
debug dumps).  Key order matters for byte-identical output.
"""

from __future__ import annotations

from .word import Word


def word_to_dict(word: Word) -> dict:
    """Serialize a :class:`Word` to the canonical 5-field dict."""
    return {
        "text": word.text,
        "start": word.start,
        "end": word.end,
        "confidence": word.confidence,
        "speaker": word.speaker,
    }


def word_from_dict(raw: dict) -> Word:
    """Build a Word, tolerating missing optional keys and debug-only extras
    (e.g. the ``changed`` flag in transcript_correct dumps)."""
    return Word(
        text=raw["text"],
        start=raw["start"],
        end=raw["end"],
        confidence=raw.get("confidence", 1.0),
        speaker=raw.get("speaker"),
    )
