"""Provider Protocols — the three pluggable dimensions of ASR.

Documentation-only: these Protocols describe the expected call shapes but
are not enforced at runtime.  The built-in providers are plain functions
(``whisperx.run``, ``whisper_cpp.transcribe``, ``align.align_words``,
``diarize.run``) that satisfy them structurally.
"""

from __future__ import annotations

from typing import Protocol

from light_models import Word


class Transcriber(Protocol):
    """Audio → word list (e.g. whisperX pipeline, whisper.cpp binary)."""

    def __call__(self, audio_path: str, *, language: str) -> list[Word]: ...


class Aligner(Protocol):
    """Correct word timestamps via forced alignment against the audio."""

    def __call__(self, words: list[Word], audio_path: str, *, language: str) -> list[Word]: ...


class Diarizer(Protocol):
    """Assign speaker labels to words based on "who speaks when"."""

    def __call__(self, words: list[Word], audio_path: str) -> list[Word]: ...
