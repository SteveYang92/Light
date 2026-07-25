"""ASR configuration — engine selection and provider parameters.

Defaults mirror the ASR fields of ``light_cli.config.SubtitleConfig``
so the orchestration layer can construct an :class:`AsrConfig` field-by-field
without changing behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum


class AsrEngine(StrEnum):
    """ASR engine type."""

    WHISPERX = "whisperx"
    WHISPER_CPP = "whisper-cpp"


@dataclass
class AsrConfig:
    """Parameters for one ASR run (engine, whisper.cpp paths, diarization)."""

    engine: AsrEngine = AsrEngine.WHISPERX
    whisper_model: str = "ggml-large-v3-turbo.bin"
    whisper_path: str = "whisper-cli"
    language: str = "auto"
    diarize: bool = False
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
