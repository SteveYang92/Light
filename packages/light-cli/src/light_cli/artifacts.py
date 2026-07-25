"""Run-level artifact paths + ASR checkpoint artifacts.

Subtitle-domain paths and (de)serialization (plan / translations /
segment words / cues / sidecars) live in :mod:`light_subtitle.artifacts`;
this module keeps only the ASR checkpoint family and the run-level
layout (``transcript.json`` / ``cues.json``).

The names and byte-level JSON layouts here are an external contract —
resume (``state_hydrate``), the regression harness, and the web backend
depend on them.  Treat any change as a format change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from light_asr import checkpoints as asr_checkpoints
from light_models import Word, word_from_dict
from light_subtitle.artifacts import read_json

from .config import AsrEngine

if TYPE_CHECKING:
    from .config import SubtitleConfig

# ── Run-level filenames ─────────────────────────────────────────────────────

TRANSCRIPT_JSON = "transcript.json"  # write: export step; read: resume, QC, backend
CUES_JSON = "cues.json"  # write: export; read: TTS, pack tooling (not a player sidecar)


def transcript_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / TRANSCRIPT_JSON


def cues_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / CUES_JSON


def read_transcript_words(path: str | Path) -> list[Word]:
    return [word_from_dict(w) for w in read_json(path).get("words", [])]


# ── ASR checkpoint artifacts — asr/asr_<provider>.json naming ───────────────
#
# Path derivation and SubtitleConfig-bound wrappers; the byte-level schema is
# owned by :mod:`light_asr.checkpoints`.

WHISPER_CPP_RAW_SUFFIX = ".raw.json"


def asr_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "asr"


def provider_name(config: SubtitleConfig) -> str:
    return config.asr.value


def asr_words_path(config: SubtitleConfig) -> Path:
    return asr_dir(config.output_dir) / f"asr_{provider_name(config)}.json"


def asr_whisper_cpp_raw_path(config: SubtitleConfig) -> Path:
    return asr_dir(config.output_dir) / f"asr_{AsrEngine.WHISPER_CPP.value}{WHISPER_CPP_RAW_SUFFIX}"


def audio_wav_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "audio_asr.wav"


def save_asr_words(config: SubtitleConfig, words: list[Word]) -> Path:
    """Persist word-level ASR output to asr/asr_<provider>.json."""
    return asr_checkpoints.save_words_checkpoint(words, asr_words_path(config), provider_name(config))


def load_asr_words(config: SubtitleConfig) -> list[Word]:
    return asr_checkpoints.load_words_checkpoint(asr_words_path(config))


def save_whisper_cpp_raw(config: SubtitleConfig, whisper_json: Path) -> Path:
    """Copy whisper-cli raw JSON to canonical asr_whisper-cpp.raw.json."""
    return asr_checkpoints.save_whisper_cpp_raw(whisper_json, asr_whisper_cpp_raw_path(config))
