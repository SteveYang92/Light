"""ASR checkpoint artifacts — asr/asr_<provider>.json naming."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from light_models import Word

from ...config import AsrEngine, SubtitleConfig

# whisper.cpp raw segment JSON (alignment anchors) — separate from words checkpoint.
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
    path = asr_words_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": "light-asr-words.v1",
        "provider": provider_name(config),
        "words": [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
                "speaker": w.speaker,
            }
            for w in words
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_asr_words(config: SubtitleConfig) -> list[Word]:
    path = asr_words_path(config)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Word(**w) for w in data.get("words", [])]


def save_whisper_cpp_raw(config: SubtitleConfig, whisper_json: Path) -> Path:
    """Copy whisper-cli raw JSON to canonical asr_whisper-cpp.raw.json."""
    dest = asr_whisper_cpp_raw_path(config)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(whisper_json, dest)
    return dest
