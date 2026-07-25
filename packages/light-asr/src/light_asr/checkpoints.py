"""Word checkpoint serialization — ``light-asr-words.v1`` JSON format.

Pure (de)serialization only: callers own path derivation.  The on-disk
schema and key order are byte-compatible with the checkpoints previously
written by ``light_cli.pipeline.asr.artifacts``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from light_models import Word, word_from_dict, word_to_dict

WORDS_FORMAT = "light-asr-words.v1"

# whisper.cpp raw segment JSON (alignment anchors) — separate from words checkpoint.
WHISPER_CPP_RAW_SUFFIX = ".raw.json"

__all__ = [
    "WORDS_FORMAT",
    "WHISPER_CPP_RAW_SUFFIX",
    "load_words_checkpoint",
    "save_whisper_cpp_raw",
    "save_words_checkpoint",
    "word_from_dict",
    "word_to_dict",
]


# ── Checkpoint read/write ───────────────────────────────────────────────────


def save_words_checkpoint(words: list[Word], path: str | Path, provider: str) -> Path:
    """Persist word-level ASR output as ``light-asr-words.v1`` JSON."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": WORDS_FORMAT,
        "provider": provider,
        "words": [word_to_dict(w) for w in words],
    }
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def load_words_checkpoint(path: str | Path) -> list[Word]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [word_from_dict(w) for w in data.get("words", [])]


def save_whisper_cpp_raw(src: str | Path, dest: str | Path) -> Path:
    """Copy whisper-cli raw JSON to its canonical checkpoint location."""
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return target
