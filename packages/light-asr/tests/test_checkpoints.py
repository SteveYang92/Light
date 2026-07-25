"""Tests for word checkpoint serialization (light-asr-words.v1)."""

from __future__ import annotations

import json

from light_asr.checkpoints import (
    load_words_checkpoint,
    save_whisper_cpp_raw,
    save_words_checkpoint,
    word_from_dict,
    word_to_dict,
)
from light_models import Word


def _words() -> list[Word]:
    return [
        Word(text=" hello", start=0.0, end=0.5, confidence=0.9),
        Word(text=" world", start=0.5, end=1.0, confidence=0.8, speaker="SPEAKER_00"),
    ]


def test_word_dict_roundtrip():
    word = _words()[1]
    assert word_from_dict(word_to_dict(word)) == word


def test_word_from_dict_tolerates_missing_optional_keys():
    word = word_from_dict({"text": " hi", "start": 0.0, "end": 0.2})
    assert word.confidence == 1.0
    assert word.speaker is None


def test_checkpoint_roundtrip(tmp_path):
    path = save_words_checkpoint(_words(), tmp_path / "asr" / "asr_whisperx.json", provider="whisperx")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "light-asr-words.v1"
    assert data["provider"] == "whisperx"
    assert list(data["words"][0].keys()) == ["text", "start", "end", "confidence", "speaker"]

    loaded = load_words_checkpoint(path)
    assert loaded == _words()


def test_save_whisper_cpp_raw_copies(tmp_path):
    src = tmp_path / "whisper_output.json"
    src.write_text('{"transcription": []}', encoding="utf-8")
    dest = save_whisper_cpp_raw(src, tmp_path / "asr" / "asr_whisper-cpp.raw.json")
    assert dest.read_text(encoding="utf-8") == '{"transcription": []}'
