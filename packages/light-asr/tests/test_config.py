"""Tests for AsrConfig defaults and AsrEngine."""

from __future__ import annotations

import os

from light_asr import AsrConfig, AsrEngine


def test_engine_values():
    assert AsrEngine.WHISPERX == "whisperx"
    assert AsrEngine.WHISPER_CPP == "whisper-cpp"


def test_config_defaults():
    cfg = AsrConfig()
    assert cfg.engine == AsrEngine.WHISPERX
    assert cfg.whisper_model == "ggml-large-v3-turbo.bin"
    assert cfg.whisper_path == "whisper-cli"
    assert cfg.language == "auto"
    assert cfg.diarize is False
    assert cfg.diarize_model == "pyannote/speaker-diarization-community-1"
    assert cfg.hf_token == os.environ.get("HF_TOKEN", "")


def test_config_hf_token_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert AsrConfig().hf_token == "test-token"
