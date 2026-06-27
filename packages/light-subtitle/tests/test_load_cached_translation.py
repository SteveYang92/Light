"""Tests for load_cached_translation word re-attachment from compose/."""

from __future__ import annotations

import json
from pathlib import Path

from light_models import Segment, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate import load_cached_translation, save_segment_words


def test_load_cached_translation_attaches_words_from_compose_dir(tmp_path: Path) -> None:
    compose_dir = tmp_path / "compose"
    tx_dir = tmp_path / "translations"
    compose_dir.mkdir()
    tx_dir.mkdir()

    seg = Segment(
        unit_id="mu0001_u0002_0",
        start=1.0,
        end=3.5,
        speaker="",
        source_text="hello world",
        words=[
            Word(text="hello", start=1.0, end=2.0, confidence=1.0),
            Word(text="world", start=2.0, end=3.5, confidence=1.0),
        ],
    )
    save_segment_words([seg], compose_dir)

    raw = [
        {
            "cue_id": "zh_0000",
            "unit_id": "mu0001_u0002_0",
            "start": 1.0,
            "end": 3.5,
            "text": "你好世界",
            "lang": "zh",
        }
    ]
    (tx_dir / "raw.json").write_text(json.dumps(raw), encoding="utf-8")

    cues, usage = load_cached_translation(tx_dir, SubtitleConfig(input_path="dummy.mp4", target_lang="zh"))
    assert usage is None
    assert len(cues) == 1
    assert len(cues[0].words) == 2
    assert cues[0].words[0].text == "hello"
    assert cues[0].words[-1].end == 3.5


def test_load_cached_translation_ignores_stale_translations_segment_words(tmp_path: Path) -> None:
    """Legacy ``translations/segment_words.json`` must not be used."""
    compose_dir = tmp_path / "compose"
    tx_dir = tmp_path / "translations"
    compose_dir.mkdir()
    tx_dir.mkdir()

    (tx_dir / "raw.json").write_text(
        json.dumps(
            [
                {
                    "cue_id": "zh_0000",
                    "unit_id": "u0",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "测试",
                    "lang": "zh",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tx_dir / "segment_words.json").write_text(
        json.dumps({"u0": [{"text": "stale", "start": 0.0, "end": 1.0, "confidence": 1.0, "speaker": None}]}),
        encoding="utf-8",
    )

    cues, _ = load_cached_translation(tx_dir, SubtitleConfig(input_path="dummy.mp4", target_lang="zh"))
    assert cues[0].words == []
