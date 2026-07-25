"""Tests for annotation persistence and resume hydrate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from light_cli.config import SubtitleConfig
from light_cli.state_hydrate import hydrate_annotations
from light_models import SubtitleCue
from light_subtitle import artifacts


def test_save_and_load_annotations(tmp_path: Path) -> None:
    artifacts.save_annotations(tmp_path, {"u1": "术语：解释", "u2": ""})
    loaded = artifacts.load_annotations(tmp_path)
    assert loaded == {"u1": "术语：解释"}


def test_hydrate_annotations_from_disk(tmp_path: Path) -> None:
    artifacts.save_annotations(tmp_path, {"a": "注解A"})
    orch = MagicMock()
    orch.config = SubtitleConfig(input_path="x.mp4", output_dir=str(tmp_path), target_lang="zh", annotate=True)
    orch.state.annotations = {}
    orch.state.translated_cues = []

    hydrate_annotations(orch)
    assert orch.state.annotations == {"a": "注解A"}


def test_hydrate_annotations_from_cues_when_no_file(tmp_path: Path) -> None:
    orch = MagicMock()
    orch.config = SubtitleConfig(input_path="x.mp4", output_dir=str(tmp_path), target_lang="zh", annotate=True)
    orch.state.annotations = {}
    orch.state.translated_cues = [
        SubtitleCue(
            cue_id="1",
            unit_id="u1",
            start=0,
            end=1,
            text="你好",
            lang="zh",
            annotation="你好：greeting",
        ),
        SubtitleCue(cue_id="2", unit_id="u2", start=1, end=2, text="世界", lang="zh"),
    ]

    hydrate_annotations(orch)
    assert orch.state.annotations == {"u1": "你好：greeting"}
