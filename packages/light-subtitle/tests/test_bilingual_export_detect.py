"""Tests for bilingual export auto-detection on resume."""

from __future__ import annotations

from pathlib import Path

from light_subtitle.config import SubtitleConfig
from light_subtitle.orchestrator import Orchestrator
from light_subtitle.step_registry import _wants_bilingual_exports


def _orch(tmp_path: Path, *, bilingual: bool = False, slug: str = "demo") -> Orchestrator:
    config = SubtitleConfig(
        input_path=str(tmp_path / "video.mp4"),
        output_dir=str(tmp_path),
        target_lang="zh",
        bilingual=bilingual,
        slug=slug,
    )
    return Orchestrator(config)


def test_wants_bilingual_exports_from_config_flag(tmp_path: Path) -> None:
    assert _wants_bilingual_exports(_orch(tmp_path, bilingual=True)) is True


def test_wants_bilingual_exports_from_existing_ass(tmp_path: Path) -> None:
    (tmp_path / "bilingual.ass").write_text("[Script Info]", encoding="utf-8")
    assert _wants_bilingual_exports(_orch(tmp_path)) is True


def test_wants_bilingual_exports_from_existing_en_track(tmp_path: Path) -> None:
    (tmp_path / "en.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
    assert _wants_bilingual_exports(_orch(tmp_path)) is True


def test_wants_bilingual_exports_monolingual(tmp_path: Path) -> None:
    (tmp_path / "zh.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
    assert _wants_bilingual_exports(_orch(tmp_path)) is False
