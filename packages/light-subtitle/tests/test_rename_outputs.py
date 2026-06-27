"""Tests for CLI output renaming after short-video pipeline runs."""

from __future__ import annotations

from pathlib import Path

from light_subtitle.cli import _has_generic_outputs, _rename_outputs


def test_has_generic_outputs_detects_bare_export_files(tmp_path: Path) -> None:
    assert not _has_generic_outputs(tmp_path)
    (tmp_path / "my_video.zh.srt").write_text("1", encoding="utf-8")
    assert not _has_generic_outputs(tmp_path)
    (tmp_path / "bilingual.ass").write_text("[Script Info]", encoding="utf-8")
    assert _has_generic_outputs(tmp_path)
    (tmp_path / "bilingual.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
    assert _has_generic_outputs(tmp_path)


def test_rename_outputs_includes_bilingual_vtt(tmp_path: Path) -> None:
    slug = "demo"
    (tmp_path / f"{slug}.bilingual.vtt").write_text("old", encoding="utf-8")
    vtt = "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n你好\nhi\n\n"
    (tmp_path / "bilingual.vtt").write_text(vtt, encoding="utf-8")

    _rename_outputs(tmp_path, slug)

    assert not (tmp_path / "bilingual.vtt").exists()
    assert "你好" in (tmp_path / f"{slug}.bilingual.vtt").read_text(encoding="utf-8")


def test_rename_outputs_overwrites_existing_slug_files(tmp_path: Path) -> None:
    """Resume-from-subtitle re-exports bare names; rename must refresh slug files."""
    slug = "demo"
    (tmp_path / f"{slug}.bilingual.ass").write_text("old", encoding="utf-8")
    (tmp_path / "bilingual.ass").write_text("new", encoding="utf-8")
    (tmp_path / "zh.srt").write_text("zh", encoding="utf-8")

    _rename_outputs(tmp_path, slug)

    assert not (tmp_path / "bilingual.ass").exists()
    assert not (tmp_path / "zh.srt").exists()
    assert (tmp_path / f"{slug}.bilingual.ass").read_text(encoding="utf-8") == "new"
    assert (tmp_path / f"{slug}.zh.srt").read_text(encoding="utf-8") == "zh"
