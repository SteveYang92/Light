"""video.-prefixed player sidecars — rename helpers stay removed."""

from __future__ import annotations

from pathlib import Path

from light_subtitle import artifacts


def test_rename_outputs_helpers_removed() -> None:
    """CLI no longer slug-prefixes exports; helpers must stay gone."""
    import light_cli.cli as cli

    assert not hasattr(cli, "_rename_outputs")
    assert not hasattr(cli, "_has_generic_outputs")


def test_sidecar_name_matches_video_stem() -> None:
    assert artifacts.sidecar_name("zh.srt") == "video.zh.srt"
    assert artifacts.sidecar_path(Path("/out"), "bilingual.ass") == Path("/out/video.bilingual.ass")


def test_find_sidecar_prefers_video_then_bare(tmp_path: Path) -> None:
    bare = tmp_path / "zh.srt"
    bare.write_text("bare", encoding="utf-8")
    assert artifacts.find_sidecar(tmp_path, "zh.srt") == bare

    preferred = tmp_path / "video.zh.srt"
    preferred.write_text("video", encoding="utf-8")
    assert artifacts.find_sidecar(tmp_path, "zh.srt") == preferred


def test_migrate_legacy_sidecars_renames_annotations(tmp_path: Path) -> None:
    (tmp_path / "annotations.ass").write_text("ass", encoding="utf-8")
    (tmp_path / "annotations.vtt").write_text("vtt", encoding="utf-8")
    (tmp_path / "video.zh.srt").write_text("zh", encoding="utf-8")
    (tmp_path / "zh.srt").write_text("old-zh", encoding="utf-8")

    changed = artifacts.migrate_legacy_sidecars(tmp_path)
    assert "annotations.ass" in changed
    assert "annotations.vtt" in changed
    assert "zh.srt" in changed
    assert (tmp_path / "video.annotations.ass").read_text(encoding="utf-8") == "ass"
    assert (tmp_path / "video.annotations.vtt").read_text(encoding="utf-8") == "vtt"
    assert not (tmp_path / "annotations.ass").exists()
    assert not (tmp_path / "zh.srt").exists()
    assert (tmp_path / "video.zh.srt").read_text(encoding="utf-8") == "zh"
