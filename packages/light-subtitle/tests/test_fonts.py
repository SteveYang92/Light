"""Tests for font resolution and ASS style patching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from light_subtitle.fonts import (
    FontConfig,
    patch_ass_styles,
    resolve_font,
    write_patched_ass,
)

SAMPLE_ASS = (
    "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Bilingual,OldFont,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
    "0,0,0,0,100,100,0,0,1,2,1,2,10,10,0,1\n"
    "Style: Annotation,OldFont,40,&H00FFFFFF,&H00000000,&H00000000,&H00000000,"
    "-1,0,0,0,100,100,0,0,1,3,2,7,10,500,10,1\n\n[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:01.00,0:00:03.00,Bilingual,,0,0,0,,中文\\N{\\fs14}English\n"
)


def test_patch_ass_styles_replaces_fontname() -> None:
    patched = patch_ass_styles(SAMPLE_ASS, "NewFont")
    assert "Style: Bilingual,NewFont," in patched
    assert "Style: Annotation,NewFont," in patched
    assert "OldFont" not in patched
    assert "{\\fs14}English" in patched


def test_patch_ass_styles_filters_by_style_name() -> None:
    patched = patch_ass_styles(SAMPLE_ASS, "NewFont", style_names={"Bilingual"})
    assert "Style: Bilingual,NewFont," in patched
    assert "Style: Annotation,OldFont," in patched


def test_write_patched_ass(tmp_path: Path) -> None:
    src = tmp_path / "in.ass"
    dst = tmp_path / "out.ass"
    src.write_text(SAMPLE_ASS, encoding="utf-8")
    write_patched_ass(src, "PatchedFont", dst)
    text = dst.read_text(encoding="utf-8")
    assert "Style: Bilingual,PatchedFont," in text


def test_resolve_font_without_fc_match_returns_primary() -> None:
    with patch("light_subtitle.fonts.shutil.which", return_value=None):
        assert resolve_font(FontConfig(primary="My Preferred")) == "My Preferred"


def test_resolve_font_uses_fc_match_chain() -> None:
    def fake_fc_match(cmd: list[str], **kwargs: object) -> object:
        candidate = cmd[-1]

        class Result:
            returncode = 0
            stdout = "Unknown\n"

        if candidate == "Noto Sans CJK SC":
            Result.stdout = "Noto Sans CJK SC\n"
        return Result()

    with (
        patch("light_subtitle.fonts.shutil.which", return_value="/usr/bin/fc-match"),
        patch("light_subtitle.fonts.subprocess.run", side_effect=fake_fc_match),
    ):
        resolved = resolve_font(FontConfig(primary="MissingFont"))
        assert resolved == "Noto Sans CJK SC"


def test_parse_fc_family_takes_first_alias_only() -> None:
    from light_subtitle.fonts import _parse_fc_family

    assert _parse_fc_family("PingFang SC,蘋方-簡,苹方-简") == "PingFang SC"
    assert _parse_fc_family("Noto Sans CJK SC") == "Noto Sans CJK SC"


def test_fc_match_family_strips_comma_aliases() -> None:
    from light_subtitle.fonts import _fc_match_family

    with patch("light_subtitle.fonts.subprocess.run") as mock_run:

        class Result:
            returncode = 0
            stdout = "PingFang SC,蘋方-簡,苹方-简\n"

        mock_run.return_value = Result()
        assert _fc_match_family("PingFang SC") == "PingFang SC"


def test_bilingual_style_line_has_no_extra_commas_in_font_field() -> None:
    from light_subtitle.fonts import bilingual_style_line

    line = bilingual_style_line("PingFang SC")
    fields = line.removeprefix("Style:").split(",", 3)
    assert fields[1].strip() == "PingFang SC"


def test_candidate_chain_deduplicates_primary_in_fallbacks() -> None:
    calls: list[str] = []

    def fake_fc_match(cmd: list[str], **kwargs: object) -> object:
        candidate = cmd[-1]
        calls.append(candidate)

        class Result:
            returncode = 1
            stdout = ""

        return Result()

    with (
        patch("light_subtitle.fonts.shutil.which", return_value="/usr/bin/fc-match"),
        patch("light_subtitle.fonts.subprocess.run", side_effect=fake_fc_match),
    ):
        resolve_font(FontConfig(primary="PingFang SC"))
        assert calls[0] == "PingFang SC"
        assert "PingFang SC" not in calls[1:]
