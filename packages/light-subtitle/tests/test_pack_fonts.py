"""Tests for pack font patching before ffmpeg burn."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from light_subtitle.fonts import BILINGUAL_MARGIN_V
from light_subtitle.pack import PackConfig, run_pack

BILINGUAL_ASS = (
    "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Bilingual,EmbeddedFont,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
    "0,0,0,0,100,100,0,0,1,2,1,2,10,10,0,1\n\n[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:01.00,0:00:03.00,Bilingual,,0,0,0,,你好\\N{\\fs14}Hello\n"
)


def test_run_pack_patches_bilingual_ass_font(tmp_path: Path) -> None:
    """Pack must patch bilingual.ass Fontname from --font before ass= burn."""
    (tmp_path / "video.mp4").write_bytes(b"\x00")
    (tmp_path / "bilingual.ass").write_text(BILINGUAL_ASS, encoding="utf-8")

    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured_cmd.append(cmd)
        if "-y" in cmd:
            out = Path(cmd[cmd.index("-y") + 1])
            out.write_bytes(b"fake")

        class Result:
            returncode = 0

        return Result()

    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ffprobe.write_text("#!/bin/sh\n", encoding="utf-8")

    with (
        patch("light_subtitle.pack._find_ffmpeg_full", return_value=(fake_ffmpeg, fake_ffprobe)),
        patch("light_subtitle.pack._probe_video_bitrate", return_value=3000),
        patch("light_subtitle.pack.subprocess.run", side_effect=fake_run),
        patch("light_subtitle.pack.resolve_font", return_value="ResolvedFont"),
    ):
        run_pack(PackConfig(output_dir=str(tmp_path), font="CustomFont"))

    assert captured_cmd, "ffmpeg should have been invoked"
    filter_arg = captured_cmd[0][captured_cmd[0].index("-filter_complex") + 1]
    assert "bilingual.patched.ass" in filter_arg
    assert "ResolvedFont" not in filter_arg  # font is inside patched file, not filter string


def test_run_pack_srt_uses_bottom_margin(tmp_path: Path) -> None:
    (tmp_path / "video.mp4").write_bytes(b"\x00")
    (tmp_path / "zh.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\n你好\n", encoding="utf-8")

    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured_cmd.append(cmd)
        if "-y" in cmd:
            out = Path(cmd[cmd.index("-y") + 1])
            out.write_bytes(b"fake")

        class Result:
            returncode = 0

        return Result()

    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ffprobe.write_text("#!/bin/sh\n", encoding="utf-8")

    with (
        patch("light_subtitle.pack._find_ffmpeg_full", return_value=(fake_ffmpeg, fake_ffprobe)),
        patch("light_subtitle.pack._probe_video_bitrate", return_value=3000),
        patch("light_subtitle.pack.subprocess.run", side_effect=fake_run),
        patch("light_subtitle.pack.resolve_font", return_value="ResolvedFont"),
    ):
        run_pack(PackConfig(output_dir=str(tmp_path)))

    filter_arg = captured_cmd[0][captured_cmd[0].index("-filter_complex") + 1]
    assert f"MarginV={BILINGUAL_MARGIN_V}" in filter_arg
