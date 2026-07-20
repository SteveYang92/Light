"""Tests for the boxed bilingual ASS builder (style/box.py).

Uses a FakeMeasurer so no real font file is needed: width is proportional to
character count, pitch proportional to size.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from light_models import SubtitleCue
from light_subtitle.pipeline.export import export_bilingual_ass
from light_subtitle.style.box import (
    EN_STYLE_NAME,
    ZH_STYLE_NAME,
    build_bilingual_boxed_events,
    rounded_rect_path,
    wrap_line,
)
from light_subtitle.style.config import PLAY_RES_Y, SubtitleStyleConfig


class FakeMeasurer:
    """Deterministic measurer: 0.5em per char, 1.2em line pitch."""

    def line_width(self, text: str, size: int) -> float:
        return len(text) * size * 0.5

    def line_pitch(self, size: int) -> float:
        return size * 1.2


MEASURER = FakeMeasurer()
CONFIG = SubtitleStyleConfig()


def _cue(text: str, start: float = 1.0, end: float = 3.0) -> SubtitleCue:
    return SubtitleCue(cue_id="c1", unit_id="u1", start=start, end=end, text=text, lang="zh")


def _box_events(events: list[str]) -> list[str]:
    return [e for e in events if ",Box," in e]


def _text_events(events: list[str], style: str) -> list[str]:
    return [e for e in events if f",{style}," in e]


def _pos_ys(events: list[str]) -> list[int]:
    """Extract the \\pos y of each text event (absolute placement, MarginV unused)."""
    return [int(re.search(r"\\pos\(-?\d+,(-?\d+)\)", e).group(1)) for e in events]


# ── rounded rect path ───────────────────────────────────


def test_rounded_rect_path_has_four_arcs_and_int_coords() -> None:
    path = rounded_rect_path(100, 50, 500, 150, 20)
    assert path.startswith("m ")
    assert path.count(" b ") == 4
    numbers = re.findall(r"-?\d+", path)
    assert numbers, "path should carry integer coordinates"
    assert len(numbers) == 2 * (1 + 1 + 3 + 1 + 3 + 1 + 3 + 1 + 3)  # m + 4×(l + b) + final l


def test_rounded_rect_radius_clamped_to_half_size() -> None:
    tiny = rounded_rect_path(0, 0, 10, 10, 999)
    assert tiny.startswith("m 5 0")


# ── wrapping ────────────────────────────────────────────


def test_wrap_breaks_latin_at_word_boundary() -> None:
    lines = wrap_line("hello world foo", MEASURER, size=10, max_width=len("hello ") * 5)
    assert lines == ["hello", "world", "foo"]


def test_wrap_breaks_cjk_per_character() -> None:
    lines = wrap_line("你好世界啊", MEASURER, size=10, max_width=10)  # 2 chars per line
    assert lines == ["你好", "世界", "啊"]


# ── event building ──────────────────────────────────────


def test_one_box_per_language_and_per_line_text_events() -> None:
    groups = [(_cue("你好\n世界"), "hello world", 1.0, 3.0)]
    events = build_bilingual_boxed_events(groups, "TestFont", MEASURER, CONFIG)

    boxes = _box_events(events)
    assert len(boxes) == 2  # one ZH box + one EN box
    assert all(e.startswith("Dialogue: 0,") for e in boxes)  # boxes on layer 0

    zh_texts = _text_events(events, ZH_STYLE_NAME)
    en_texts = _text_events(events, EN_STYLE_NAME)
    assert len(zh_texts) == 2  # two ZH lines → two events
    assert len(en_texts) == 1
    assert all(e.startswith("Dialogue: 1,") for e in zh_texts + en_texts)

    # Multi-line ZH: first reading line sits on top, so \pos y ascends in
    # emission order (smaller y = higher on screen).
    assert "你好" in zh_texts[0] and "世界" in zh_texts[1]
    ys = _pos_ys(zh_texts)
    assert ys == sorted(ys)
    # ZH block sits above EN block.
    assert max(_pos_ys(zh_texts)) < min(_pos_ys(en_texts))


def test_box_width_wraps_text() -> None:
    narrow = build_bilingual_boxed_events([(None, "hi", 0.0, 1.0)], "F", MEASURER, CONFIG)
    wide = build_bilingual_boxed_events([(None, "hi there friend", 0.0, 1.0)], "F", MEASURER, CONFIG)

    def en_box_width(events: list[str]) -> int:
        path = _box_events(events)[0].split("}", 1)[1]
        xs = [int(x) for x in re.findall(r"-?\d+", path)[0::2]]
        return max(xs) - min(xs)

    assert en_box_width(wide) > en_box_width(narrow)


def test_multi_line_box_is_taller() -> None:
    one = build_bilingual_boxed_events([(_cue("一行"), None, 0.0, 1.0)], "F", MEASURER, CONFIG)
    two = build_bilingual_boxed_events([(_cue("一行\n两行"), None, 0.0, 1.0)], "F", MEASURER, CONFIG)

    def zh_box_height(events: list[str]) -> int:
        path = _box_events(events)[0].split("}", 1)[1]
        ys = [int(y) for y in re.findall(r"-?\d+", path)[1::2]]
        return max(ys) - min(ys)

    assert zh_box_height(two) > zh_box_height(one)


def test_zh_only_and_en_only_groups() -> None:
    zh_only = build_bilingual_boxed_events([(_cue("只有中文"), None, 0.0, 1.0)], "F", MEASURER, CONFIG)
    assert len(_box_events(zh_only)) == 1
    assert _text_events(zh_only, EN_STYLE_NAME) == []

    en_only = build_bilingual_boxed_events([(None, "english only", 0.0, 1.0)], "F", MEASURER, CONFIG)
    assert len(_box_events(en_only)) == 1
    assert _text_events(en_only, ZH_STYLE_NAME) == []


def test_en_bottom_line_anchors_at_config_margin_v() -> None:
    events = build_bilingual_boxed_events([(None, "bottom line", 0.0, 1.0)], "F", MEASURER, CONFIG)
    # Bottom line's line-box bottom sits at PLAY_RES_Y - margin_v.
    assert _pos_ys(_text_events(events, EN_STYLE_NAME)) == [PLAY_RES_Y - CONFIG.margin_v]
    # EN box bottom stays inside the frame.
    path = _box_events(events)[0].split("}", 1)[1]
    ys = [int(y) for y in re.findall(r"-?\d+", path)[1::2]]
    assert max(ys) <= PLAY_RES_Y


def test_play_res_follows_non_16x9_frame() -> None:
    """Non-16:9 frames get a matching PlayResX so boxes are not anisotropically crushed."""
    from light_subtitle.style.config import play_res_for_frame

    assert play_res_for_frame(1920, 1080) == (1920, 1080)
    assert play_res_for_frame(3840, 2160) == (1920, 1080)
    # 3324×2160 (277:180) → PlayResX = round(1080 * 3324 / 2160) = 1662
    assert play_res_for_frame(3324, 2160) == (1662, 1080)

    play = play_res_for_frame(3324, 2160)
    events = build_bilingual_boxed_events([(None, "hello world", 0.0, 1.0)], "F", MEASURER, CONFIG, play_res=play)
    text = _text_events(events, EN_STYLE_NAME)[0]
    assert f"\\pos({play[0] // 2}," in text


def test_long_en_wraps_into_multiple_lines() -> None:
    long_en = "word " * 40
    events = build_bilingual_boxed_events([(None, long_en, 0.0, 1.0)], "F", MEASURER, CONFIG)
    assert len(_text_events(events, EN_STYLE_NAME)) > 1


# ── export integration ──────────────────────────────────


def _zh_cue(text: str = "你好", start: float = 1.0, end: float = 3.0) -> SubtitleCue:
    return SubtitleCue(cue_id="z1", unit_id="u1", start=start, end=end, text=text, lang="zh")


def _en_cue(text: str = "hello", start: float = 1.0, end: float = 3.0) -> SubtitleCue:
    return SubtitleCue(cue_id="e1", unit_id="u1", start=start, end=end, text=text, lang="en")


def test_export_bilingual_ass_boxed(tmp_path) -> None:
    out = tmp_path / "bilingual.ass"
    with patch("light_subtitle.pipeline.export.bilingual._resolved_font", return_value="TestFont"):
        export_bilingual_ass([_en_cue()], [_zh_cue()], str(out), font="TestFont", measurer=MEASURER)
    text = out.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in text
    assert f"Style: {ZH_STYLE_NAME},TestFont" in text
    assert "Dialogue: 0," in text  # box drawing
    assert "\\p1" in text  # vector drawing mode


def test_export_bilingual_ass_box_disabled_falls_back_to_plain(tmp_path) -> None:
    out = tmp_path / "bilingual.ass"
    with patch("light_subtitle.pipeline.export.bilingual._resolved_font", return_value="TestFont"):
        export_bilingual_ass(
            [_en_cue()],
            [_zh_cue()],
            str(out),
            font="TestFont",
            style=SubtitleStyleConfig(box_enabled=False),
        )
    text = out.read_text(encoding="utf-8")
    assert "Style: Bilingual,TestFont" in text
    assert "\\p1" not in text


def test_export_bilingual_ass_missing_font_file_falls_back(tmp_path) -> None:
    out = tmp_path / "bilingual.ass"
    with (
        patch("light_subtitle.pipeline.export.bilingual._resolved_font", return_value="TestFont"),
        patch("light_subtitle.style.fonts.resolve_font_file", return_value=None),
    ):
        export_bilingual_ass([_en_cue()], [_zh_cue()], str(out), font="TestFont")
    text = out.read_text(encoding="utf-8")
    assert "Style: Bilingual,TestFont" in text  # plain fallback style
    assert "\\p1" not in text


# ── resolve_font_file ───────────────────────────────────


def test_resolve_font_file_via_fc_match(tmp_path) -> None:
    from light_subtitle.style import fonts

    fake_font = tmp_path / "Fake.ttf"
    fake_font.write_bytes(b"x")

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        class Result:
            returncode = 0
            stdout = str(fake_font) + "\n"

        return Result()

    with (
        patch("light_subtitle.style.fonts.shutil.which", return_value="/usr/bin/fc-match"),
        patch("light_subtitle.style.fonts.subprocess.run", side_effect=fake_run),
    ):
        assert fonts.resolve_font_file("Whatever") == fake_font


def test_resolve_font_file_returns_none_when_unmatched() -> None:
    from light_subtitle.style import fonts

    with patch("light_subtitle.style.fonts.shutil.which", return_value=None):
        assert fonts.resolve_font_file("No Such Font XYZ") is None


@pytest.mark.parametrize("opacity,alpha", [(1.0, "00"), (0.0, "FF")])
def test_alpha_tag_extremes(opacity: float, alpha: str) -> None:
    from light_subtitle.style.box import _alpha_tag

    assert _alpha_tag(opacity) == f"\\1a&H{alpha}&"
