"""Tests for segment output merging (bilingual ASS cue-group handling)."""

from __future__ import annotations

from pathlib import Path

from light_subtitle.merge_outputs import _merge_bilingual_ass

_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1920\n"
    "PlayResY: 1080\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def _cue_group(start: str, end: str, en: str, zh: str) -> str:
    """Build the four Dialogue events of one bilingual display cue (box+EN+box+ZH)."""
    return (
        f"Dialogue: 0,{start},{end},Box,,0,0,0,,{{\\p1}}m 0 0 l 10 0 10 10 0 10\n"
        f"Dialogue: 1,{start},{end},BilingualEn,,0,0,0,,{en}\n"
        f"Dialogue: 0,{start},{end},Box,,0,0,0,,{{\\p1}}m 0 0 l 20 0 20 20 0 20\n"
        f"Dialogue: 1,{start},{end},BilingualZh,,0,0,0,,{zh}\n"
    )


def test_bilingual_ass_merge_keeps_box_and_both_languages(tmp_path: Path):
    seg = tmp_path / ".seg1"
    seg.mkdir()
    (seg / "video.bilingual.ass").write_text(
        _HEADER
        + _cue_group("0:00:01.00", "0:00:03.00", "hello there", "你好")
        + _cue_group("0:00:04.00", "0:00:06.00", "second cue", "第二条"),
        encoding="utf-8",
    )

    _merge_bilingual_ass(tmp_path, [seg], [0.0], [30.0], None, "slug")

    merged = (tmp_path / "video.bilingual.ass").read_text(encoding="utf-8")
    dialogue = [line for line in merged.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue) == 8
    assert sum("BilingualEn" in line for line in dialogue) == 2
    assert sum("BilingualZh" in line for line in dialogue) == 2
    assert sum(",Box," in line for line in dialogue) == 4
    assert "hello there" in merged and "你好" in merged


def test_bilingual_ass_merge_dedups_overlaps_by_group(tmp_path: Path):
    seg1 = tmp_path / ".seg1"
    seg2 = tmp_path / ".seg2"
    seg1.mkdir()
    seg2.mkdir()
    # seg1 tail cue (99.0–101.0 global) overlaps seg2's first kept cue.
    (seg1 / "video.bilingual.ass").write_text(
        _HEADER + _cue_group("0:01:39.00", "0:01:41.00", "seg1 tail", "甲段结尾"),
        encoding="utf-8",
    )
    # seg2 starts at 90s global (10s overlap before the split point at 100s):
    # local 10.2–12.0 → global 100.2–102.0.
    (seg2 / "video.bilingual.ass").write_text(
        _HEADER + _cue_group("0:00:10.20", "0:00:12.00", "seg2 head", "乙段开头"),
        encoding="utf-8",
    )

    _merge_bilingual_ass(
        tmp_path,
        [seg1, seg2],
        [0.0, 90.0],
        [110.0, 110.0],
        [0.0, 100.0, 200.0],
        "slug",
    )

    merged = (tmp_path / "video.bilingual.ass").read_text(encoding="utf-8")
    dialogue = [line for line in merged.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue) == 4  # the later group survives whole, the earlier one is dropped
    assert "seg2 head" in merged and "乙段开头" in merged
    assert "seg1 tail" not in merged
