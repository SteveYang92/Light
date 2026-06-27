"""Tests for merge_outputs.py — SRT/VTT time-shift merge, overlap trimming, and annotation dedup."""

from __future__ import annotations

import tempfile
from pathlib import Path

from light_subtitle.merge_outputs import (
    _copy_single_segment,
    _dedup_annotation_terms,
    _extract_annotation_term,
    _parse_srt,
    _parse_vtt,
    _seconds_to_srt,
    _srt_to_seconds,
    _strip_annotation_marker,
    _write_srt,
    _write_vtt,
)


class TestTimeConversion:
    def test_srt_roundtrip(self) -> None:
        ts = "01:23:45,678"
        assert _seconds_to_srt(_srt_to_seconds(ts)) == ts

    def test_srt_zero(self) -> None:
        assert _seconds_to_srt(0.0) == "00:00:00,000"

    def test_srt_one_hour(self) -> None:
        assert _seconds_to_srt(3600.0) == "01:00:00,000"

    def test_srt_milliseconds(self) -> None:
        assert _seconds_to_srt(1.5) == "00:00:01,500"


class TestParseSrt:
    def test_single_cue(self) -> None:
        content = "1\n00:00:01,000 --> 00:00:05,000\nHello world\n\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        cues = _parse_srt(path)
        path.unlink()
        assert len(cues) == 1
        assert cues[0] == (1.0, 5.0, "Hello world")

    def test_multiple_cues(self) -> None:
        content = "1\n00:00:01,000 --> 00:00:03,000\nFirst\n\n2\n00:00:04,000 --> 00:00:06,000\nSecond\n\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        cues = _parse_srt(path)
        path.unlink()
        assert len(cues) == 2
        assert cues[1] == (4.0, 6.0, "Second")

    def test_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write("")
            path = Path(f.name)
        cues = _parse_srt(path)
        path.unlink()
        assert cues == []

    def test_nonexistent_file(self) -> None:
        assert _parse_srt(Path("/nonexistent.srt")) == []

    def test_ignores_index_line_with_timestamp_on_line_1(self) -> None:
        # When the index line is missing, timestamp is on line 0
        content = "00:00:01,000 --> 00:00:05,000\nNo index line\n\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        cues = _parse_srt(path)
        path.unlink()
        assert len(cues) == 1
        assert cues[0][2] == "No index line"


class TestWriteSrt:
    def test_write_and_parse_roundtrip(self) -> None:
        cues = [(1.0, 5.0, "Hello"), (6.0, 10.0, "World")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            path = Path(f.name)
        _write_srt(cues, path)
        parsed = _parse_srt(path)
        path.unlink()
        assert len(parsed) == 2
        assert parsed[0] == (1.0, 5.0, "Hello")
        assert parsed[1] == (6.0, 10.0, "World")


class TestParseVtt:
    def test_single_cue(self) -> None:
        content = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:05.000 align:start line:0%\nHello world\n\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        cues = _parse_vtt(path)
        path.unlink()
        assert len(cues) == 1
        start, end, text, settings = cues[0]
        assert start == 1.0
        assert end == 5.0
        assert text == "Hello world"
        assert settings == "align:start line:0%"

    def test_nonexistent_file(self) -> None:
        assert _parse_vtt(Path("/nonexistent.vtt")) == []


class TestWriteVtt:
    def test_write_and_parse_roundtrip(self) -> None:
        cues = [(1.0, 5.0, "Hello", "align:start line:0%")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as f:
            path = Path(f.name)
        _write_vtt(cues, path)
        parsed = _parse_vtt(path)
        path.unlink()
        assert len(parsed) == 1
        assert parsed[0][:3] == (1.0, 5.0, "Hello")


# ── Annotation term dedup ───────────────────────────────────────────────────


class TestStripAnnotationMarker:
    def test_removes_marker(self) -> None:
        assert _strip_annotation_marker("※ RL训练") == "RL训练"

    def test_removes_multiple_markers(self) -> None:
        assert _strip_annotation_marker("※※ RL训练") == "RL训练"

    def test_no_marker_unchanged(self) -> None:
        assert _strip_annotation_marker("RL训练") == "RL训练"

    def test_marker_with_no_space(self) -> None:
        assert _strip_annotation_marker("※RL训练") == "RL训练"

    def test_only_marker_becomes_empty(self) -> None:
        assert _strip_annotation_marker("※") == ""

    def test_whitespace_marker(self) -> None:
        assert _strip_annotation_marker("  ※   RL训练") == "RL训练"


class TestExtractAnnotationTerm:
    def test_chinese_colon(self) -> None:
        assert _extract_annotation_term("※ RL训练：强化学习的方法") == "rl训练"

    def test_ascii_colon(self) -> None:
        assert _extract_annotation_term("※ RL训练: 强化学习的方法") == "rl训练"

    def test_no_separator(self) -> None:
        assert _extract_annotation_term("※ 简单术语") == "简单术语"

    def test_no_marker(self) -> None:
        assert _extract_annotation_term("RL训练：强化学习") == "rl训练"

    def test_case_insensitive(self) -> None:
        assert _extract_annotation_term("※ RL训练：强化学习") == _extract_annotation_term("※ rl训练：强化学习")

    def test_multiple_colons_in_explanation(self) -> None:
        assert _extract_annotation_term("※ RL训练：优势：高效学习") == "rl训练"


class TestDedupAnnotationTerms:
    def test_same_term_keeps_first(self) -> None:
        cues = [
            (1.0, 5.0, "※ RL训练：强化学习的方法", "align:start line:0%"),
            (10.0, 15.0, "※ RL训练：另一种解释", "align:start line:0%"),
        ]
        result = _dedup_annotation_terms(cues)
        assert len(result) == 1
        assert result[0] == cues[0]

    def test_different_terms_both_kept(self) -> None:
        cues = [
            (1.0, 5.0, "※ RL训练：强化学习", "align:start line:0%"),
            (10.0, 15.0, "※ 梯度下降：优化算法", "align:start line:0%"),
        ]
        result = _dedup_annotation_terms(cues)
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert _dedup_annotation_terms([]) == []

    def test_no_colon_uses_full_body_as_key(self) -> None:
        cues = [
            (1.0, 5.0, "※ 鲁迅", "align:start line:0%"),
            (10.0, 15.0, "※ 鲁迅", "align:start line:0%"),
        ]
        result = _dedup_annotation_terms(cues)
        assert len(result) == 1

    def test_same_term_without_marker(self) -> None:
        cues = [
            (1.0, 5.0, "RL训练：强化学习", ""),
            (10.0, 15.0, "※ RL训练：另一种解释", ""),
        ]
        result = _dedup_annotation_terms(cues)
        assert len(result) == 1
        assert result[0] == cues[0]

    def test_three_same_keeps_first(self) -> None:
        cues = [
            (1.0, 5.0, "※ RL训练：解释A", ""),
            (10.0, 15.0, "※ RL训练：解释B", ""),
            (20.0, 25.0, "※ RL训练：解释C", ""),
        ]
        result = _dedup_annotation_terms(cues)
        assert len(result) == 1
        assert result[0] == cues[0]

    def test_term_with_whitespace_variation(self) -> None:
        cues = [
            (1.0, 5.0, "※  RL训练  ：强化学习", ""),
            (10.0, 15.0, "※ RL训练：另一种解释", ""),
        ]
        result = _dedup_annotation_terms(cues)
        assert len(result) == 1


def test_copy_single_segment_overwrites_existing_slug_files(tmp_path: Path) -> None:
    """Single-segment merge must refresh root slug files after segment re-export."""
    slug = "demo"
    seg_dir = tmp_path / ".seg1"
    seg_dir.mkdir()
    (seg_dir / "bilingual.ass").write_text("new", encoding="utf-8")
    (tmp_path / f"{slug}.bilingual.ass").write_text("old", encoding="utf-8")

    _copy_single_segment(tmp_path, seg_dir, slug)

    assert (tmp_path / f"{slug}.bilingual.ass").read_text(encoding="utf-8") == "new"
