"""Tests for merge_outputs.py — SRT/VTT time-shift merge and overlap trimming."""

from __future__ import annotations

import tempfile
from pathlib import Path

from light_subtitle.merge_outputs import (
    _parse_srt,
    _parse_vtt,
    _seconds_to_srt,
    _srt_to_seconds,
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
