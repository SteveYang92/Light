"""Tests for cli reporter selection (_is_tty / _make_reporter)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from light_cli.cli import _finished_payload, _is_tty, _make_reporter, _resolve_media_duration
from light_cli.reporting import PlainReporter
from light_cli.reporting.rich_ui import RichReporter


def _tty(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: value)


class TestIsTty:
    def test_true_interactive_terminal(self, monkeypatch):
        _tty(monkeypatch, True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert _is_tty() is True

    def test_not_a_tty(self, monkeypatch):
        _tty(monkeypatch, False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        assert _is_tty() is False

    def test_no_color_disables(self, monkeypatch):
        _tty(monkeypatch, True)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TERM", "xterm")
        assert _is_tty() is False

    def test_dumb_term_disables(self, monkeypatch):
        _tty(monkeypatch, True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert _is_tty() is False


class TestMakeReporter:
    def test_verbose_forces_plain(self, monkeypatch):
        _tty(monkeypatch, True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        assert isinstance(_make_reporter(True), PlainReporter)

    def test_tty_picks_rich(self, monkeypatch):
        _tty(monkeypatch, True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        assert isinstance(_make_reporter(False), RichReporter)

    def test_non_tty_picks_plain(self, monkeypatch):
        _tty(monkeypatch, False)
        assert isinstance(_make_reporter(False), PlainReporter)

    def test_no_color_tty_picks_plain(self, monkeypatch):
        _tty(monkeypatch, True)
        monkeypatch.setenv("NO_COLOR", "1")
        assert isinstance(_make_reporter(False), PlainReporter)


class TestFinishedPayloadDuration:
    def test_resolves_legacy_slug_video_when_stale_path_missing(self, tmp_path: Path):
        slug = "Demo_Talk"
        work = tmp_path / slug
        work.mkdir()
        video = work / f"{slug}.webm"
        video.write_bytes(b"fake")
        stale = work / "video.webm"  # missing — old rename left only {slug}.webm

        with patch("light_cli.utils.ffmpeg.probe_duration", return_value=123.4) as probe:
            duration = _resolve_media_duration(work, stale, slug)
            assert duration == 123.4
            probe.assert_called_once_with(str(video))

        with patch("light_cli.utils.ffmpeg.probe_duration", return_value=123.4):
            payload = _finished_payload(work, slug, stale)
        assert payload["duration"] == 123.4

    def test_transcript_fallback_when_no_video(self, tmp_path: Path):
        work = tmp_path / "demo"
        work.mkdir()
        (work / "transcript.json").write_text(
            json.dumps({"words": [{"text": "hi", "start": 0.0, "end": 1.5}], "segments": []}),
            encoding="utf-8",
        )
        duration = _resolve_media_duration(work, work / "missing.mp4", "demo")
        assert duration == 1.5
