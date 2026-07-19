"""Tests for cli reporter selection (_is_tty / _make_reporter)."""

from __future__ import annotations

import sys

from light_subtitle.cli import _is_tty, _make_reporter
from light_subtitle.reporting import PlainReporter
from light_subtitle.reporting.rich_ui import RichReporter


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
