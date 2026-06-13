"""Tests for video_split.py — should_split and split point computation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from light_subtitle.video_split import SilenceInterval, _best_silence, should_split


class TestShouldSplit:
    def test_short_video_returns_false(self) -> None:
        with patch("light_subtitle.video_split.probe_duration", return_value=120.0):
            assert should_split(Path("short.mp4"), threshold=2700) is False

    def test_long_video_returns_true(self) -> None:
        with patch("light_subtitle.video_split.probe_duration", return_value=3600.0):
            assert should_split(Path("long.mp4"), threshold=2700) is True

    def test_exactly_threshold(self) -> None:
        with patch("light_subtitle.video_split.probe_duration", return_value=2700.0):
            assert should_split(Path("exact.mp4"), threshold=2700) is False

    def test_custom_threshold(self) -> None:
        with patch("light_subtitle.video_split.probe_duration", return_value=600.0):
            assert should_split(Path("video.mp4"), threshold=300) is True


class TestBestSilence:
    def _make_interval(self, start: float, end: float, duration: float) -> SilenceInterval:
        return SilenceInterval(start=start, end=end, duration=duration)

    def test_picks_long_silence_over_closer_short_one(self) -> None:
        # Target at 100s.  Long silence (3s) at 95s, short (0.5s) at 101s.
        intervals = [
            self._make_interval(94.0, 97.0, 3.0),
            self._make_interval(100.5, 101.0, 0.5),
        ]
        result = _best_silence(100.0, intervals, window=60)
        # Long silence should be preferred despite being farther
        assert result == 97.0

    def test_falls_back_to_narrow_window(self) -> None:
        # No long silences in wide window, but a short one nearby.
        intervals = [
            self._make_interval(99.0, 99.8, 0.8),
        ]
        result = _best_silence(100.0, intervals, window=60)
        assert result == 99.8

    def test_returns_none_when_no_silence_found(self) -> None:
        intervals: list[SilenceInterval] = []
        result = _best_silence(100.0, intervals, window=60)
        assert result is None

    def test_ignores_silence_far_outside_window(self) -> None:
        intervals = [
            self._make_interval(80.0, 81.0, 1.0),  # outside ±60 window for target=150
        ]
        result = _best_silence(150.0, intervals, window=60)
        assert result is None
