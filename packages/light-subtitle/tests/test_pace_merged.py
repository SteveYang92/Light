"""Tests for pace max_duration exemption on display-merged cues."""

from __future__ import annotations

from light_models import SubtitleCue
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.subtitle.pace import _enforce_cps_ceiling, _fix_cue_duration


def _config(**kwargs) -> SubtitleConfig:
    return SubtitleConfig(input_path="dummy.mp4", max_duration=7.0, **kwargs)


class TestMergedDurationExempt:
    def test_fix_cue_duration_skips_cap_for_merged(self):
        cue = SubtitleCue(
            cue_id="zh_0",
            unit_id="u0",
            start=0.0,
            end=10.0,
            text="长合并字幕",
            lang="zh",
            merged_from=["u1"],
        )
        result = _fix_cue_duration(cue, _config())
        assert result[0].end == 10.0

    def test_fix_cue_duration_caps_non_merged(self):
        cue = SubtitleCue(
            cue_id="zh_0",
            unit_id="u0",
            start=0.0,
            end=10.0,
            text="普通字幕",
            lang="zh",
        )
        result = _fix_cue_duration(cue, _config())
        assert result[0].end == 7.0

    def test_cps_ceiling_allows_borrow_past_max_for_merged(self):
        long_text = "字" * 80
        cue = SubtitleCue(
            cue_id="zh_0",
            unit_id="u0",
            start=0.0,
            end=7.0,
            text=long_text,
            lang="zh",
            merged_from=["u1"],
        )
        next_cue = SubtitleCue(
            cue_id="zh_1",
            unit_id="u2",
            start=20.0,
            end=21.0,
            text="next",
            lang="zh",
        )
        result = _enforce_cps_ceiling([cue, next_cue], _config(cps_limit=8))
        assert result[0].end > 7.0

        plain = SubtitleCue(
            cue_id="zh_0",
            unit_id="u0",
            start=0.0,
            end=7.0,
            text=long_text,
            lang="zh",
        )
        capped = _enforce_cps_ceiling([plain, next_cue], _config(cps_limit=8))
        assert capped[0].end == 7.0
