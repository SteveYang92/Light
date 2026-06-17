"""Tests for translate-time merge hint application."""

from __future__ import annotations

from light_models import Segment, SubtitleCue, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.merge_apply import apply_display_merges


def _config(*, max_duration: float = 7.0) -> SubtitleConfig:
    return SubtitleConfig(input_path="dummy.mp4", target_lang="zh", max_duration=max_duration, llm_api_key="test")


def _seg(unit_id: str, *, start: float, end: float) -> Segment:
    return Segment(
        unit_id=unit_id,
        start=start,
        end=end,
        speaker="",
        source_text="x",
        words=[Word(text="x", start=start, end=end, confidence=0.9)],
    )


def _cue(unit_id: str, text: str, *, start: float, end: float) -> SubtitleCue:
    return SubtitleCue(
        cue_id="zh_0000",
        unit_id=unit_id,
        start=start,
        end=end,
        text=text,
        lang="zh",
    )


class TestApplyDisplayMerges:
    def test_merges_adjacent_pair(self):
        cues = [
            _cue("u0", "被基于人工智能的", start=0.0, end=3.0),
            _cue("u1", "算法推送摆布的人。", start=3.08, end=6.0),
        ]
        hints = [(_seg("u0", start=0.0, end=3.0), _seg("u1", start=3.08, end=6.0), cues[0].text, cues[1].text)]
        result = apply_display_merges(cues, hints, _config())
        assert len(result) == 1
        assert result[0].text == "被基于人工智能的算法推送摆布的人。"
        assert result[0].start == 0.0
        assert result[0].end == 6.0

    def test_skips_when_duration_exceeds_twice_max(self):
        cues = [
            _cue("u0", "这样你就不会成为", start=0.0, end=7.0),
            _cue("u1", "一个被基于人工智能的", start=7.5, end=14.5),
        ]
        hints = [(_seg("u0", start=0.0, end=7.0), _seg("u1", start=7.5, end=14.5), cues[0].text, cues[1].text)]
        result = apply_display_merges(cues, hints, _config(max_duration=7.0))
        assert len(result) == 2
        assert result[0].text == cues[0].text
        assert result[1].text == cues[1].text

    def test_merges_chain(self):
        cues = [
            _cue("u0", "这样你就不会成为", start=0.0, end=2.0),
            _cue("u1", "一个被基于人工智能的", start=2.08, end=4.0),
            _cue("u2", "算法推送摆布的人。", start=4.34, end=6.0),
        ]
        segs = [_seg(uid, start=c.start, end=c.end) for uid, c in zip(["u0", "u1", "u2"], cues, strict=True)]
        hints = [
            (segs[0], segs[1], cues[0].text, cues[1].text),
            (segs[1], segs[2], cues[1].text, cues[2].text),
        ]
        result = apply_display_merges(cues, hints, _config())
        assert len(result) == 1
        assert result[0].merged_from == ["u1", "u2"]
        assert "算法推送" in result[0].text


class TestCoveredUnitIds:
    def test_includes_head_and_absorbed(self):
        cues = [
            SubtitleCue(
                cue_id="zh_0000",
                unit_id="u0",
                start=0.0,
                end=6.0,
                text="merged",
                lang="zh",
                merged_from=["u1", "u2"],
            ),
            SubtitleCue(cue_id="zh_0001", unit_id="u3", start=7.0, end=9.0, text="solo", lang="zh"),
        ]
        from light_subtitle.pipeline.translate.merge_apply import covered_unit_ids

        assert covered_unit_ids(cues) == {"u0", "u1", "u2", "u3"}
