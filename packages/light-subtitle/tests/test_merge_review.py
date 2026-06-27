"""Tests for LLM merge-hint review (parse + hint assembly)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from light_models import Segment, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.merge_review import (
    _apply_gap_filters,
    _build_review_units,
    _gap_to_next_ms,
    _hints_from_flags,
    _parse_merge_review_response,
    log_merge_hints,
    review_merge_hints,
)


def _seg(unit_id: str, text: str, *, start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(
        unit_id=unit_id,
        start=start,
        end=end,
        speaker="",
        source_text=text,
        words=[Word(text="x", start=start, end=(start + end) / 2, confidence=0.9)],
    )


def _config() -> SubtitleConfig:
    return SubtitleConfig(input_path="dummy.mp4", target_lang="zh", llm_api_key="test")


class TestParseMergeReviewResponse:
    def test_parses_flags_by_index(self):
        response = json.dumps(
            [
                {"batch_index": 1, "merge_with_next": True},
                {"batch_index": 0, "merge_with_next": False},
            ]
        )
        flags = _parse_merge_review_response(response, 2)
        assert flags == {0: False, 1: True}

    def test_incomplete_defaults_missing_to_false(self):
        response = json.dumps([{"batch_index": 0, "merge_with_next": False}])
        flags = _parse_merge_review_response(response, 2)
        assert flags == {0: False, 1: False}


class TestHintsFromFlags:
    def test_true_positive_modifier_pair(self):
        segments = [
            _seg("u0", "a"),
            _seg("u1", "b"),
            _seg("u2", "c"),
        ]
        texts = {0: "被基于人工智能的", 1: "算法推送牵着走的人。", 2: "结果类似。"}
        flags = {0: True, 1: False, 2: False}
        hints = _hints_from_flags(segments, texts, flags)
        assert len(hints) == 1
        assert hints[0][0].unit_id == "u0"
        assert hints[0][1].unit_id == "u1"
        assert "算法推送" in hints[0][3]

    def test_false_beat_pair_no_hint(self):
        segments = [_seg("u0", "a"), _seg("u1", "b")]
        texts = {0: "她停了一下，", 1: "然后继续往下讲。"}
        flags = {0: False, 1: False}
        assert _hints_from_flags(segments, texts, flags) == []

    def test_sentence_end_not_flagged(self):
        segments = [_seg("u0", "a"), _seg("u1", "b")]
        texts = {0: "算法推送牵着走的人。", 1: "结果你和别人消费着同样的信息。"}
        flags = {0: False, 1: False}
        assert _hints_from_flags(segments, texts, flags) == []


class TestGapToNextMs:
    def test_gap_from_segment_timing(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.92, end=3.0),
        ]
        assert _gap_to_next_ms(segments, 0) == 920
        assert _gap_to_next_ms(segments, 1) is None

    def test_build_review_units_includes_gap(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.08, end=2.0),
        ]
        units = _build_review_units(segments, {0: "我所知道的。", 1: "所以这不仅仅是。"})
        assert units[0]["gap_to_next_ms"] == 80
        assert "gap_to_next_ms" not in units[1]


class TestGapFilters:
    def test_forces_false_on_closure_and_long_gap(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.92, end=3.0),
            _seg("u2", "c", start=3.1, end=4.0),
        ]
        texts = {0: "我所知道的。", 1: "被基于人工智能的", 2: "下一句。"}
        flags = {0: True, 1: True, 2: False}
        filtered = _apply_gap_filters(flags, segments, texts)
        assert filtered[0] is False
        assert filtered[1] is True

    def test_forces_false_on_closure_regardless_of_gap(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.08, end=2.0),
        ]
        texts = {0: "很多人用酒精。", 1: "来维持社交生活。"}
        flags = {0: True, 1: False}
        filtered = _apply_gap_filters(flags, segments, texts)
        assert filtered[0] is False

    def test_keeps_true_when_gap_below_threshold_and_no_closure(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.08, end=2.0),
        ]
        texts = {0: "很多人用酒精", 1: "来维持社交生活。"}
        flags = {0: True, 1: False}
        filtered = _apply_gap_filters(flags, segments, texts)
        assert filtered[0] is True

    def test_forces_false_on_long_gap_without_dangling(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=2.902, end=4.0),
        ]
        texts = {0: "我的消化系统", 1: "不太适应这类食物。"}
        flags = {0: True, 1: False}
        filtered = _apply_gap_filters(flags, segments, texts)
        assert filtered[0] is False

    def test_keeps_true_on_long_gap_with_de(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.92, end=3.0),
        ]
        texts = {0: "被基于人工智能的", 1: "算法推送摆布的人。"}
        flags = {0: True, 1: False}
        filtered = _apply_gap_filters(flags, segments, texts)
        assert filtered[0] is True

    def test_llm_override_closure_long_gap(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.92, end=3.0),
        ]
        texts = {0: "我所知道的。", 1: "所以这不仅仅是。"}
        client = MagicMock()
        client.chat.return_value = (
            json.dumps([{"batch_index": 0, "merge_with_next": True}, {"batch_index": 1, "merge_with_next": False}]),
            {},
        )
        hints, _usage = review_merge_hints(client, segments, texts, _config())
        assert hints == []


class TestReviewMergeHints:
    def test_calls_llm_and_builds_hints(self):
        segments = [_seg("u0", "a"), _seg("u1", "b")]
        texts = {0: "这家工厂生产的", 1: "外壳公差可以控制在两丝以内。"}
        client = MagicMock()
        client.chat.return_value = (
            json.dumps([{"batch_index": 0, "merge_with_next": True}, {"batch_index": 1, "merge_with_next": False}]),
            {},
        )
        hints, _usage = review_merge_hints(client, segments, texts, _config())
        assert len(hints) == 1
        assert hints[0][2] == "这家工厂生产的"
        assert "外壳" in hints[0][3]

    def test_payload_includes_gap_to_next_ms(self):
        segments = [
            _seg("u0", "a", start=0.0, end=1.0),
            _seg("u1", "b", start=1.92, end=3.0),
        ]
        texts = {0: "我所知道的。", 1: "所以这不仅仅是。"}
        client = MagicMock()
        client.chat.return_value = (
            json.dumps([{"batch_index": 0, "merge_with_next": False}, {"batch_index": 1, "merge_with_next": False}]),
            {},
        )
        review_merge_hints(client, segments, texts, _config())
        payload = json.loads(client.chat.call_args[0][0][1]["content"])
        assert payload["gap_closure_false_ms"] == 800
        assert payload["units"][0]["gap_to_next_ms"] == 920
        assert "gap_to_next_ms" not in payload["units"][1]

    def test_single_segment_skips_review(self):
        client = MagicMock()
        hints, _usage = review_merge_hints(client, [_seg("u0", "a")], {0: "单独一句。"}, _config())
        assert hints == []
        client.chat.assert_not_called()


class TestLogMergeHints:
    def test_logs_chinese_next_text(self, capsys):
        segments = [
            _seg("u0", "a", start=0.0, end=6.695),
            _seg("u1", "b", start=7.035, end=10.257),
        ]
        hints = [(segments[0], segments[1], "被基于人工智能的", "算法推送牵着走的人。")]
        log_merge_hints(hints)
        out = capsys.readouterr().out
        assert "算法推送" in out
        assert "Layout merge hint" in out
        assert "gap=340ms" in out
        assert "curr_dur=6695ms" in out
        assert "next_dur=3222ms" in out
