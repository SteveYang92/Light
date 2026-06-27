"""Tests for split-aware translation payload and punctuation."""

from __future__ import annotations

import json

import pytest
from light_models import Segment, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.translate import (
    _adjust_chunk_end,
    _build_payload,
    _chunk_pending_segments,
    _is_last_split_part,
    _normalize_punctuation,
    _parse_response,
    _parse_split_part,
    _split_group_part_counts,
)

# ── Helpers ─────────────────────────────────────────────────────


def _seg(unit_id: str, text: str, *, start: float = 0.0, end: float = 5.0) -> Segment:
    words = [
        Word(text=word, start=start + i * 0.5, end=start + i * 0.5 + 0.4, confidence=0.9)
        for i, word in enumerate(text.split())
    ]
    return Segment(
        unit_id=unit_id,
        start=words[0].start if words else start,
        end=words[-1].end if words else end,
        speaker="",
        source_text=text,
        words=words,
    )


def _config() -> SubtitleConfig:
    return SubtitleConfig(input_path="dummy.mp4", target_lang="zh")


# ── Split part parsing ───────────────────────────────────────────


class TestSplitPartParsing:
    def test_parse_split_part(self):
        assert _parse_split_part("mu0059_u0060_0") == ("mu0059_u0060", 0)
        assert _parse_split_part("mu0059_u0060_1") == ("mu0059_u0060", 1)
        assert _parse_split_part("mu0187_u0190_0_1") == ("mu0187_u0190_0", 1)
        assert _parse_split_part("mu0187_u0190_0_2") == ("mu0187_u0190_0", 2)
        assert _parse_split_part("mu0059_u0060") is None

    def test_split_group_part_counts(self):
        segments = [
            _seg("mu0059_u0060_0", "part zero"),
            _seg("mu0059_u0060_1", "part one"),
            _seg("mu0001_u0002", "whole unit"),
        ]
        assert _split_group_part_counts(segments) == {"mu0059_u0060": 2}

    def test_is_last_split_part(self):
        counts = {"mu0059_u0060": 2}
        assert _is_last_split_part("mu0059_u0060_0", counts) is False
        assert _is_last_split_part("mu0059_u0060_1", counts) is True
        assert _is_last_split_part("mu0001_u0002", counts) is None


# ── Punctuation normalization ────────────────────────────────────


class TestSplitAwarePunctuation:
    def test_non_final_split_part_keeps_open_ending(self):
        text = _normalize_punctuation(
            "你如何防止这些自由价值观",
            "zh",
            is_last_split_part=False,
            source_ends_sentence=False,
        )
        assert not text.endswith("。")

    def test_final_split_part_gets_period(self):
        text = _normalize_punctuation(
            "被金钱腐蚀",
            "zh",
            is_last_split_part=True,
            source_ends_sentence=True,
        )
        assert text.endswith("。")

    def test_non_split_unit_still_gets_period(self):
        text = _normalize_punctuation("这是完整一句", "zh")
        assert text.endswith("。")


# ── Payload metadata ─────────────────────────────────────────────


class TestSplitPayload:
    def test_build_payload_includes_split_group(self):
        segments = [
            _seg("mu0059_u0060_0", "values for freedom being"),
            _seg("mu0059_u0060_1", "corrupted by money"),
        ]
        payload = _build_payload(segments, segments, 0, _config())
        units = payload["units"]
        translate_units = [u for u in units if "translate" not in u or u.get("translate") is not False]
        assert len(translate_units) == 2
        assert translate_units[0]["batch_index"] == 0
        assert translate_units[1]["batch_index"] == 1
        assert translate_units[0]["split_group"] == "mu0059_u0060"
        assert translate_units[0]["part_index"] == 0
        assert translate_units[0]["part_count"] == 2
        assert translate_units[0]["is_continuation"] is False
        assert translate_units[1]["is_continuation"] is True


class TestParseResponse:
    def test_binds_by_batch_index(self):
        segments = [
            _seg("mu0194_u0194", "It's haunted"),
            _seg("mu0195_u0195", "Peruvian government"),
        ]
        response = json.dumps(
            [
                {"batch_index": 1, "text": "秘鲁政府"},
                {"batch_index": 0, "text": "这里闹鬼"},
            ]
        )
        cues, parsed = _parse_response(response, segments, _config(), segments)
        assert len(cues) == 2
        assert cues[0].unit_id == "mu0194_u0194"
        assert cues[0].text.startswith("这里闹鬼")
        assert cues[1].unit_id == "mu0195_u0195"
        assert "秘鲁" in parsed[1]

    def test_fallback_unit_id_when_no_batch_index(self):
        segments = [_seg("mu0001_u0002", "hello")]
        response = json.dumps([{"unit_id": "mu0001_u0002", "text": "你好"}])
        cues, _ = _parse_response(response, segments, _config(), segments)
        assert len(cues) == 1
        assert cues[0].text == "你好。"

    def test_incomplete_batch_raises(self):
        segments = [
            _seg("mu0001_u0002", "one"),
            _seg("mu0003_u0004", "two"),
        ]
        response = json.dumps([{"batch_index": 0, "text": "一"}])
        with pytest.raises(ValueError, match="Batch incomplete"):
            _parse_response(response, segments, _config(), segments)

    def test_duplicate_batch_index_raises(self):
        segments = [_seg("mu0001_u0002", "one")]
        response = json.dumps(
            [
                {"batch_index": 0, "text": "一"},
                {"batch_index": 0, "text": "又一"},
            ]
        )
        with pytest.raises(ValueError, match="Duplicate batch_index"):
            _parse_response(response, segments, _config(), segments)

    def test_parse_returns_texts_not_merge_hints(self):
        segments = [
            _seg("mu0187_u0190_0", "AI based"),
            _seg("mu0187_u0190_1", "algorithm push"),
        ]
        response = json.dumps(
            [
                {"batch_index": 0, "text": "基于AI的"},
                {"batch_index": 1, "text": "算法推送"},
            ]
        )
        cues, parsed = _parse_response(response, segments, _config(), segments)
        assert len(cues) == 2
        assert parsed[0] == "基于AI的"
        assert parsed[1].startswith("算法推送")


class TestChunkPendingSegments:
    def test_keeps_split_group_in_one_batch(self):
        pending = [_seg(f"mu0001_u0002_{i}", f"word{i} text") for i in range(98)] + [
            _seg("mu0059_u0060_0", "first part of split"),
            _seg("mu0059_u0060_1", "second part of split"),
        ]
        chunks = _chunk_pending_segments(pending, 100)
        assert len(chunks) == 1
        assert chunks[0][-2].unit_id == "mu0059_u0060_0"
        assert chunks[0][-1].unit_id == "mu0059_u0060_1"

    def test_adjust_chunk_end_extends_for_split_sibling(self):
        pending = [
            _seg("mu0000_u0001", "one"),
            _seg("mu0059_u0060_0", "left"),
            _seg("mu0059_u0060_1", "right"),
        ]
        end = _adjust_chunk_end(pending, 0, 2, 2)
        assert end == 3

    def test_never_splits_group_across_batches(self):
        pending = [_seg(f"u{i:04d}", f"text {i}") for i in range(99)] + [
            _seg("mu0059_u0060_0", "first part"),
            _seg("mu0059_u0060_1", "second part"),
        ]
        chunks = _chunk_pending_segments(pending, 100)
        assert len(chunks) == 2
        assert len(chunks[0]) == 99
        assert [s.unit_id for s in chunks[1]] == ["mu0059_u0060_0", "mu0059_u0060_1"]

    def test_oversized_group_single_batch(self):
        pending = [_seg(f"mu0001_u0002_{i}", f"part {i}") for i in range(105)]
        chunks = _chunk_pending_segments(pending, 100)
        assert len(chunks) == 1
        assert len(chunks[0]) == 105


ROSSOLIE_MU0193_0199 = [
    {
        "unit_id": "mu0193_u0193",
        "text": "Um, you know, it's like, people say there's, there's, there's Bigfoot or don't go there.",
    },
    {
        "unit_id": "mu0194_u0194",
        "text": "It's haunted or something, you know, it's like, there was like a tall tale almost.",
    },
    {
        "unit_id": "mu0195_u0196_0",
        "text": "And even the Peruvian government at the time that I went to Peru first, which was 2006,",
    },
    {
        "unit_id": "mu0195_u0196_1",
        "text": "their official position was that the tribes are a myth.",
    },
    {
        "unit_id": "mu0197_u0197",
        "text": "There's no such thing as the tribes.",
    },
    {
        "unit_id": "mu0198_u0198",
        "text": "That was the official position.",
    },
    {
        "unit_id": "mu0199_u0199",
        "text": "And you would hear these stories of people that got shot.",
    },
]


class TestRosolieRegression:
    """Regression guards for Paul Rosolie seg1 mu0193–mu0199 mis-mapping case."""

    def _segments(self) -> list[Segment]:
        return [
            _seg(item["unit_id"], item["text"], start=759.0 + i, end=760.0 + i)
            for i, item in enumerate(ROSSOLIE_MU0193_0199)
        ]

    def test_split_siblings_stay_in_one_chunk(self):
        pending = self._segments()
        chunks = _chunk_pending_segments(pending, 100)
        assert len(chunks) == 1
        split_ids = [s.unit_id for s in chunks[0] if s.unit_id.startswith("mu0195_u0196")]
        assert split_ids == ["mu0195_u0196_0", "mu0195_u0196_1"]

    def test_batch_index_maps_haunted_to_mu0194_not_peruvian(self):
        """LLM may return wrong unit_id; batch_index must bind semantics correctly."""
        segments = self._segments()
        response = json.dumps(
            [
                {"batch_index": 0, "unit_id": "mu0199_u0199", "text": "有人说那里有大脚怪，别去"},
                {"batch_index": 1, "unit_id": "mu0193_u0193", "text": "闹鬼什么的，像个荒诞故事"},
                {"batch_index": 2, "text": "甚至秘鲁政府"},
                {"batch_index": 3, "text": "官方立场是部落是神话"},
                {"batch_index": 4, "text": "根本不存在什么部落"},
                {"batch_index": 5, "text": "那是官方立场"},
                {"batch_index": 6, "text": "你会听到有人中枪的故事"},
            ]
        )
        cues, _ = _parse_response(response, segments, _config(), segments)
        by_id = {c.unit_id: c.text for c in cues}
        assert "闹鬼" in by_id["mu0194_u0194"]
        assert "秘鲁" in by_id["mu0195_u0196_0"]
        assert "大脚怪" in by_id["mu0193_u0193"]
