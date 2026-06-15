"""Tests for split-aware translation payload and punctuation."""

from __future__ import annotations

from light_models import Segment, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.translate import (
    _adjust_chunk_end,
    _build_payload,
    _chunk_pending_segments,
    _is_last_split_part,
    _normalize_punctuation,
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
        assert translate_units[0]["split_group"] == "mu0059_u0060"
        assert translate_units[0]["part_index"] == 0
        assert translate_units[0]["part_count"] == 2
        assert translate_units[0]["is_continuation"] is False
        assert translate_units[1]["is_continuation"] is True


class TestChunkPendingSegments:
    def test_keeps_split_group_in_one_batch(self):
        pending = [
            _seg(f"mu0001_u0002_{i}", f"word{i} text") for i in range(98)
        ] + [
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
