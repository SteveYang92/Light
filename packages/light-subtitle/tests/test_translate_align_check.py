"""Tests for batch translation alignment check."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from light_models import Segment, Word
from light_subtitle.config import TranslateConfig
from light_subtitle.translate import align_check
from light_subtitle.translate.translate import _translate_batch


def _config(**kwargs) -> TranslateConfig:
    defaults = {"target_lang": "zh"}
    defaults.update(kwargs)
    return TranslateConfig(**defaults)


def _seg(unit_id: str, text: str = "hello") -> Segment:
    return Segment(
        unit_id=unit_id,
        start=0.0,
        end=1.0,
        speaker="",
        source_text=text,
        words=[Word(text=text, start=0.0, end=1.0, confidence=0.9)],
    )


class TestAlignmentSampleIndices:
    def test_single_unit(self):
        assert align_check._alignment_sample_indices(1) == [0]

    def test_two_units(self):
        assert align_check._alignment_sample_indices(2) == [0, 1]

    def test_five_units(self):
        assert align_check._alignment_sample_indices(5) == [0, 2, 4]

    def test_empty(self):
        assert align_check._alignment_sample_indices(0) == []


class TestBuildCheckEntry:
    def test_middle_sample_with_cross_batch_neighbors(self):
        all_segments = [_seg(f"g{i}", f"source {i}") for i in range(7)]
        parsed = {0: "译2", 1: "译3", 2: "译4"}
        entry = align_check._build_check_entry(3, all_segments, batch_idx=2, batch_len=3, parsed_texts=parsed)

        assert entry["source"] == "source 3"
        assert entry["translation"] == "译3"
        assert entry["before"] == [
            {"source": "source 1"},
            {"source": "source 2", "translation": "译2"},
        ]
        assert entry["after"] == [
            {"source": "source 4", "translation": "译4"},
            {"source": "source 5"},
        ]

    def test_first_global_index_has_empty_before(self):
        all_segments = [_seg("g0"), _seg("g1"), _seg("g2")]
        entry = align_check._build_check_entry(0, all_segments, batch_idx=0, batch_len=1, parsed_texts={0: "译0"})
        assert entry["before"] == []
        assert len(entry["after"]) == 2

    def test_last_global_index_has_empty_after(self):
        all_segments = [_seg("g0"), _seg("g1"), _seg("g2")]
        entry = align_check._build_check_entry(2, all_segments, batch_idx=2, batch_len=1, parsed_texts={0: "译2"})
        assert entry["after"] == []
        assert len(entry["before"]) == 2
        assert "translation" not in entry["before"][0]


class TestBuildAlignPayload:
    def test_checks_structure_without_unit_id(self):
        all_segments = [_seg(f"g{i}", f"source {i}") for i in range(5)]
        batch = all_segments[1:4]
        parsed = {0: "译1", 1: "译2", 2: "译3"}
        payload = align_check._build_align_payload(batch, parsed, [1], all_segments, batch_idx=1, config=_config())
        assert payload["target_lang"] == "zh"
        assert len(payload["checks"]) == 1
        check = payload["checks"][0]
        assert check["source"] == "source 2"
        assert check["translation"] == "译2"
        assert "unit_id" not in check
        assert "translation" not in check["before"][0]

    def test_sampled_units_each_have_check_entry(self):
        batch = [_seg("b0"), _seg("b1"), _seg("b2"), _seg("b3"), _seg("b4")]
        parsed = {i: f"译{i}" for i in range(5)}
        payload = align_check._build_align_payload(batch, parsed, [0, 2, 4], batch, batch_idx=0, config=_config())
        assert len(payload["checks"]) == 3
        for check in payload["checks"]:
            assert "source" in check
            assert "translation" in check
            assert "before" in check
            assert "after" in check


class TestParseAlignResponse:
    def test_high_confidence_misaligned(self):
        segments = [_seg("u0"), _seg("u1"), _seg("u2")]
        response = json.dumps(
            [
                {"aligned": True, "reason": "", "confidence": 0.95},
                {"aligned": False, "reason": "wrong topic", "confidence": 0.92},
                {"aligned": True, "reason": "", "confidence": 0.90},
            ]
        )
        parsed = {0: "a", 1: "b", 2: "c"}
        aligned, failures = align_check._parse_align_response(response, 3, [0, 1, 2], segments, parsed)
        assert aligned is False
        assert len(failures) == 1
        assert failures[0].unit_id == "u1"
        assert failures[0].reason == "wrong topic"
        assert failures[0].confidence == 0.92
        assert failures[0].translation == "b"

    def test_low_confidence_false_passes(self):
        segments = [_seg("u0")]
        response = json.dumps([{"aligned": False, "reason": "maybe wrong", "confidence": 0.60}])
        aligned, failures = align_check._parse_align_response(response, 1, [0], segments, {0: "x"})
        assert aligned is True
        assert failures == []

    def test_length_mismatch_conservative_pass(self):
        segments = [_seg("u0"), _seg("u1")]
        response = json.dumps([{"aligned": False, "reason": "x", "confidence": 0.99}])
        aligned, failures = align_check._parse_align_response(response, 2, [0, 1], segments, {0: "a", 1: "b"})
        assert aligned is True
        assert failures == []


class TestCheckBatchAlignment:
    def test_all_aligned(self):
        client = MagicMock()
        client.chat.return_value = (
            json.dumps([{"aligned": True, "reason": "", "confidence": 0.95}]),
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        segments = [_seg("u0", "hello")]
        aligned, failures, usage = align_check.check_batch_alignment(
            client, segments, {0: "你好"}, segments, 0, _config()
        )
        assert aligned is True
        assert failures == []
        assert usage["total_tokens"] == 15
        client.chat.assert_called_once()
        user_payload = json.loads(client.chat.call_args[0][0][1]["content"])
        assert "checks" in user_payload
        assert "before" in user_payload["checks"][0]

    def test_misaligned_returns_failures(self):
        client = MagicMock()
        client.chat.return_value = (
            json.dumps([{"aligned": False, "reason": "clear mismatch", "confidence": 0.91}]),
            {},
        )
        segments = [_seg("mid", "scheduling")]
        aligned, failures, _ = align_check.check_batch_alignment(
            client, segments, {0: "价格讨论"}, segments, 0, _config()
        )
        assert aligned is False
        assert failures[0].unit_id == "mid"
        assert failures[0].reason == "clear mismatch"

    def test_format_align_failures_includes_source_and_translation(self):
        failure = align_check.AlignFailure(
            unit_id="u1",
            reason="wrong topic",
            confidence=0.91,
            source="scheduling meeting",
            translation="价格讨论",
        )
        text = align_check.format_align_failures([failure])
        assert "u1" in text
        assert "scheduling meeting" in text
        assert "价格讨论" in text


class TestTranslateBatchAlignmentRetry:
    def _translation_response(self, segments: list[Segment]) -> str:
        items = [{"batch_index": i, "text": f"译{i}"} for i in range(len(segments))]
        return json.dumps(items)

    def _align_response(self, aligned: bool, confidence: float = 0.95) -> str:
        return json.dumps([{"aligned": aligned, "reason": "" if aligned else "bad", "confidence": confidence}])

    def test_high_confidence_misaligned_retries_then_succeeds(self):
        segments = [_seg("u0")]
        config = _config()
        client = MagicMock()
        client.chat.side_effect = [
            (self._translation_response(segments), {"total_tokens": 1}),
            (self._align_response(False, 0.92), {"total_tokens": 2}),
            (self._translation_response(segments), {"total_tokens": 3}),
            (self._align_response(True), {"total_tokens": 4}),
        ]

        with patch(
            "light_subtitle.translate.translate._render_translate_prompt",
            return_value="system",
        ):
            cues, _, _ = _translate_batch(client, "system", segments, segments, 0, config)

        assert len(cues) == 1
        assert cues[0].text == "译0。"
        assert client.chat.call_count == 4

    def test_low_confidence_false_does_not_retry(self):
        segments = [_seg("u0")]
        config = _config()
        client = MagicMock()
        client.chat.side_effect = [
            (self._translation_response(segments), {"total_tokens": 1}),
            (self._align_response(False, 0.55), {"total_tokens": 2}),
        ]

        with patch(
            "light_subtitle.translate.translate._render_translate_prompt",
            return_value="system",
        ):
            cues, _, _ = _translate_batch(client, "system", segments, segments, 0, config)

        assert len(cues) == 1
        assert client.chat.call_count == 2

    def test_exhausted_alignment_retries_soft_fail(self):
        segments = [_seg("u0")]
        config = _config()
        client = MagicMock()
        client.chat.side_effect = [
            (self._translation_response(segments), {}),
            (self._align_response(False, 0.95), {}),
            (self._translation_response(segments), {}),
            (self._align_response(False, 0.95), {}),
            (self._translation_response(segments), {}),
            (self._align_response(False, 0.95), {}),
        ]

        with patch(
            "light_subtitle.translate.translate._render_translate_prompt",
            return_value="system",
        ):
            cues, _, _ = _translate_batch(client, "system", segments, segments, 0, config)

        assert len(cues) == 1
        assert client.chat.call_count == 6
