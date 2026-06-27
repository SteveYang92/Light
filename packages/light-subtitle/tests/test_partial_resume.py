"""Tests for partial.json checkpoint (1:1 cues + merge_hints)."""

from __future__ import annotations

import json
from unittest.mock import patch

from light_models import Segment, SubtitleCue, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.merge_apply import covered_unit_ids
from light_subtitle.pipeline.translate.translate import (
    _dedupe_hint_records,
    _hints_from_records,
    _save_partial,
    _segment_graph_fingerprint,
    load_partial,
    run,
)


def _config(**kwargs) -> SubtitleConfig:
    return SubtitleConfig(input_path="dummy.mp4", target_lang="zh", llm_api_key="test", **kwargs)


def _seg(unit_id: str, *, start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(
        unit_id=unit_id,
        start=start,
        end=end,
        speaker="",
        source_text="hello",
        words=[Word(text="hello", start=start, end=end, confidence=0.9)],
    )


def _cue(unit_id: str, text: str, *, start: float = 0.0, end: float = 1.0) -> SubtitleCue:
    return SubtitleCue(
        cue_id=f"zh_{unit_id}",
        unit_id=unit_id,
        start=start,
        end=end,
        text=text,
        lang="zh",
    )


class TestPartialSchema:
    def test_save_and_load_wrapper(self, tmp_path):
        segments = [_seg("u0"), _seg("u1", start=1.0, end=2.0)]
        cues = [_cue("u0", "a"), _cue("u1", "b", start=1.0, end=2.0)]
        hints = [{"curr_unit_id": "u0", "next_unit_id": "u1"}]
        _save_partial(tmp_path, cues, hints, segments)

        loaded_cues, loaded_hints = load_partial(tmp_path, _config())
        assert len(loaded_cues) == 2
        assert loaded_cues[0].unit_id == "u0"
        assert loaded_cues[0].merged_from == []
        assert loaded_hints == hints

        raw = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
        assert raw["version"] == 1
        assert "segments_fingerprint" in raw
        assert "merge_hints" in raw

    def test_legacy_array_without_merged_from(self, tmp_path):
        legacy = [{"cue_id": "zh_0", "unit_id": "u0", "start": 0, "end": 1, "text": "a", "lang": "zh"}]
        (tmp_path / "partial.json").write_text(json.dumps(legacy), encoding="utf-8")
        cues, hints = load_partial(tmp_path, _config())
        assert len(cues) == 1
        assert hints == []

    def test_dedupe_hint_records(self):
        records = [
            {"curr_unit_id": "u0", "next_unit_id": "u1"},
            {"curr_unit_id": "u0", "next_unit_id": "u1"},
            {"curr_unit_id": "u1", "next_unit_id": "u2"},
        ]
        assert len(_dedupe_hint_records(records)) == 2

    def test_hints_from_records(self):
        segments = [_seg("u0"), _seg("u1", start=1.0, end=2.0)]
        records = [{"curr_unit_id": "u0", "next_unit_id": "u1"}]
        hints = _hints_from_records(segments, records)
        assert len(hints) == 1
        assert hints[0][0].unit_id == "u0"
        assert hints[0][1].unit_id == "u1"


class TestPartialResumeRun:
    def test_resume_skips_covered_unit_ids(self, tmp_path):
        segments = [_seg("u0"), _seg("u1", start=1.0, end=2.0), _seg("u2", start=2.0, end=3.0)]
        _save_partial(
            tmp_path,
            [
                _cue("u0", "a"),
                _cue("u1", "b", start=1.0, end=2.0),
                _cue("u2", "solo", start=2.0, end=3.0),
            ],
            [],
            segments,
        )

        with patch("light_subtitle.pipeline.translate.translate._translate_batch") as mock_batch:
            cues, _ = run(segments, _config(merge_hints_apply=False), tx_dir=tmp_path)
            mock_batch.assert_not_called()

        assert len(cues) == 3

    def test_pending_uses_covered_unit_ids_not_only_head(self, tmp_path):
        legacy = [
            {
                "cue_id": "zh_u0",
                "unit_id": "u0",
                "start": 0.0,
                "end": 2.0,
                "text": "merged",
                "lang": "zh",
                "merged_from": ["u1"],
            }
        ]
        (tmp_path / "partial.json").write_text(json.dumps(legacy), encoding="utf-8")
        loaded, _ = load_partial(tmp_path, _config())
        covered = covered_unit_ids(loaded)
        assert covered == {"u0", "u1"}

    def test_early_exit_applies_merge_hints(self, tmp_path):
        segments = [_seg("u0"), _seg("u1", start=1.0, end=2.0)]
        _save_partial(
            tmp_path,
            [_cue("u0", "part0"), _cue("u1", "part1", start=1.0, end=2.0)],
            [{"curr_unit_id": "u0", "next_unit_id": "u1"}],
            segments,
        )

        with patch("light_subtitle.pipeline.translate.translate._translate_batch") as mock_batch:
            cues, _ = run(segments, _config(), tx_dir=tmp_path)
            mock_batch.assert_not_called()

        assert len(cues) == 1
        assert cues[0].merged_from == ["u1"]
        assert "part0" in cues[0].text and "part1" in cues[0].text


class TestPartialStaleDiscard:
    def test_translate_discards_partial_when_segment_graph_changes(self, tmp_path):
        old_segments = [_seg("u0")]
        new_segments = [_seg("u_new")]
        _save_partial(tmp_path, [_cue("u0", "stale")], [], old_segments)

        with patch("light_subtitle.pipeline.translate.translate._translate_batch") as mock_batch:
            mock_batch.return_value = ([_cue("u_new", "fresh")], {}, [])
            run(new_segments, _config(merge_hints_apply=False), tx_dir=tmp_path)
            mock_batch.assert_called_once()

        raw = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
        assert raw["segments_fingerprint"] == _segment_graph_fingerprint(new_segments)

    def test_translate_keeps_partial_when_segment_graph_matches(self, tmp_path):
        segments = [_seg("u0"), _seg("u1", start=1.0, end=2.0)]
        _save_partial(tmp_path, [_cue("u0", "a"), _cue("u1", "b", start=1.0, end=2.0)], [], segments)

        with patch("light_subtitle.pipeline.translate.translate._translate_batch") as mock_batch:
            cues, _ = run(segments, _config(merge_hints_apply=False), tx_dir=tmp_path)
            mock_batch.assert_not_called()

        assert len(cues) == 2
