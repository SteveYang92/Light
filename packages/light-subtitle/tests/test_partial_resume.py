"""Tests for partial.json checkpoint (1:1 cues, no merge hints)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from light_models import Segment, SubtitleCue, Word
from light_subtitle.config import TranslateConfig
from light_subtitle.translate.checkpoint import (
    _save_partial,
    load_partial,
    segment_graph_fingerprint,
)
from light_subtitle.translate.translate import (
    covered_unit_ids,
    run,
)


def _config(**kwargs) -> TranslateConfig:
    return TranslateConfig(target_lang="zh", **kwargs)


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
        _save_partial(tmp_path, cues, segments)

        loaded_cues = load_partial(tmp_path, _config())
        assert len(loaded_cues) == 2
        assert loaded_cues[0].unit_id == "u0"
        assert loaded_cues[0].merged_from == []

        raw = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
        assert raw["version"] == 2
        assert "segments_fingerprint" in raw

    def test_legacy_array_without_merged_from(self, tmp_path):
        legacy = [{"cue_id": "zh_0", "unit_id": "u0", "start": 0, "end": 1, "text": "a", "lang": "zh"}]
        (tmp_path / "partial.json").write_text(json.dumps(legacy), encoding="utf-8")
        cues = load_partial(tmp_path, _config())
        assert len(cues) == 1


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
            segments,
        )

        with patch("light_subtitle.translate.translate._translate_batch") as mock_batch:
            cues, _ = run(segments, _config(), tx_dir=tmp_path, llm=MagicMock())
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
        loaded = load_partial(tmp_path, _config())
        covered = covered_unit_ids(loaded)
        assert covered == {"u0", "u1"}

    def test_finalize_assigns_sequential_cue_ids(self, tmp_path):
        segments = [_seg("u0"), _seg("u1", start=1.0, end=2.0)]
        _save_partial(tmp_path, [_cue("u0", "a"), _cue("u1", "b", start=1.0, end=2.0)], segments)

        with patch("light_subtitle.translate.translate._translate_batch") as mock_batch:
            cues, _ = run(segments, _config(), tx_dir=tmp_path, llm=MagicMock())
            mock_batch.assert_not_called()

        assert [c.cue_id for c in cues] == ["zh_0000", "zh_0001"]


class TestPartialStaleDiscard:
    def test_translate_discards_partial_when_segment_graph_changes(self, tmp_path):
        old_segments = [_seg("u0")]
        new_segments = [_seg("u_new")]
        _save_partial(tmp_path, [_cue("u0", "stale")], old_segments)

        with patch("light_subtitle.translate.translate._translate_batch") as mock_batch:
            mock_batch.return_value = ([_cue("u_new", "fresh")], {}, {})
            run(new_segments, _config(), tx_dir=tmp_path, llm=MagicMock())
            mock_batch.assert_called_once()

        raw = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
        assert raw["segments_fingerprint"] == segment_graph_fingerprint(new_segments)

    def test_translate_keeps_partial_when_segment_graph_matches(self, tmp_path):
        segments = [_seg("u0"), _seg("u1", start=1.0, end=2.0)]
        _save_partial(tmp_path, [_cue("u0", "a"), _cue("u1", "b", start=1.0, end=2.0)], segments)

        with patch("light_subtitle.translate.translate._translate_batch") as mock_batch:
            cues, _ = run(segments, _config(), tx_dir=tmp_path, llm=MagicMock())
            mock_batch.assert_not_called()

        assert len(cues) == 2
