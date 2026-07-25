"""Tests for intra-stage fractional progress callbacks (pipeline batch loops)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from light_models import Segment, SubtitleCue, Word
from light_subtitle import annotate as annotate_pipeline
from light_subtitle.config import TranslateConfig
from light_subtitle.translate.translate import run as translate_run


def _seg(unit_id: str, *, start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(
        unit_id=unit_id,
        start=start,
        end=end,
        speaker="",
        source_text="hello",
        words=[Word(text="hello", start=start, end=end, confidence=0.9)],
    )


def _cue(unit_id: str, *, start: float = 0.0, end: float = 1.0) -> SubtitleCue:
    return SubtitleCue(
        cue_id=f"zh_{unit_id}",
        unit_id=unit_id,
        start=start,
        end=end,
        text="你好",
        lang="zh",
    )


def _config(**kwargs) -> TranslateConfig:
    return TranslateConfig(target_lang="zh", **kwargs)


class TestTranslateBatchProgress:
    def test_progress_fires_per_batch_monotonic(self):
        # 150 plain units → 2 chunks at CHUNK_SIZE=100.
        segments = [_seg(f"p{i:04d}") for i in range(150)]
        calls: list[tuple[float, str]] = []

        def fake_batch(client, system_prompt, chunk, all_segments, batch_idx, config, *, align_system_prompt=None):
            return [_cue(s.unit_id) for s in chunk], {}, {}

        with patch("light_subtitle.translate.translate._translate_batch", side_effect=fake_batch):
            translate_run(segments, _config(), llm=MagicMock(), progress=lambda f, m: calls.append((f, m)))

        fractions = [f for f, _ in calls]
        assert fractions == sorted(fractions)
        assert all(0.0 < f <= 1.0 for f in fractions)
        assert fractions[-1] == 1.0
        assert [m for _, m in calls] == [f"翻译中... {i}/{len(fractions)}" for i in range(1, len(fractions) + 1)]

    def test_no_progress_param_keeps_zero_overhead(self):
        segments = [_seg(f"p{i:04d}") for i in range(5)]

        def fake_batch(client, system_prompt, chunk, all_segments, batch_idx, config, *, align_system_prompt=None):
            return [_cue(s.unit_id) for s in chunk], {}, {}

        with patch("light_subtitle.translate.translate._translate_batch", side_effect=fake_batch):
            cues, _usage = translate_run(segments, _config(), llm=MagicMock())  # no progress — must not raise
        assert len(cues) == 5


class TestAnnotateBatchProgress:
    def test_progress_fires_per_batch(self):
        # 25 cues → 2 batches at BATCH_SIZE=20.
        cues = [_cue(f"p{i:04d}") for i in range(25)]
        segments = [_seg(f"p{i:04d}") for i in range(25)]
        calls: list[tuple[float, str]] = []

        class _FakeClient:
            def __init__(self):
                self.model = "fake"

            def chat(self, messages, temperature=0.0):
                return json.dumps([{"unit_id": "p0000", "annotation": "RL：强化学习"}]), {"total_tokens": 1}

        annotate_pipeline.generate_annotations(
            cues, segments, llm=_FakeClient(), progress=lambda f, m: calls.append((f, m))
        )

        assert calls == [
            (0.5, "生成注解中... 1/2"),
            (1.0, "生成注解中... 2/2"),
        ]
