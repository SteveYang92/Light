"""Retry missing translations respects display-merged unit coverage."""

from __future__ import annotations

from unittest.mock import patch

from light_models import Segment, SubtitleCue, Word
from light_subtitle.config import TranslateConfig
from light_subtitle.translate.retry import retry_missing


def _config() -> TranslateConfig:
    return TranslateConfig(target_lang="zh")


def _seg(unit_id: str) -> Segment:
    return Segment(
        unit_id=unit_id,
        start=0.0,
        end=1.0,
        speaker="",
        source_text="x",
        words=[Word(text="x", start=0.0, end=1.0, confidence=0.9)],
    )


class TestRetryMissingWithMerge:
    def test_does_not_retry_absorbed_unit_ids(self):
        segments = [_seg("u0"), _seg("u1"), _seg("u2")]
        cues = [
            SubtitleCue(
                cue_id="zh_0000",
                unit_id="u0",
                start=0.0,
                end=3.0,
                text="merged text",
                lang="zh",
                merged_from=["u1"],
            ),
            SubtitleCue(cue_id="zh_0001", unit_id="u2", start=3.0, end=4.0, text="solo", lang="zh"),
        ]

        with patch("light_subtitle.translate.retry.translate_missing") as mock_retry:
            result, _ = retry_missing(cues, segments, _config(), None)
            mock_retry.assert_not_called()

        assert result == cues
