"""Tests for layout module — cue merging (short-adjacent)."""

from __future__ import annotations

from light_models import SubtitleCue
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.subtitle.layout import prepare

# ── Helpers ────────────────────────────────────────────


def _cue(
    cue_id: str = "c0",
    unit_id: str = "u0",
    start: float = 0.0,
    end: float = 2.0,
    text: str = "Hello world.",
    lang: str = "en",
    speaker: str = "",
) -> SubtitleCue:
    return SubtitleCue(
        cue_id=cue_id,
        unit_id=unit_id,
        start=start,
        end=end,
        text=text,
        lang=lang,
        speaker=speaker,
    )


def _config(**overrides) -> SubtitleConfig:
    """Build a SubtitleConfig with defaults suitable for testing."""
    defaults = {
        "input_path": "test.mp4",
        "output_dir": "./output",
        "max_lines": 2,
        "max_lines_zh": 1,
        "max_chars_per_line_zh": 40,
        "max_chars_per_line_en": 42,
        "max_duration": 7.0,
    }
    defaults.update(overrides)
    return SubtitleConfig(**defaults)


# ═══════════════════════════════════════════════════════
# prepare() — integration with split + merge pipeline
# ═══════════════════════════════════════════════════════


class TestPrepareIntegration:
    """Full prepare() pipeline: split → merge_short."""

    def test_prepare_merges_short_cues(self):
        """Cues that cannot stand alone merge via merge_short."""
        cues = [
            _cue("c0", "u0", 0.0, 2.0, "Hello"),
            _cue("c1", "u1", 2.1, 4.0, "world."),
        ]
        result = prepare(cues, _config())
        assert len(result) == 1

    def test_prepare_preserves_unmerged(self):
        """Cues with large gap stay separate through the full pipeline."""
        cues = [
            _cue("c0", "u0", 0.0, 2.0, "Hello"),
            _cue("c1", "u1", 5.0, 7.0, "world."),
        ]
        result = prepare(cues, _config())
        assert len(result) == 2

    def test_forward_conjunction_merge_records_merged_from(self):
        """Conjunction forward-merge must chain absorbed unit ids for bilingual export."""
        cues = [
            _cue("c_prev", "mu0288_u0288", 1108.713, 1109.213, "他们是被迫的。", lang="zh"),
            _cue(
                "c0",
                "mu0289_u0291_0_0",
                1110.394,
                1110.654,
                "所以，",
                lang="zh",
            ),
            _cue(
                "c1",
                "mu0289_u0291_0_1",
                1112.194,
                1117.696,
                "有很多儿子再也回不到父母身边了，你知道。",
                lang="zh",
            ),
        ]
        result = prepare(cues, _config())
        assert len(result) == 2
        merged = result[1]
        assert merged.unit_id == "mu0289_u0291_0_0"
        assert merged.merged_from == ["mu0289_u0291_0_1"]
        assert "所以" in merged.text.replace("，", "")
        assert "有很多儿子" in merged.text
