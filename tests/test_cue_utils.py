from light_models import SubtitleCue, covered_source_text, covered_time_window, effective_unit_ids


def _cue(unit_id: str, *, merged_from: list[str] | None = None, start: float = 1.0, end: float = 4.0) -> SubtitleCue:
    return SubtitleCue(
        cue_id=unit_id,
        unit_id=unit_id,
        start=start,
        end=end,
        text="x",
        lang="zh",
        merged_from=merged_from or [],
    )


class TestEffectiveUnitIds:
    def test_head_only(self):
        assert effective_unit_ids(_cue("u0")) == {"u0"}

    def test_with_merged_from(self):
        assert effective_unit_ids(_cue("u0", merged_from=["u1", "u2"])) == {"u0", "u1", "u2"}


class TestCoveredSourceText:
    def test_head_only(self):
        cue = _cue("u0")
        assert covered_source_text(cue, {"u0": "Hello world"}) == "Hello world"

    def test_merged_english_joined_with_space(self):
        cue = _cue("u0", merged_from=["u1", "u2"])
        source_map = {"u0": "First", "u1": "second", "u2": "third"}
        assert covered_source_text(cue, source_map) == "First second third"

    def test_merged_chinese_concatenated(self):
        cue = _cue("u0", merged_from=["u1"])
        source_map = {"u0": "第一", "u1": "第二"}
        assert covered_source_text(cue, source_map) == "第一第二"


class TestCoveredTimeWindow:
    def test_single_unit(self):
        cue = _cue("u0")
        window = covered_time_window(cue, {"u0": (1.0, 3.5)})
        assert window == (1.0, 3.5)

    def test_merged_span(self):
        cue = _cue("u0", merged_from=["u1", "u2"], start=1.0, end=9.0)
        unit_times = {"u0": (1.0, 2.0), "u1": (2.5, 5.0), "u2": (5.5, 8.0)}
        assert covered_time_window(cue, unit_times) == (1.0, 8.0)

    def test_missing_bounds_returns_none(self):
        assert covered_time_window(_cue("u0"), {}) is None
