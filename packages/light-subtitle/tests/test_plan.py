"""Tests for the cue planner (pipeline/plan).

LLM calls are mocked: planner tests patch OpenAIClient.chat, and plan.run
tests patch planner.plan_groups / split_span directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from light_models import Segment, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline import plan as plan_module
from light_subtitle.pipeline.plan import fallback, planner


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=1.0)


def _seg(unit_id: str, text: str, words: list[Word], speaker: str = "") -> Segment:
    return Segment(
        unit_id=unit_id,
        start=words[0].start,
        end=words[-1].end,
        speaker=speaker,
        source_text=text,
        words=words,
    )


def _config(max_duration: float = 7.0, **kwargs) -> SubtitleConfig:
    kwargs.setdefault("llm_api_key", "")
    return SubtitleConfig(input_path="dummy.mp4", target_lang="zh", max_duration=max_duration, **kwargs)


def _fragments() -> list[Segment]:
    """Three fragments forming two sentences."""
    return [
        _seg("u0001", "Well,", [_word("Well,", 0.0, 0.5)]),
        _seg(
            "u0002",
            "the agents are working.",
            [_word("the", 0.6, 0.8), _word("agents", 0.8, 1.2), _word("are", 1.2, 1.4), _word("working.", 1.4, 2.0)],
        ),
        _seg("u0003", "Right.", [_word("Right.", 2.2, 2.6)]),
    ]


# ── fallback.merge_fragments ──────────────────────────────


def test_merge_fragments_joins_sentence_fragments() -> None:
    groups = fallback.merge_fragments(_fragments())
    assert groups == [[0, 1], [2]]


def test_merge_fragments_stops_at_long_gap() -> None:
    segs = _fragments()
    # Push "the agents..." 5s later so the gap from "Well," exceeds 3s.
    segs[1] = _seg(
        "u0002",
        "the agents are working.",
        [_word("the", 6.0, 6.2), _word("agents", 6.2, 6.6), _word("are", 6.6, 6.8), _word("working.", 6.8, 7.4)],
    )
    assert fallback.merge_fragments(segs) == [[0], [1], [2]]


def test_merge_fragments_stops_at_speaker_change() -> None:
    segs = [
        _seg("u0001", "yes", [_word("yes", 0.0, 0.5)], speaker="S1"),
        _seg("u0002", "no", [_word("no", 0.6, 1.0)], speaker="S2"),
    ]
    assert fallback.merge_fragments(segs) == [[0], [1]]


# ── fallback.split_at_gaps ────────────────────────────────


def test_split_at_gaps_splits_at_largest_silence() -> None:
    # Two 2.4s speech runs separated by a 1.6s silence.
    words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(5)]
    words += [_word(f"w{i}", (i - 5) * 0.5 + 4.0, (i - 5) * 0.5 + 4.4) for i in range(5, 10)]
    ranges = fallback.split_at_gaps(words, max_duration=3.0)
    assert ranges == [(0, 5), (5, 10)]


def test_split_at_gaps_recurses_until_all_parts_fit() -> None:
    words = [_word(f"w{i}", float(i) * 0.5, float(i) * 0.5 + 0.4) for i in range(30)]
    ranges = fallback.split_at_gaps(words, max_duration=3.0)
    assert len(ranges) > 2
    for s, e in ranges:
        assert words[e - 1].end - words[s].start <= 3.0
    # ranges tile the whole span in order
    assert ranges[0][0] == 0 and ranges[-1][1] == 30
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))


def test_split_at_gaps_merges_stub_tail_into_previous() -> None:
    # 10 words over ~9.9s: a naive 3s-gap split would leave a 2-word tail;
    # the stub fold must attach it to the previous part instead.
    words = [_word(f"w{i}", float(i), float(i) + 0.9) for i in range(10)]
    ranges = fallback.split_at_gaps(words, max_duration=3.0)
    assert all(e - s >= 3 or len(ranges) == 1 for s, e in ranges)
    assert ranges[0][0] == 0 and ranges[-1][1] == 10
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))


# ── planner parsing & validation ──────────────────────────


def test_parse_groups_accepts_valid_json() -> None:
    assert planner._parse_groups('{"cues": [[0, 1], [2]]}') == [[0, 1], [2]]
    assert planner._parse_groups('blah {"cues": [[0]]} trailing') == [[0]]


def test_parse_groups_rejects_bad_shapes() -> None:
    assert planner._parse_groups("not json") is None
    assert planner._parse_groups('{"cues": "nope"}') is None
    assert planner._parse_groups('{"cues": [[0, "a"]]}') is None
    assert planner._parse_groups('{"cues": []}') is None


def test_group_problems_checks_coverage() -> None:
    segs = _fragments()
    assert planner._group_problems([[0, 1], [2]], segs) == []
    assert planner._group_problems([[0], [2]], segs)  # missing segment 1
    assert planner._group_problems([[1], [0], [2]], segs)  # out of order


def test_group_problems_checks_speaker_mixing() -> None:
    segs = [
        _seg("u0001", "yes", [_word("yes", 0.0, 0.5)], speaker="S1"),
        _seg("u0002", "no", [_word("no", 0.6, 1.0)], speaker="S2"),
    ]
    problems = planner._group_problems([[0, 1]], segs)
    assert problems and "speaker" in problems[0].lower()


def test_parse_breaks_and_validation() -> None:
    assert planner._parse_breaks('{"breaks": [{"after": 2}, {"after": 5}]}') == [2, 5]
    assert planner._parse_breaks('{"breaks": [2, 5]}') == [2, 5]
    assert planner._parse_breaks('{"breaks": [{"after": "x"}]}') is None

    words = [_word(f"w{i}", float(i), float(i) + 0.9) for i in range(10)]  # span ~9.9s
    # break at 4 → parts 4.9s / 5.0s, both under 7*1.15
    assert planner._break_problems([4], words, 7.0) == []
    # no breaks → problem
    assert planner._break_problems([], words, 7.0)
    # out of range
    assert planner._break_problems([9], words, 7.0)
    # unsorted
    assert planner._break_problems([5, 2], words, 7.0)
    # break at 0 → second part 8.9s > 8.05 cap → problem
    assert planner._break_problems([0], words, 7.0)
    # break at 1 → first part only 2 words (< 3) → problem
    assert any("words" in p for p in planner._break_problems([1], words, 7.0))
    # span too small to enforce the 3-word minimum → duration-only checks
    short = [_word(f"w{i}", float(i) * 2.0, float(i) * 2.0 + 1.9) for i in range(4)]  # 4 words, ~7.9s
    assert planner._break_problems([1], short, 7.0) == []


def test_breaks_to_ranges() -> None:
    assert planner._breaks_to_ranges([2, 5], 8) == [(0, 3), (3, 6), (6, 8)]


# ── planner LLM flow (mocked chat) ────────────────────────


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def test_plan_groups_retries_with_feedback_then_succeeds() -> None:
    segs = _fragments()
    client = _FakeClient(['{"cues": [[0], [2]]}', '{"cues": [[0, 1], [2]]}'])
    with patch("light_subtitle.pipeline.plan.planner.OpenAIClient", return_value=client):
        groups, usage = planner.plan_groups(segs, _config(llm_api_key="k"))
    assert groups == [[0, 1], [2]]
    assert usage is not None
    # second call carries validation feedback
    assert "previous_error" in client.calls[1][1]["content"]


def test_plan_groups_returns_none_after_two_invalid_attempts() -> None:
    segs = _fragments()
    client = _FakeClient(['{"cues": [[0]]}', '{"cues": [[0]]}'])
    with patch("light_subtitle.pipeline.plan.planner.OpenAIClient", return_value=client):
        groups, _ = planner.plan_groups(segs, _config(llm_api_key="k"))
    assert groups is None


def test_plan_groups_none_without_api_key() -> None:
    groups, usage = planner.plan_groups(_fragments(), _config())
    assert groups is None and usage is None


# ── plan.run end to end (mocked planner) ──────────────────


def test_run_materializes_units_and_writes_plan_json(tmp_path: Path) -> None:
    segs = _fragments()
    with patch("light_subtitle.pipeline.plan.planner.plan_groups", return_value=([[0, 1], [2]], None)):
        units, _ = plan_module.run(segs, _config(), tmp_path)

    assert [u.unit_id for u in units] == ["p0000", "p0001"]
    assert units[0].source_text == "Well, the agents are working."
    assert units[0].start == 0.0 and units[0].end == 2.0
    assert len(units[0].words) == 5

    data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert [u["unit_id"] for u in data["units"]] == ["p0000", "p0001"]
    assert data["units"][0]["word_start"] == 0 and data["units"][0]["word_end"] == 5

    loaded = plan_module.load_plan_units(tmp_path)
    assert loaded is not None and [u.unit_id for u in loaded] == ["p0000", "p0001"]


def test_run_falls_back_when_llm_unavailable(tmp_path: Path) -> None:
    # No API key → planner returns None → deterministic merge fallback.
    units, usage = plan_module.run(_fragments(), _config(), tmp_path)
    assert usage is None
    assert [u.unit_id for u in units] == ["p0000", "p0001"]
    assert units[0].source_text == "Well, the agents are working."


def test_run_splits_overlong_group_at_word_level(tmp_path: Path) -> None:
    words = [_word(f"w{i}", float(i), float(i) + 0.9) for i in range(10)]  # ~9.9s span
    segs = [_seg("u0001", "one two", words[:5]), _seg("u0002", "three four", words[5:])]
    with (
        patch("light_subtitle.pipeline.plan.planner.plan_groups", return_value=([[0, 1]], None)),
        patch("light_subtitle.pipeline.plan.planner.split_span", return_value=([(0, 5), (5, 10)], None)),
    ):
        units, _ = plan_module.run(segs, _config(llm_api_key="fake"), tmp_path)

    assert [u.unit_id for u in units] == ["p0000_0", "p0000_1"]
    assert units[0].start == 0.0 and units[0].end == 4.9
    assert units[1].start == 5.0 and units[1].end == 9.9  # last part keeps group end
    assert units[0].source_text == "w0 w1 w2 w3 w4"


def test_run_overlong_falls_back_to_gap_split(tmp_path: Path) -> None:
    words = [_word(f"w{i}", float(i), float(i) + 0.9) for i in range(10)]
    segs = [_seg("u0001", "one two", words[:5]), _seg("u0002", "three four", words[5:])]
    with (
        patch("light_subtitle.pipeline.plan.planner.plan_groups", return_value=([[0, 1]], None)),
        patch("light_subtitle.pipeline.plan.planner.split_span", return_value=(None, None)),
    ):
        units, _ = plan_module.run(segs, _config(max_duration=3.0, llm_api_key="fake"), tmp_path)
    assert len(units) >= 3  # 9.9s split into ≤3s parts at silence gaps


def test_run_empty_segments(tmp_path: Path) -> None:
    units, usage = plan_module.run([], _config(), tmp_path)
    assert units == [] and usage is None
    assert json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))["units"] == []


def test_load_plan_units_missing_returns_none(tmp_path: Path) -> None:
    assert plan_module.load_plan_units(tmp_path) is None
