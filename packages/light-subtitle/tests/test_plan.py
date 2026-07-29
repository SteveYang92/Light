from __future__ import annotations

from pathlib import Path

from light_models import Segment, Word
from light_subtitle import plan as plan_module
from light_subtitle.config import PlanConfig


def _word(text: str, start: float, end: float, confidence: float = 1.0, speaker: str | None = None) -> Word:
    return Word(text=text, start=start, end=end, confidence=confidence, speaker=speaker)


def _seg(unit_id: str, text: str, words: list[Word], speaker: str = "") -> Segment:
    return Segment(
        unit_id=unit_id, start=words[0].start, end=words[-1].end,
        speaker=speaker, source_text=text, words=words,
    )


def _config(**kwargs) -> PlanConfig:
    return PlanConfig(**kwargs)


# ── S0 normalize ──────────────────────────────────────────


def test_normalize_speaker_fill() -> None:
    from light_subtitle.plan.normalize import normalize
    words = [_word("Hello", 0.0, 0.3), _word("world.", 0.5, 0.9)]
    nwords = normalize(words)
    assert all(nw.speaker == "UNKNOWN" for nw in nwords)


def test_not_sentence_final_for_dr() -> None:
    from light_subtitle.plan.normalize import normalize
    words = [_word("Dr.", 0.0, 0.3), _word("Smith", 0.5, 0.8)]
    nwords = normalize(words)
    assert nwords[0].is_sentence_final is False


# ── Planner ────────────────────────────────────────────────


def test_parse_breaks_valid() -> None:
    from light_subtitle.plan.planner import _parse_breaks
    assert _parse_breaks('{"breaks": [2, 5, 9]}', 10) == [2, 5, 9]


def test_parse_breaks_invalid() -> None:
    from light_subtitle.plan.planner import _parse_breaks
    assert _parse_breaks("not json", 10) is None
    assert _parse_breaks('{"breaks": [2, 2]}', 10) is None  # duplicates
    assert _parse_breaks('{"breaks": [5]}', 10) is None  # last != n-1
    assert _parse_breaks('{"breaks": [12]}', 10) is None  # out of range


def test_fix_illegal_tails() -> None:
    from light_subtitle.plan.planner import fix_illegal_tails
    words = [_word("the", 0.0, 0.2), _word("end.", 0.4, 0.8), _word("and", 1.0, 1.2), _word("so", 1.3, 1.5)]
    breaks = fix_illegal_tails([0], words)  # break after "the" → slide to after "end."
    assert breaks == [1]


# ── Gap split ──────────────────────────────────────────────


def test_gap_split_splits_long() -> None:
    from light_subtitle.plan.gap import gap_split
    words = [_word(f"w{i}", float(i) * 0.5, float(i) * 0.5 + 0.45) for i in range(30)]
    ranges = gap_split(words, 3.0)
    assert len(ranges) > 2
    for s, e in ranges:
        assert words[e - 1].end - words[s].start <= 3.0


def test_gap_split_stub_merge() -> None:
    from light_subtitle.plan.gap import gap_split
    words = [_word(f"w{i}", float(i), float(i) + 0.9) for i in range(10)]
    ranges = gap_split(words, 4.0)
    assert all(e - s >= 3 or len(ranges) == 1 for s, e in ranges)
    assert ranges[0][0] == 0 and ranges[-1][1] == 10


# ── End-to-end ─────────────────────────────────────────────


def test_run_empty(tmp_path: Path) -> None:
    units, usage = plan_module.run([], _config(), tmp_path)
    assert units == [] and usage is None


def test_run_no_llm(tmp_path: Path) -> None:
    segs = [_seg("u0001", "hello.", [_word("hello.", 0.0, 0.3)])]
    units, usage = plan_module.run(segs, _config(), tmp_path)
    assert units == [] and usage is None


def test_load_plan_units_missing(tmp_path: Path) -> None:
    assert plan_module.load_plan_units(tmp_path) is None
