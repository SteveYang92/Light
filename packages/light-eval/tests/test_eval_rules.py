"""Rule-judge tests — hand-built outputs with known violations."""

from __future__ import annotations

from light_eval.judges.rules import PlanRulesJudge, TranslateRulesJudge, judge_for_step
from light_eval.loader import Fixture
from light_eval.models import EvalCase, StepOutput
from light_models import Segment, word_to_dict

from .conftest import make_segment, make_words


def _plan_case(params: dict | None = None) -> EvalCase:
    return EvalCase(name="c", step="plan", kind="control", source="test", params=params or {})


def _tx_case(params: dict | None = None) -> EvalCase:
    return EvalCase(name="c", step="translate", kind="control", source="test", params=params or {"target_lang": "zh"})


def _unit_dict(unit_id, words, *, start=None, end=None, text=None) -> dict:
    return {
        "unit_id": unit_id,
        "start": words[0].start if start is None else start,
        "end": words[-1].end if end is None else end,
        "text": " ".join(w.text for w in words) if text is None else text,
        "speaker": "",
        "words": [word_to_dict(w) for w in words],
    }


def _scores_by_dim(scores):
    return {s.dimension: s for s in scores}


# ── plan dimensions ─────────────────────────────────────────────────────────


def test_word_coverage_pass_and_fail() -> None:
    words = make_words(["hello", "world", "test", "case."])
    fixture = Fixture(segments=[make_segment("u0001", words)])
    judge = PlanRulesJudge()

    full = StepOutput(case="c", output=[_unit_dict("p0000", words)])
    score = _scores_by_dim(judge.score(_plan_case(), fixture, full))["word_coverage"]
    assert score.passed and score.score == 1.0

    dropped = StepOutput(case="c", output=[_unit_dict("p0000", words[:-1])])
    score = _scores_by_dim(judge.score(_plan_case(), fixture, dropped))["word_coverage"]
    assert not score.passed
    assert score.score == 0.75
    assert any("case." in e for e in score.evidence)


def test_duration_violations_respects_params_and_defaults() -> None:
    words = make_words(["a", "b", "c", "d"], step=3.0)  # unit spans 12s > 7s default
    fixture = Fixture(segments=[make_segment("u0001", words)])
    output = StepOutput(case="c", output=[_unit_dict("p0000", words)])
    judge = PlanRulesJudge()

    score = _scores_by_dim(judge.score(_plan_case(), fixture, output))["duration_violations"]
    assert not score.passed and score.score == 1.0

    score = _scores_by_dim(judge.score(_plan_case({"max_duration": 15.0}), fixture, output))["duration_violations"]
    assert score.passed and score.score == 0.0


def test_dangling_tails_flags_function_word_endings() -> None:
    words = make_words(["this", "is", "the"])  # ends on a function word
    fixture = Fixture(segments=[make_segment("u0001", words)])
    output = StepOutput(case="c", output=[_unit_dict("p0000", words)])
    score = _scores_by_dim(PlanRulesJudge().score(_plan_case(), fixture, output))["dangling_tails"]
    assert not score.passed
    assert any('"the"' in e for e in score.evidence)


def test_dangling_tails_exempts_clause_punctuation() -> None:
    words = make_words(["and", "then,", "and"])  # trailing "and" dangles; "then," is fine
    words[-1] = words[-1].__class__(text="and.", start=words[-1].start, end=words[-1].end, confidence=1.0)
    fixture = Fixture(segments=[make_segment("u0001", words)])
    output = StepOutput(case="c", output=[_unit_dict("p0000", words)])
    score = _scores_by_dim(PlanRulesJudge().score(_plan_case(), fixture, output))["dangling_tails"]
    assert score.passed


def test_empty_units_flagged() -> None:
    words = make_words(["a", "b"])
    fixture = Fixture(segments=[make_segment("u0001", words)])
    output = StepOutput(case="c", output=[_unit_dict("p0000", words, text="  ")])
    score = _scores_by_dim(PlanRulesJudge().score(_plan_case(), fixture, output))["empty_units"]
    assert not score.passed and score.evidence == ["p0000"]


# ── translate dimensions ────────────────────────────────────────────────────


def _tx_fixture() -> Fixture:
    units = [
        Segment(unit_id="p0000", start=0.0, end=2.0, speaker="", source_text="hello world", words=[]),
        Segment(unit_id="p0001", start=2.0, end=4.0, speaker="", source_text="test case", words=[]),
    ]
    return Fixture(segments=units)


def _cue_dict(cue_id, unit_id, start, end, text, merged_from=None) -> dict:
    data = {"cue_id": cue_id, "unit_id": unit_id, "start": start, "end": end, "text": text, "lang": "zh"}
    if merged_from:
        data["merged_from"] = merged_from
    return data


def test_unit_coverage_counts_merge_chains() -> None:
    fixture = _tx_fixture()
    judge = TranslateRulesJudge()

    partial = StepOutput(case="c", output=[_cue_dict("zh_0000", "p0000", 0.0, 2.0, "你好世界。")])
    score = _scores_by_dim(judge.score(_tx_case(), fixture, partial))["unit_coverage"]
    assert not score.passed and score.score == 0.5 and score.evidence == ["p0001"]

    merged = StepOutput(case="c", output=[_cue_dict("zh_0000", "p0000", 0.0, 4.0, "你好世界测试。", ["p0001"])])
    score = _scores_by_dim(judge.score(_tx_case(), fixture, merged))["unit_coverage"]
    assert score.passed and score.score == 1.0


def test_empty_translations_flagged() -> None:
    output = StepOutput(case="c", output=[_cue_dict("zh_0000", "p0000", 0.0, 2.0, "")])
    score = _scores_by_dim(TranslateRulesJudge().score(_tx_case(), _tx_fixture(), output))["empty_translations"]
    assert not score.passed and score.evidence == ["p0000"]


def test_target_lang_ratio_zh() -> None:
    judge = TranslateRulesJudge()
    fixture = _tx_fixture()

    zh = StepOutput(case="c", output=[_cue_dict("zh_0000", "p0000", 0.0, 2.0, "你好世界这是测试文本")])
    score = _scores_by_dim(judge.score(_tx_case(), fixture, zh))["target_lang_ratio"]
    assert score.passed and score.score == 1.0

    en = StepOutput(case="c", output=[_cue_dict("zh_0000", "p0000", 0.0, 2.0, "hello world")])
    score = _scores_by_dim(judge.score(_tx_case(), fixture, en))["target_lang_ratio"]
    assert not score.passed and score.score == 0.0


def test_source_fidelity_checks_timing_and_ids() -> None:
    judge = TranslateRulesJudge()
    fixture = _tx_fixture()

    good = StepOutput(case="c", output=[_cue_dict("zh_0000", "p0000", 0.0, 4.0, "你好世界测试。", ["p0001"])])
    score = _scores_by_dim(judge.score(_tx_case(), fixture, good))["source_fidelity"]
    assert score.passed and score.score == 1.0

    shifted = StepOutput(case="c", output=[_cue_dict("zh_0000", "p0000", 0.5, 2.0, "你好世界。")])
    score = _scores_by_dim(judge.score(_tx_case(), fixture, shifted))["source_fidelity"]
    assert not score.passed and any("timing" in e for e in score.evidence)

    unknown = StepOutput(case="c", output=[_cue_dict("zh_0000", "p9999", 0.0, 2.0, "你好。")])
    score = _scores_by_dim(judge.score(_tx_case(), fixture, unknown))["source_fidelity"]
    assert not score.passed and any("unknown" in e for e in score.evidence)


def test_judges_return_nothing_for_skipped_or_errored() -> None:
    skipped = StepOutput(case="c", skipped=True)
    errored = StepOutput(case="c", error="boom")
    assert PlanRulesJudge().score(_plan_case(), Fixture(), skipped) == []
    assert TranslateRulesJudge().score(_tx_case(), Fixture(), errored) == []


def test_judge_for_step() -> None:
    assert isinstance(judge_for_step("plan"), PlanRulesJudge)
    assert isinstance(judge_for_step("translate"), TranslateRulesJudge)
