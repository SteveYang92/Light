"""Model serialization roundtrips and report aggregation."""

from __future__ import annotations

import json

from light_eval.models import Annotation, CaseResult, DimensionScore, EvalCase, EvalReport, StepOutput


def _case() -> EvalCase:
    return EvalCase(name="c1", step="plan", kind="control", source="test", params={"target_lang": "zh"})


def test_step_output_roundtrip() -> None:
    output = StepOutput(
        case="c1",
        output=[{"unit_id": "p0000", "text": "hello"}],
        usage={"total_tokens": 42},
        duration_s=1.234,
    )
    restored = StepOutput.from_dict(json.loads(json.dumps(output.to_dict())))
    assert restored.case == output.case
    assert restored.output == output.output
    assert restored.usage == output.usage
    assert restored.duration_s == output.duration_s
    assert restored.error is None
    assert restored.skipped is False


def test_dimension_score_roundtrip() -> None:
    score = DimensionScore(dimension="word_coverage", score=0.5, passed=False, detail="1/2", evidence=["w1"])
    restored = DimensionScore.from_dict(json.loads(json.dumps(score.to_dict())))
    assert restored == score


def test_dimension_score_issues_roundtrip() -> None:
    score = DimensionScore(
        dimension="faithfulness",
        score=3.0,
        passed=False,
        detail="漏译一处",
        evidence=["p0001"],
        issues=[{"unit_id": "p0001", "problem": "漏译数字"}],
    )
    data = score.to_dict()
    assert data["issues"] == [{"unit_id": "p0001", "problem": "漏译数字"}]
    restored = DimensionScore.from_dict(json.loads(json.dumps(data)))
    assert restored == score


def test_dimension_score_empty_issues_omitted() -> None:
    """Empty issues stay absent from the serialized form (base schema stable)."""
    assert "issues" not in DimensionScore(dimension="d", score=1.0, passed=True).to_dict()


def test_annotation_roundtrip_without_suggestion() -> None:
    """Base schema (old annotation.yaml) — optional fields stay absent, not null."""
    ann = Annotation(dimensions={"boundary_quality": 4}, defects=[{"unit_id": "u1", "issue": "碎"}], overall="pass")
    data = json.loads(json.dumps(ann.to_dict()))
    assert "judge_suggestion" not in data
    assert "reviewed_by" not in data
    restored = Annotation.from_dict(data)
    assert restored == ann
    assert restored.judge_suggestion is None
    assert restored.reviewed_by == ""


def test_annotation_roundtrip_with_suggestion() -> None:
    suggestion = {
        "dimensions": {"boundary_quality": {"score": 5, "reason": "边界合理", "evidence": ["u1"]}},
        "suggested_overall": "pass",
    }
    ann = Annotation(
        dimensions={"boundary_quality": 4},
        defects=[],
        overall="pass",
        judge_suggestion=suggestion,
        reviewed_by="human",
    )
    restored = Annotation.from_dict(json.loads(json.dumps(ann.to_dict())))
    assert restored == ann
    assert restored.judge_suggestion == suggestion
    assert restored.reviewed_by == "human"


def test_eval_report_to_json_and_aggregate() -> None:
    good = DimensionScore(dimension="word_coverage", score=1.0, passed=True)
    bad = DimensionScore(dimension="empty_units", score=1.0, passed=False, evidence=["p0001"])
    report = EvalReport(
        step="plan",
        cases=[
            CaseResult(case=_case(), scores=[good], output=StepOutput(case="c1", output=[{"unit_id": "p0000"}])),
            CaseResult(case=_case(), scores=[bad], output=StepOutput(case="c2", output=[])),
            CaseResult(case=_case(), scores=[], output=StepOutput(case="c3", skipped=True)),
            CaseResult(case=_case(), scores=[], output=StepOutput(case="c4", error="boom")),
        ],
    )
    agg = report.aggregate()
    assert agg["n_cases"] == 4
    assert agg["n_passed"] == 1
    assert agg["n_skipped"] == 1
    assert agg["n_errored"] == 1
    assert agg["dimensions"] == {
        "word_coverage": {"total": 1, "passed": 1},
        "empty_units": {"total": 1, "passed": 0},
    }

    payload = json.loads(report.to_json())
    assert payload["step"] == "plan"
    assert payload["aggregate"]["n_cases"] == 4
    assert len(payload["cases"]) == 4


def test_eval_report_save(tmp_path) -> None:
    report = EvalReport(step="plan")
    path = report.save(tmp_path / "sub" / "report.json")
    assert json.loads(path.read_text(encoding="utf-8"))["step"] == "plan"
