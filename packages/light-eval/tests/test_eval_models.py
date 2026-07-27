"""Model serialization roundtrips and report aggregation."""

from __future__ import annotations

import json

from light_eval.models import Annotation, CaseResult, Defect, EvalCase, EvalReport, ProblemTypeStats, StepOutput


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


def test_problem_type_stats_roundtrip() -> None:
    score = ProblemTypeStats(problem_type="semantic_boundary", error_count=1, warning_count=0, passed=False)
    restored = ProblemTypeStats.from_dict(json.loads(json.dumps(score.to_dict())))
    assert restored == score


def test_problem_type_stats_issues_roundtrip() -> None:
    score = ProblemTypeStats(
        problem_type="semantic_drift",
        error_count=1,
        warning_count=0,
        passed=False,
        evidence=["p0001"],
        issues=[{"unit_id": "p0001", "problem": "语义偏移"}],
    )
    data = score.to_dict()
    assert data["issues"] == [{"unit_id": "p0001", "problem": "语义偏移"}]
    restored = ProblemTypeStats.from_dict(json.loads(json.dumps(data)))
    assert restored == score


def test_problem_type_stats_empty_issues_omitted() -> None:
    assert "issues" not in ProblemTypeStats(problem_type="d", error_count=0, passed=True).to_dict()


def test_annotation_roundtrip_without_suggestion() -> None:
    ann = Annotation(
        defects=[Defect(unit_id="u1", problem_type="semantic_boundary", note="碎", confirmed=True)],
        overall="pass",
    )
    data = json.loads(json.dumps(ann.to_dict()))
    assert "judge_suggestion" not in data
    assert "reviewed_by" not in data
    restored = Annotation.from_dict(data)
    assert restored == ann
    assert restored.judge_suggestion is None
    assert restored.reviewed_by == ""


def test_annotation_roundtrip_with_suggestion() -> None:
    suggestion = {
        "problem_types": {"semantic_boundary": {"error_count": 1, "warning_count": 0, "passed": False}},
        "suggested_overall": "fail",
    }
    ann = Annotation(
        defects=[Defect(unit_id="u1", problem_type="semantic_boundary", note="边界不当", confirmed=True)],
        overall="fail",
        judge_suggestion=suggestion,
        reviewed_by="human",
    )
    restored = Annotation.from_dict(json.loads(json.dumps(ann.to_dict())))
    assert restored == ann
    assert restored.judge_suggestion == suggestion
    assert restored.reviewed_by == "human"


def test_defect_roundtrip() -> None:
    d = Defect(unit_id="p0001", problem_type="semantic_drift", note="语义偏移", confirmed=True)
    restored = Defect.from_dict(json.loads(json.dumps(d.to_dict())))
    assert restored == d
    assert restored.unit_id == "p0001"
    assert restored.problem_type == "semantic_drift"

    d2 = Defect(unit_id="p0002", problem_type="translation_ese")
    assert d2.confirmed is None
    data = d2.to_dict()
    assert "confirmed" not in data


def test_eval_report_to_json_and_aggregate() -> None:
    good = ProblemTypeStats(problem_type="semantic_boundary", error_count=0, warning_count=0, passed=True)
    bad = ProblemTypeStats(problem_type="empty_unit", error_count=1, warning_count=0, passed=False, evidence=["p0001"])
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
    assert agg["problem_types"] == {
        "semantic_boundary": {"total": 1, "passed": 1},
        "empty_unit": {"total": 1, "passed": 0},
    }

    payload = json.loads(report.to_json())
    assert payload["step"] == "plan"
    assert payload["aggregate"]["n_cases"] == 4
    assert len(payload["cases"]) == 4


def test_eval_report_save(tmp_path) -> None:
    report = EvalReport(step="plan")
    path = report.save(tmp_path / "sub" / "report.json")
    assert json.loads(path.read_text(encoding="utf-8"))["step"] == "plan"
