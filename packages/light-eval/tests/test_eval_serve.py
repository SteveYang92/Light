"""Serve tests — FastAPI TestClient against the workbench API (no real server)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from light_eval import loader
from light_eval.judges.llm import LLMJudge
from light_eval.models import Defect, ProblemTypeStats
from light_eval.serve import create_app

from .test_eval_harvest import make_flat_run, make_full_run


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    make_flat_run(tmp_path / "output")
    make_full_run(tmp_path / "output")
    app = create_app(suite_dir=tmp_path / "suite", output_dirs=[str(tmp_path / "output")])
    return TestClient(app)


# ── UI + candidates ─────────────────────────────────────────────────────────


def test_index_serves_html(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "light-eval workbench" in res.text


def test_list_candidates(client: TestClient) -> None:
    res = client.get("/api/candidates")
    assert res.status_code == 200
    names = {c["name"] for c in res.json()["candidates"]}
    assert names == {"flat_run_p1", "full_run"}


# ── Case creation / listing ─────────────────────────────────────────────────


def test_create_and_list_case(client: TestClient) -> None:
    res = client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "edge"})
    assert res.status_code == 201
    case = res.json()["case"]
    assert case["id"] == "plan/flat_run_p1"
    assert case["annotated"] is False

    listed = client.get("/api/cases").json()["cases"]
    assert [c["id"] for c in listed] == ["plan/flat_run_p1"]
    assert set(listed[0]["problem_types"].keys()) == {
        "semantic_boundary",
        "over_fragmentation",
        "over_long_unit",
        "dangling_word",
        "empty_unit",
        "flash_unit",
    }


def test_create_case_conflict_and_missing(client: TestClient) -> None:
    body = {"candidate": "flat_run_p1", "step": "plan", "kind": "control"}
    assert client.post("/api/cases", json=body).status_code == 201
    assert client.post("/api/cases", json=body).status_code == 409
    assert client.post("/api/cases", json={"candidate": "nope", "step": "plan", "kind": "control"}).status_code == 404
    bogus_step = {"candidate": "flat_run_p1", "step": "bogus", "kind": "control"}
    assert client.post("/api/cases", json=bogus_step).status_code == 422


# ── Candidate detail + unit scanning ──────────────────────────────────────


def test_candidate_detail_endpoint(client: TestClient) -> None:
    res = client.get("/api/candidates/full_run", params={"step": "translate"})
    assert res.status_code == 200
    data = res.json()
    assert data["step"] == "translate"
    assert [u["unit_id"] for u in data["units"]] == ["p0000", "p0001"]
    assert {"unit_id", "start", "end", "text"} <= set(data["units"][0])

    res = client.get("/api/candidates/full_run", params={"step": "plan"})
    assert res.status_code == 200
    assert [u["unit_id"] for u in res.json()["units"]] == ["u0000"]

    assert client.get("/api/candidates/nope").status_code == 404
    assert client.get("/api/candidates/flat_run_p1", params={"step": "translate"}).status_code == 200
    assert client.get("/api/candidates/flat_run_p1", params={"step": "translate"}).json()["step"] == "plan"


def test_candidate_detail_defaults_to_first_step(client: TestClient) -> None:
    res = client.get("/api/candidates/full_run")
    assert res.status_code == 200
    assert res.json()["step"] in ("plan", "translate")


def test_create_case_with_unit_ids(client: TestClient, tmp_path: Path) -> None:
    body = {
        "candidate": "full_run",
        "step": "translate",
        "kind": "control",
        "unit_ids": ["p0000", "p0001"],
    }
    res = client.post("/api/cases", json=body)
    assert res.status_code == 201
    assert res.json()["case"]["id"] == "translate/full_run__p0000-p0001"

    plan = json.loads(
        (tmp_path / "suite" / "translate" / "full_run__p0000-p0001" / "fixture" / "plan.json").read_text("utf-8")
    )
    assert [u["unit_id"] for u in plan["units"]] == ["p0000", "p0001"]

    # same unit_ids again → numbered suffix
    res = client.post("/api/cases", json=body)
    assert res.status_code == 201
    assert res.json()["case"]["id"] == "translate/full_run__p0000-p0001_2"


def test_create_case_with_custom_name(client: TestClient) -> None:
    body = {
        "candidate": "full_run",
        "step": "translate",
        "kind": "control",
        "unit_ids": ["p0000"],
        "name": "my_custom_case",
    }
    res = client.post("/api/cases", json=body)
    assert res.status_code == 201
    assert res.json()["case"]["id"] == "translate/my_custom_case"
    assert res.json()["case"]["name"] == "my_custom_case"


def test_create_case_invalid_unit_ids_422(client: TestClient) -> None:
    base = {"candidate": "full_run", "step": "translate", "kind": "control"}
    unknown = client.post("/api/cases", json={**base, "unit_ids": ["p9999"]})
    assert unknown.status_code == 422
    empty = client.post("/api/cases", json={**base, "unit_ids": []})
    assert empty.status_code == 422


# ── Run + output ────────────────────────────────────────────────────────────


def test_run_plan_case_and_read_output(client: TestClient) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})

    res = client.post("/api/cases/plan/flat_run_p1/run")
    assert res.status_code == 200
    data = res.json()
    assert data["error"] is None
    assert data["items"], "plan fallback should produce units"
    assert {"unit_id", "start", "end", "text"} <= set(data["items"][0])

    stored = client.get("/api/cases/plan/flat_run_p1/output").json()
    assert stored["has_output"] is True
    assert stored["items"] == data["items"]


def test_output_before_run(client: TestClient) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    data = client.get("/api/cases/plan/flat_run_p1/output").json()
    assert data["has_output"] is False
    assert data["items"] == []


def test_run_translate_without_llm_key(client: TestClient) -> None:
    client.post("/api/cases", json={"candidate": "full_run", "step": "translate", "kind": "control"})
    res = client.post("/api/cases/translate/full_run/run")
    assert res.status_code == 409
    assert "DEEPSEEK_API_KEY" in res.json()["detail"]


def test_unknown_case_404(client: TestClient) -> None:
    assert client.get("/api/cases/plan/ghost/output").status_code == 404
    assert client.post("/api/cases/bogus/x/run").status_code == 404


# ── Annotation ──────────────────────────────────────────────────────────────


def test_annotation_round_trip(client: TestClient, tmp_path: Path) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})

    empty = client.get("/api/cases/plan/flat_run_p1/annotation").json()
    assert empty == {"defects": [], "overall": ""}

    payload = {
        "defects": [{"unit_id": "p0000_0", "problem_type": "semantic_boundary", "note": "切在从句中间"}],
        "overall": "pass",
    }
    assert client.put("/api/cases/plan/flat_run_p1/annotation", json=payload).status_code == 200
    assert client.get("/api/cases/plan/flat_run_p1/annotation").json() == payload

    # written through to annotation.yaml in the loader's schema
    case = loader.load_case(tmp_path / "suite" / "plan" / "flat_run_p1")
    annotation = loader.load_annotation(case)
    assert annotation is not None
    assert annotation.defects == [Defect(unit_id="p0000_0", problem_type="semantic_boundary", note="切在从句中间")]
    assert annotation.overall == "pass"

    listed = client.get("/api/cases").json()["cases"]
    assert listed[0]["annotated"] is True


def test_annotation_validation(client: TestClient) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    bad_type = {"defects": [{"unit_id": "p0000", "problem_type": "bogus"}], "overall": ""}
    assert client.put("/api/cases/plan/flat_run_p1/annotation", json=bad_type).status_code == 422
    bad_overall = {"defects": [], "overall": "maybe"}
    assert client.put("/api/cases/plan/flat_run_p1/annotation", json=bad_overall).status_code == 422


# ── AI 预评（judge 端点）─────────────────────────────────────────────────────


def _preset_scores(self, case, fixture, output):  # noqa: ARG001
    return [
        ProblemTypeStats(
            problem_type="semantic_boundary",
            error_count=0,
            warning_count=1,
            passed=True,
            evidence=["p0000_0"],
            issues=[{"unit_id": "p0000_0", "problem": "断在从句中间"}],
        ),
        ProblemTypeStats(problem_type="over_fragmentation", error_count=1, warning_count=0, passed=False, evidence=[]),
    ]


@pytest.fixture()
def judge_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake the LLM: client exists, LLMJudge.score returns preset DimensionScores."""
    monkeypatch.setattr("light_eval.serve.build_llm_client", lambda: object())
    monkeypatch.setattr(LLMJudge, "score", _preset_scores)


def test_judge_endpoint(client: TestClient, judge_mock: None, tmp_path: Path) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    client.post("/api/cases/plan/flat_run_p1/run")

    res = client.post("/api/cases/plan/flat_run_p1/judge")
    assert res.status_code == 200
    data = res.json()
    assert data["suggested_overall"] == "fail"
    bq = data["problem_types"]["semantic_boundary"]
    assert bq["error_count"] == 0
    assert bq["warning_count"] == 1
    assert bq["passed"] is True
    assert bq["issues"] == [{"unit_id": "p0000_0", "problem": "断在从句中间"}]
    of = data["problem_types"]["over_fragmentation"]
    assert of["error_count"] == 1
    assert of["passed"] is False

    # persisted verbatim to .eval_run/judge.json
    stored = json.loads((tmp_path / "suite" / "plan" / "flat_run_p1" / ".eval_run" / "judge.json").read_text("utf-8"))
    assert stored == data


def test_judge_rerun_overwrites_stale_judge_json(client: TestClient, judge_mock: None, tmp_path: Path) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    client.post("/api/cases/plan/flat_run_p1/run")
    assert client.post("/api/cases/plan/flat_run_p1/judge").status_code == 200
    judge_path = tmp_path / "suite" / "plan" / "flat_run_p1" / ".eval_run" / "judge.json"
    judge_path.write_text('{"stale": true}', encoding="utf-8")
    res = client.post("/api/cases/plan/flat_run_p1/judge")
    assert res.status_code == 200
    assert json.loads(judge_path.read_text("utf-8")) == res.json()  # overwritten, not stale


def test_judge_failure_leaves_no_stale_judge_json(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 重跑先删旧评审：judge 失败（输出不可用）时不得残留过期结果
    monkeypatch.setattr("light_eval.serve.build_llm_client", lambda: object())
    monkeypatch.setattr(LLMJudge, "score", lambda *a, **k: [])
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    client.post("/api/cases/plan/flat_run_p1/run")
    judge_path = tmp_path / "suite" / "plan" / "flat_run_p1" / ".eval_run" / "judge.json"
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path.write_text('{"stale": true}', encoding="utf-8")
    res = client.post("/api/cases/plan/flat_run_p1/judge")
    assert res.status_code == 409
    assert not judge_path.exists()


def _save_annotation(client: TestClient) -> None:
    res = client.put(
        "/api/cases/plan/flat_run_p1/annotation",
        json={
            "defects": [{"unit_id": "p0000_0", "problem_type": "semantic_boundary", "note": "旧缺陷"}],
            "overall": "pass",
        },
    )
    assert res.status_code == 200


def test_judge_rerun_deletes_saved_annotation(client: TestClient, judge_mock: None, tmp_path: Path) -> None:
    # 重跑预评成功 → 旧标注删除，本轮从干净状态开始
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    client.post("/api/cases/plan/flat_run_p1/run")
    _save_annotation(client)
    annotation_path = tmp_path / "suite" / "plan" / "flat_run_p1" / "annotation.yaml"
    assert annotation_path.is_file()
    res = client.post("/api/cases/plan/flat_run_p1/judge")
    assert res.status_code == 200
    assert not annotation_path.exists()


def test_judge_failure_keeps_saved_annotation(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 重跑预评失败 → 旧标注保留（不丢人工成果）
    monkeypatch.setattr("light_eval.serve.build_llm_client", lambda: object())
    monkeypatch.setattr(LLMJudge, "score", lambda *a, **k: [])
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    client.post("/api/cases/plan/flat_run_p1/run")
    _save_annotation(client)
    annotation_path = tmp_path / "suite" / "plan" / "flat_run_p1" / "annotation.yaml"
    res = client.post("/api/cases/plan/flat_run_p1/judge")
    assert res.status_code == 409
    assert annotation_path.is_file()


def test_judge_without_output_409(client: TestClient, judge_mock: None) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    res = client.post("/api/cases/plan/flat_run_p1/judge")
    assert res.status_code == 409
    assert "运行" in res.json()["detail"]


def test_judge_without_llm_key_409(client: TestClient) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    client.post("/api/cases/plan/flat_run_p1/run")
    res = client.post("/api/cases/plan/flat_run_p1/judge")
    assert res.status_code == 409
    assert "DEEPSEEK_API_KEY" in res.json()["detail"]


def test_annotation_with_judge_suggestion_round_trip(client: TestClient, tmp_path: Path) -> None:
    client.post("/api/cases", json={"candidate": "flat_run_p1", "step": "plan", "kind": "control"})
    suggestion = {
        "problem_types": {"semantic_boundary": {"error_count": 0, "warning_count": 0, "passed": True, "issues": []}},
        "suggested_overall": "pass",
    }
    payload = {
        "defects": [],
        "overall": "pass",
        "judge_suggestion": suggestion,
        "reviewed_by": "human",
    }
    assert client.put("/api/cases/plan/flat_run_p1/annotation", json=payload).status_code == 200
    assert client.get("/api/cases/plan/flat_run_p1/annotation").json() == payload

    # written through to annotation.yaml in the loader's schema
    case = loader.load_case(tmp_path / "suite" / "plan" / "flat_run_p1")
    annotation = loader.load_annotation(case)
    assert annotation is not None
    assert annotation.judge_suggestion == suggestion
    assert annotation.reviewed_by == "human"

    listed = client.get("/api/cases").json()["cases"]
    assert listed[0]["has_judge_suggestion"] is True
