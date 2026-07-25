"""CLI smoke tests — typer CliRunner, no network (LLM disabled)."""

from __future__ import annotations

import json

import pytest
from light_eval.cli import app
from light_eval.judges.llm import LLMJudge
from light_eval.models import StepOutput
from typer.testing import CliRunner

from .conftest import make_words, plan_fixture_files, translate_fixture_files, write_case

runner = CliRunner()

_WORDS = ["hello", "world", "this", "is", "a", "small", "test", "case", "for", "planning."]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def _write_plan_case(root) -> None:
    words = make_words(_WORDS)
    units = [{"unit_id": "u0001", "start": 0.0, "end": 5.0, "source_text": " ".join(_WORDS), "speaker": ""}]
    write_case(root, "plan", "basic", fixture_files=plan_fixture_files(words, units))


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "harvest", "calibrate", "serve"):
        assert cmd in result.stdout


def test_run_plan_suite_json(tmp_path) -> None:
    _write_plan_case(tmp_path)
    report_path = tmp_path / "report.json"
    result = runner.invoke(app, ["run", str(tmp_path), "--step", "plan", "-o", str(report_path)])
    assert result.exit_code == 0, result.stdout
    assert "[PASS] plan/basic" in result.stdout

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["step"] == "plan"
    assert report["aggregate"]["n_cases"] == 1
    assert report["aggregate"]["dimensions"]["word_coverage"] == {"total": 1, "passed": 1}


def test_run_translate_skipped_without_llm(tmp_path) -> None:
    meta = [{"unit_id": "p0000", "start": 0.0, "end": 2.0, "speaker": "", "text": "hello world"}]
    write_case(tmp_path, "translate", "tx", fixture_files=translate_fixture_files(meta))
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "[SKIPPED] translate/tx" in result.stdout


def test_run_html_report(tmp_path) -> None:
    _write_plan_case(tmp_path)
    report_path = tmp_path / "report.html"
    result = runner.invoke(app, ["run", str(tmp_path), "--step", "plan", "-f", "html", "-o", str(report_path)])
    assert result.exit_code == 0, result.stdout
    page = report_path.read_text(encoding="utf-8")
    assert "word_coverage" in page and "PASS" in page


def test_run_empty_suite_fails(tmp_path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1
    assert "No cases found" in result.stdout


def test_calibrate_requires_api_key(tmp_path) -> None:
    """Annotated case without a stored suggestion must be re-judged live → key required."""
    _write_annotated_plan_case(tmp_path)
    result = runner.invoke(app, ["calibrate", str(tmp_path)])
    assert result.exit_code == 1
    assert "requires an LLM API key" in result.stdout


def _write_annotated_plan_case(root, *, annotation: dict | None = None):
    """Plan case with annotation + persisted .eval_run/output.json (no real run)."""
    words = make_words(_WORDS)
    units = [{"unit_id": "u0001", "start": 0.0, "end": 5.0, "source_text": " ".join(_WORDS), "speaker": ""}]
    if annotation is None:
        annotation = {"dimensions": {"boundary_quality": 4}, "defects": [], "overall": "pass"}
    case_dir = write_case(root, "plan", "basic", fixture_files=plan_fixture_files(words, units), annotation=annotation)
    run_dir = case_dir / ".eval_run"
    run_dir.mkdir()
    output = StepOutput(case="basic", output=[{"unit_id": "u0001", "start": 0.0, "end": 5.0, "text": " ".join(_WORDS)}])
    (run_dir / "output.json").write_text(json.dumps(output.to_dict(), ensure_ascii=False), encoding="utf-8")
    return case_dir


def test_calibrate_prefers_stored_suggestion_without_llm(tmp_path, monkeypatch) -> None:
    """judge_suggestion in annotation.yaml is used verbatim — no LLM key or call needed."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(LLMJudge, "score", lambda *a, **k: pytest.fail("stored suggestion should skip the LLM judge"))
    suggestion = {
        "dimensions": {
            "boundary_quality": {"score": 4, "reason": "边界合理", "evidence": []},
            "split_necessity": {"score": 5, "reason": "拆分恰当", "evidence": []},
        },
        "suggested_overall": "pass",
    }
    annotation = {
        "dimensions": {"boundary_quality": 3, "split_necessity": 5},  # human adjusted one dim 4→3
        "defects": [],
        "overall": "borderline",
        "judge_suggestion": suggestion,
        "reviewed_by": "human",
    }
    _write_annotated_plan_case(tmp_path, annotation=annotation)

    report_path = tmp_path / "cal.json"
    result = runner.invoke(app, ["calibrate", str(tmp_path), "-o", str(report_path)])
    assert result.exit_code == 0, result.stdout
    assert "[SUGGESTION] plan/basic" in result.stdout
    assert "人工调整率: 1/2" in result.stdout  # boundary_quality 4 → 3

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_cases"] == 1
    assert report["n_pairs"] == 2
    dims = {d["dimension"]: d for d in report["dimensions"]}
    assert dims["boundary_quality"]["mae"] == 1.0
    assert dims["split_necessity"]["mae"] == 0.0


def test_harvest_table_and_json(tmp_path) -> None:
    transcript = {
        "format": "light-transcript.v1",
        "language": "en",
        "words": [{"text": t, "start": i * 0.5, "end": (i + 1) * 0.5} for i, t in enumerate(_WORDS)],
    }
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "run_a_p1.transcript.json").write_text(json.dumps(transcript), encoding="utf-8")

    result = runner.invoke(app, ["harvest", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "run_a_p1" in result.stdout and "plan" in result.stdout

    result = runner.invoke(app, ["harvest", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.stdout
    candidates = json.loads(result.stdout)
    assert candidates[0]["name"] == "run_a_p1"
    assert candidates[0]["steps"] == ["plan"]


def test_harvest_empty_dir(tmp_path) -> None:
    result = runner.invoke(app, ["harvest", str(tmp_path)])
    assert result.exit_code == 0
    assert "No candidates" in result.stdout


def test_serve_help() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    for opt in ("--suite-dir", "--output-dir", "--port"):
        assert opt in result.stdout
