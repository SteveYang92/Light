"""CLI smoke tests — typer CliRunner, no network (LLM disabled)."""

from __future__ import annotations

import json

import pytest
from light_eval.cli import app
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
    for cmd in ("run", "harvest", "serve"):
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
    assert report["aggregate"]["problem_types"]["word_coverage"] == {"total": 1, "passed": 1}


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
