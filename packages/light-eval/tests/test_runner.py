"""Runner tests — real capability-package calls, no mocks."""

from __future__ import annotations

import json

from light_eval.loader import Fixture
from light_eval.models import EvalCase
from light_eval.runner import run_case


def _case(step: str, tmp_path, params: dict | None = None) -> EvalCase:
    return EvalCase(name="case1", step=step, kind="control", source="test", params=params or {}, case_dir=tmp_path)


def test_run_plan_without_llm_uses_fallback(ten_word_segment, tmp_path) -> None:
    fixture = Fixture(segments=[ten_word_segment])
    output = run_case(_case("plan", tmp_path), fixture, llm=None, work_dir=tmp_path / "runs")

    assert output.error is None
    assert not output.skipped
    assert output.duration_s > 0
    assert len(output.output) == 1  # one group, under max_duration → single unit

    unit = output.output[0]
    assert unit["unit_id"] == "p0000"
    assert unit["text"] == ten_word_segment.source_text
    assert len(unit["words"]) == 10  # full word timing kept for rule metrics

    plan_json = tmp_path / "runs" / "case1" / "plan" / "plan.json"
    assert plan_json.is_file()
    assert json.loads(plan_json.read_text(encoding="utf-8"))["units"][0]["unit_id"] == "p0000"


def test_run_plan_error_is_captured(ten_word_segment, tmp_path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")  # plan_dir mkdir must fail
    fixture = Fixture(segments=[ten_word_segment])
    output = run_case(_case("plan", tmp_path), fixture, llm=None, work_dir=blocker)
    assert output.error is not None  # exception captured, not raised
    assert output.output == []


def test_run_translate_without_llm_is_skipped(tmp_path) -> None:
    fixture = Fixture(segments=[])
    output = run_case(_case("translate", tmp_path, {"target_lang": "zh"}), fixture, llm=None)
    assert output.skipped is True
    assert output.error is None
    assert output.output == []
