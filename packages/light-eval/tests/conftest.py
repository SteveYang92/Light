"""Shared helpers for light-eval tests: fake words/segments and case dirs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from light_models import Segment, Word, word_to_dict


def make_words(texts: list[str], *, start: float = 0.0, step: float = 0.5) -> list[Word]:
    """Build a word timeline: consecutive ``step``-second words from *start*."""
    return [
        Word(text=text, start=start + i * step, end=start + (i + 1) * step, confidence=1.0)
        for i, text in enumerate(texts)
    ]


def make_segment(unit_id: str, words: list[Word], *, speaker: str = "") -> Segment:
    return Segment(
        unit_id=unit_id,
        start=words[0].start,
        end=words[-1].end,
        speaker=speaker,
        source_text=" ".join(w.text for w in words),
        words=words,
    )


def write_case(
    root: Path,
    step: str,
    name: str,
    *,
    kind: str = "control",
    source: str = "test-fixture",
    params: dict | None = None,
    fixture_files: dict[str, object] | None = None,
    annotation: dict | None = None,
) -> Path:
    """Materialize one case dir: case.yaml + fixture/*.json + optional annotation."""
    case_dir = root / step / name
    (case_dir / "fixture").mkdir(parents=True)
    case_yaml = {"step": step, "kind": kind, "source": source, "params": params or {}}
    (case_dir / "case.yaml").write_text(yaml.safe_dump(case_yaml, allow_unicode=True), encoding="utf-8")
    for filename, data in (fixture_files or {}).items():
        (case_dir / "fixture" / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if annotation is not None:
        (case_dir / "annotation.yaml").write_text(yaml.safe_dump(annotation, allow_unicode=True), encoding="utf-8")
    return case_dir


def plan_fixture_files(words: list[Word], units: list[dict]) -> dict[str, object]:
    """plan-step fixture: segment.json (pipeline schema) + words.json timeline."""
    return {
        "segment.json": {"units": units},
        "words.json": [word_to_dict(w) for w in words],
    }


def translate_fixture_files(units: list[dict]) -> dict[str, object]:
    """translate-step fixture: plan.json (pipeline schema, version wrapper)."""
    return {"plan.json": {"version": 1, "units": units}}


@pytest.fixture()
def ten_word_segment() -> Segment:
    """One 10-word, 5-second segment (fits default PlanConfig budgets)."""
    words = make_words(["hello", "world", "this", "is", "a", "small", "test", "case", "for", "planning."])
    return make_segment("u0001", words)
