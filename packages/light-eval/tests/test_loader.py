"""Loader tests — discovery, fixture loading, annotation parsing."""

from __future__ import annotations

import pytest
from light_eval import loader

from .conftest import make_words, plan_fixture_files, translate_fixture_files, write_case

_UNITS = [
    {"unit_id": "u0001", "start": 0.0, "end": 2.0, "source_text": "hello world this is", "speaker": ""},
    {"unit_id": "u0002", "start": 2.0, "end": 4.0, "source_text": "a small test case.", "speaker": ""},
]

_PLAN_META = [
    {"unit_id": "p0000", "start": 0.0, "end": 2.0, "speaker": "", "text": "hello world"},
    {"unit_id": "p0001", "start": 2.0, "end": 4.0, "speaker": "", "text": "test case"},
]


def _write_plan_case(root, name="plan_case", annotation=None):
    words = make_words(["hello", "world", "this", "is", "a", "small", "test", "case."], step=0.5)
    return write_case(
        root,
        "plan",
        name,
        kind="edge",
        params={"min_duration": 0.5},
        fixture_files=plan_fixture_files(words, _UNITS),
        annotation=annotation,
    )


def test_discover_cases_sorted_and_filtered(tmp_path) -> None:
    _write_plan_case(tmp_path, "b_case")
    _write_plan_case(tmp_path, "a_case")
    write_case(
        tmp_path,
        "translate",
        "tx_case",
        fixture_files=translate_fixture_files(_PLAN_META),
    )

    all_cases = loader.discover_cases(tmp_path)
    assert [(c.step, c.name) for c in all_cases] == [
        ("plan", "a_case"),
        ("plan", "b_case"),
        ("translate", "tx_case"),
    ]

    plan_only = loader.discover_cases(tmp_path, step="plan")
    assert [c.name for c in plan_only] == ["a_case", "b_case"]
    assert plan_only[0].kind == "edge"
    assert plan_only[0].params == {"min_duration": 0.5}
    assert plan_only[0].source == "test-fixture"


def test_load_fixture_plan_reassembles_words(tmp_path) -> None:
    case_dir = _write_plan_case(tmp_path)
    case = loader.load_case(case_dir)
    fixture = loader.load_fixture(case)

    assert [s.unit_id for s in fixture.segments] == ["u0001", "u0002"]
    first = fixture.segments[0]
    assert first.source_text == "hello world this is"
    assert [w.text for w in first.words] == ["hello", "world", "this", "is"]
    assert sum(len(s.words) for s in fixture.segments) == 8


def test_load_fixture_translate_with_glossary(tmp_path) -> None:
    files = translate_fixture_files(_PLAN_META)
    files["glossary.json"] = {"Light": "光"}
    files["summary.json"] = {"topic": "testing"}
    case_dir = write_case(tmp_path, "translate", "tx", params={"target_lang": "zh"}, fixture_files=files)

    fixture = loader.load_fixture(loader.load_case(case_dir))
    assert [u.unit_id for u in fixture.segments] == ["p0000", "p0001"]
    assert fixture.segments[0].source_text == "hello world"
    assert fixture.glossary == {"Light": "光"}
    assert fixture.summary == {"topic": "testing"}


def test_load_fixture_translate_optional_sidecars_absent(tmp_path) -> None:
    case_dir = write_case(tmp_path, "translate", "tx", fixture_files=translate_fixture_files(_PLAN_META))
    fixture = loader.load_fixture(loader.load_case(case_dir))
    assert fixture.glossary is None
    assert fixture.summary is None


def test_load_annotation(tmp_path) -> None:
    annotation = {
        "defects": [
            {"unit_id": "u0001", "problem_type": "boundary_quality", "note": "score 4"},
            {"unit_id": "u0002", "problem_type": "readability", "note": "bad split"},
        ],
        "overall": "acceptable",
    }
    case = loader.load_case(_write_plan_case(tmp_path, annotation=annotation))
    parsed = loader.load_annotation(case)
    assert parsed is not None
    assert len(parsed.defects) == 2
    assert parsed.defects[0].problem_type == "boundary_quality"
    assert parsed.defects[0].unit_id == "u0001"
    assert parsed.defects[1].problem_type == "readability"
    assert parsed.defects[1].unit_id == "u0002"
    assert parsed.overall == "acceptable"


def test_load_annotation_absent_returns_none(tmp_path) -> None:
    case = loader.load_case(_write_plan_case(tmp_path))
    assert loader.load_annotation(case) is None


def test_load_case_invalid_step_rejected(tmp_path) -> None:
    case_dir = tmp_path / "weird" / "x"
    (case_dir / "fixture").mkdir(parents=True)
    (case_dir / "case.yaml").write_text("step: weird\nkind: control\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid step"):
        loader.load_case(case_dir)
