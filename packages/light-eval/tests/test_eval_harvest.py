"""Harvest tests — fake run dirs, candidate scan, and create_case round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from light_eval import loader
from light_eval.harvest import create_case, format_duration, format_table, scan_candidate_units, scan_candidates
from light_models import word_to_dict

from .conftest import make_words

_WORDS = ["hello", "world", "this", "is", "a", "small", "test", "case", "for", "planning."]


def _transcript_payload() -> dict:
    return {
        "format": "light-transcript.v1",
        "source": "test",
        "language": "en",
        "words": [word_to_dict(w) for w in make_words(_WORDS)],
        "segments": [],
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_flat_run(root: Path, name: str = "flat_run") -> Path:
    """Flat layout: renamed transcript sidecar only, no artifact subdirs."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    _write_json(run_dir / f"{name}_p1.transcript.json", _transcript_payload())
    (run_dir / f"{name}_p1.zh.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    return run_dir


def make_full_run(root: Path, name: str = "full_run") -> Path:
    """Full layout: canonical transcript + segment/ + plan/ + context/."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "transcript.json", _transcript_payload())
    # slug-named duplicate sidecar — must not become a second candidate
    _write_json(run_dir / f"{name}.transcript.json", _transcript_payload())
    _write_json(
        run_dir / "segment" / "segment.json",
        {
            "total_words": len(_WORDS),
            "total_units": 1,
            "words": [word_to_dict(w) for w in make_words(_WORDS)],
            "units": [
                {
                    "unit_id": "u0000",
                    "start": 0.0,
                    "end": 5.0,
                    "duration": 5.0,
                    "speaker": "",
                    "word_count": len(_WORDS),
                    "source_text": " ".join(_WORDS),
                    "word_range": {"from": "hello", "to": "planning."},
                }
            ],
        },
    )
    _write_json(
        run_dir / "plan" / "plan.json",
        {
            "version": 1,
            "units": [
                {"unit_id": "p0000", "start": 0.0, "end": 2.5, "speaker": "", "text": "hello world this is a"},
                {"unit_id": "p0001", "start": 2.5, "end": 5.0, "speaker": "", "text": "small test case for planning."},
            ],
        },
    )
    _write_json(run_dir / "context" / "glossary.json", {"Light": "光"})
    _write_json(run_dir / "context" / "summary.json", {"topic": "test"})
    (run_dir / f"{name}.zh.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    return run_dir


_MULTI_WORDS = [f"w{i:02d}" for i in range(12)]  # 12 × 0.5s → 0–6s


def make_multi_run(root: Path, name: str = "multi_run") -> Path:
    """Full-layout run with 3 segment units (u0000-u0002) and 3 plan units (p0000-p0002)."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    words = make_words(_MULTI_WORDS)
    _write_json(
        run_dir / "transcript.json",
        {
            "format": "light-transcript.v1",
            "source": "test",
            "language": "en",
            "words": [word_to_dict(w) for w in words],
            "segments": [],
        },
    )
    seg_units = []
    for k in range(3):
        chunk = words[k * 4 : (k + 1) * 4]
        seg_units.append(
            {
                "unit_id": f"u{k:04d}",
                "start": chunk[0].start,
                "end": chunk[-1].end,
                "duration": round(chunk[-1].end - chunk[0].start, 3),
                "speaker": "",
                "word_count": len(chunk),
                "source_text": " ".join(w.text for w in chunk),
                "word_range": {"from": chunk[0].text, "to": chunk[-1].text},
            }
        )
    _write_json(
        run_dir / "segment" / "segment.json",
        {
            "total_words": len(words),
            "total_units": 3,
            "words": [word_to_dict(w) for w in words],
            "units": seg_units,
        },
    )
    _write_json(
        run_dir / "plan" / "plan.json",
        {
            "version": 1,
            "units": [
                {
                    "unit_id": f"p{k:04d}",
                    "start": k * 2.0,
                    "end": (k + 1) * 2.0,
                    "speaker": "",
                    "text": " ".join(_MULTI_WORDS[k * 4 : (k + 1) * 4]),
                }
                for k in range(3)
            ],
        },
    )
    _write_json(run_dir / "context" / "glossary.json", {"Light": "光"})
    (run_dir / f"{name}.zh.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    return run_dir


# ── Scanning ────────────────────────────────────────────────────────────────


def test_scan_finds_plan_and_translate_candidates(tmp_path: Path) -> None:
    make_flat_run(tmp_path)
    make_full_run(tmp_path)
    candidates = {c.name: c for c in scan_candidates(tmp_path)}

    assert set(candidates) == {"flat_run_p1", "full_run"}

    flat = candidates["flat_run_p1"]
    assert flat.steps == ["plan"]
    assert flat.n_words == len(_WORDS)
    assert flat.duration_s == pytest.approx(5.0)
    assert flat.source_lang == "en"
    assert flat.target_langs == ["zh"]
    assert flat.lang_pair == "en→zh"

    full = candidates["full_run"]
    assert full.steps == ["plan", "translate"]
    assert full.n_words == len(_WORDS)
    assert full.n_units == 2


def test_scan_run_dir_directly(tmp_path: Path) -> None:
    run_dir = make_flat_run(tmp_path)
    candidates = scan_candidates(run_dir)
    assert [c.name for c in candidates] == ["flat_run_p1"]


def test_scan_ignores_invalid_transcripts(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()
    _write_json(run_dir / "bad_p1.transcript.json", {"format": "other", "words": []})
    _write_json(run_dir / "broken_p1.transcript.json", {"format": "light-transcript.v1"})  # no words
    assert scan_candidates(tmp_path) == []


def test_scan_translate_only_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "plan_only"
    _write_json(
        run_dir / "plan" / "plan.json",
        {"version": 1, "units": [{"unit_id": "p0000", "start": 0.0, "end": 9.0, "text": "x"}]},
    )
    candidates = scan_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0].steps == ["translate"]
    assert candidates[0].n_units == 1
    assert candidates[0].duration_s == pytest.approx(9.0)


# ── create_case ─────────────────────────────────────────────────────────────


def test_create_plan_case_from_flat_run(tmp_path: Path) -> None:
    """Flat run: segments are regenerated on the fly; loader reads it back."""
    make_flat_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    case_dir = create_case(candidate, "plan", "edge", tmp_path / "suite")

    case = loader.load_case(case_dir)
    assert case.step == "plan" and case.kind == "edge"
    assert case.source == str(tmp_path / "flat_run")

    fixture = loader.load_fixture(case)
    assert fixture.segments
    assert sum(len(seg.words) for seg in fixture.segments) == len(_WORDS)


def test_create_plan_case_from_full_run_reuses_segment_json(tmp_path: Path) -> None:
    make_full_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    case_dir = create_case(candidate, "plan", "control", tmp_path / "suite")

    copied = json.loads((case_dir / "fixture" / "segment.json").read_text(encoding="utf-8"))
    assert copied["units"][0]["unit_id"] == "u0000"  # verbatim copy, not regenerated

    fixture = loader.load_fixture(loader.load_case(case_dir))
    assert len(fixture.segments) == 1
    assert len(fixture.segments[0].words) == len(_WORDS)


def test_create_translate_case(tmp_path: Path) -> None:
    make_full_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    case_dir = create_case(candidate, "translate", "boundary", tmp_path / "suite")

    case = loader.load_case(case_dir)
    assert case.params["target_lang"] == "zh"

    fixture = loader.load_fixture(case)
    assert [seg.unit_id for seg in fixture.segments] == ["p0000", "p0001"]
    assert fixture.glossary == {"Light": "光"}
    assert fixture.summary == {"topic": "test"}


def test_create_case_rejects_missing_material(tmp_path: Path) -> None:
    make_flat_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    with pytest.raises(ValueError, match="no material"):
        create_case(candidate, "translate", "control", tmp_path / "suite")


def test_create_case_rejects_duplicate(tmp_path: Path) -> None:
    make_flat_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    create_case(candidate, "plan", "control", tmp_path / "suite")
    with pytest.raises(FileExistsError):
        create_case(candidate, "plan", "control", tmp_path / "suite")


# ── scan_candidate_units ────────────────────────────────────────────────────


def test_scan_candidate_units_translate(tmp_path: Path) -> None:
    make_multi_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    units = scan_candidate_units(candidate, "translate")
    assert [u["unit_id"] for u in units] == ["p0000", "p0001", "p0002"]
    assert units[1]["start"] == pytest.approx(2.0)
    assert units[1]["end"] == pytest.approx(4.0)
    assert units[1]["text"] == "w04 w05 w06 w07"


def test_scan_candidate_units_plan_full_layout(tmp_path: Path) -> None:
    make_multi_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    units = scan_candidate_units(candidate, "plan")
    assert [u["unit_id"] for u in units] == ["u0000", "u0001", "u0002"]
    assert units[0]["text"] == "w00 w01 w02 w03"


def test_scan_candidate_units_plan_flat_layout(tmp_path: Path) -> None:
    """Flat run: segments regenerated from the word timeline on the fly."""
    make_flat_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    units = scan_candidate_units(candidate, "plan")
    assert units
    assert all({"unit_id", "start", "end", "text"} <= set(u) for u in units)


def test_scan_candidate_units_rejects_missing_material(tmp_path: Path) -> None:
    make_flat_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    with pytest.raises(ValueError, match="no material"):
        scan_candidate_units(candidate, "translate")


# ── create_case with unit range ─────────────────────────────────────────────


def test_create_translate_case_with_range(tmp_path: Path) -> None:
    make_multi_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    case_dir = create_case(candidate, "translate", "control", tmp_path / "suite", start_unit="p0001", end_unit="p0002")

    assert case_dir.name == "multi_run__p0001-p0002"
    case_yaml = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
    assert case_yaml["range"] == {"start_unit": "p0001", "end_unit": "p0002"}

    plan = json.loads((case_dir / "fixture" / "plan.json").read_text(encoding="utf-8"))
    assert plan["version"] == 1  # schema preserved
    assert [u["unit_id"] for u in plan["units"]] == ["p0001", "p0002"]
    assert (case_dir / "fixture" / "glossary.json").is_file()  # sidecar copied whole

    fixture = loader.load_fixture(loader.load_case(case_dir))
    assert [seg.unit_id for seg in fixture.segments] == ["p0001", "p0002"]


def test_create_plan_case_with_range(tmp_path: Path) -> None:
    make_multi_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    case_dir = create_case(candidate, "plan", "edge", tmp_path / "suite", start_unit="u0000", end_unit="u0001")

    segment = json.loads((case_dir / "fixture" / "segment.json").read_text(encoding="utf-8"))
    assert [u["unit_id"] for u in segment["units"]] == ["u0000", "u0001"]
    assert segment["total_units"] == 2
    assert segment["total_words"] == 8
    assert len(segment["words"]) == 8  # only words covered by u0000-u0001

    words = json.loads((case_dir / "fixture" / "words.json").read_text(encoding="utf-8"))
    assert len(words) == 8

    fixture = loader.load_fixture(loader.load_case(case_dir))
    assert [seg.unit_id for seg in fixture.segments] == ["u0000", "u0001"]
    assert sum(len(seg.words) for seg in fixture.segments) == 8


def test_create_plan_case_with_range_flat_run(tmp_path: Path) -> None:
    """Flat run: segments regenerated, then filtered to the range."""
    make_flat_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    units = scan_candidate_units(candidate, "plan")
    assert len(units) >= 1
    case_dir = create_case(
        candidate, "plan", "control", tmp_path / "suite", start_unit=units[0]["unit_id"], end_unit=units[0]["unit_id"]
    )
    fixture = loader.load_fixture(loader.load_case(case_dir))
    assert [seg.unit_id for seg in fixture.segments] == [units[0]["unit_id"]]


def test_create_case_range_names_unique_on_repeat(tmp_path: Path) -> None:
    """Same range twice → second case gets a numbered suffix (multi-segment harvesting)."""
    make_multi_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    first = create_case(candidate, "translate", "control", tmp_path / "suite", start_unit="p0000", end_unit="p0001")
    second = create_case(candidate, "translate", "control", tmp_path / "suite", start_unit="p0000", end_unit="p0001")
    assert first.name == "multi_run__p0000-p0001"
    assert second.name == "multi_run__p0000-p0001_2"


def test_create_case_rejects_invalid_range(tmp_path: Path) -> None:
    make_multi_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    suite = tmp_path / "suite"
    with pytest.raises(ValueError, match="unknown start_unit"):
        create_case(candidate, "translate", "control", suite, start_unit="p9999", end_unit="p0001")
    with pytest.raises(ValueError, match="comes before"):
        create_case(candidate, "translate", "control", suite, start_unit="p0002", end_unit="p0000")
    with pytest.raises(ValueError, match="together"):
        create_case(candidate, "translate", "control", suite, start_unit="p0000")
    # failed attempts leave no case dirs behind
    assert not (suite / "translate").exists() or not list((suite / "translate").iterdir())


def test_create_case_without_range_keeps_whole_video(tmp_path: Path) -> None:
    make_multi_run(tmp_path)
    candidate = scan_candidates(tmp_path)[0]
    case_dir = create_case(candidate, "translate", "control", tmp_path / "suite")
    case_yaml = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
    assert "range" not in case_yaml
    plan = json.loads((case_dir / "fixture" / "plan.json").read_text(encoding="utf-8"))
    assert len(plan["units"]) == 3


# ── Console rendering ───────────────────────────────────────────────────────


def test_format_helpers(tmp_path: Path) -> None:
    make_flat_run(tmp_path)
    table = format_table(scan_candidates(tmp_path))
    assert "NAME" in table and "flat_run_p1" in table
    assert format_duration(65.0) == "01:05"
    assert format_duration(3661.0) == "1:01:01"
