"""Case discovery and fixture loading.

Directory convention::

    <suite_root>/<step>/<case_name>/
        case.yaml          # step / kind / source / params
        fixture/           # step inputs, pipeline artifact formats
        annotation.yaml    # optional human annotation

Fixture files reuse the pipeline's own artifact schemas so cases can be
harvested verbatim from a real run's ``output/`` directory:

- ``plan``: ``fixture/segment.json`` (pipeline ``segment/segment.json``)
  plus ``fixture/words.json`` (global word timeline, ``word_to_dict`` rows)
  — reassembled via :func:`light_subtitle.artifacts.read_segment_units`.
- ``translate``: ``fixture/plan.json`` (pipeline ``plan/plan.json``) via
  :func:`light_subtitle.artifacts.read_plan_units`, plus optional
  ``fixture/glossary.json`` / ``fixture/summary.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from light_models import Segment, word_from_dict
from light_subtitle import artifacts

from .models import VALID_KINDS, VALID_STEPS, Annotation, EvalCase

CASE_YAML = "case.yaml"
ANNOTATION_YAML = "annotation.yaml"
FIXTURE_DIR = "fixture"


# ── Fixture container ───────────────────────────────────────────────────────


@dataclass
class Fixture:
    """Loaded step inputs for one case.

    ``segments`` holds plan-step input segments (with word timing) or
    translate-step input plan units (wordless, from ``plan.json``).
    """

    segments: list[Segment] = field(default_factory=list)
    glossary: dict[str, str] | None = None
    summary: dict | None = None


# ── Discovery ───────────────────────────────────────────────────────────────


def discover_cases(root: str | Path, *, step: str | None = None) -> list[EvalCase]:
    """Find all cases under ``<root>/<step>/<case_name>/case.yaml``.

    *step* optionally restricts discovery to one step directory.
    Cases are returned sorted by (step, name) for deterministic runs.
    """
    root = Path(root)
    steps = [step] if step else list(VALID_STEPS)
    cases: list[EvalCase] = []
    for step_name in steps:
        step_dir = root / step_name
        if not step_dir.is_dir():
            continue
        for case_yaml in sorted(step_dir.glob(f"*/{CASE_YAML}")):
            cases.append(load_case(case_yaml.parent))
    return sorted(cases, key=lambda c: (c.step, c.name))


def load_case(case_dir: str | Path) -> EvalCase:
    """Parse one ``case.yaml`` into an :class:`EvalCase`."""
    case_dir = Path(case_dir)
    with open(case_dir / CASE_YAML, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    step = str(data.get("step", ""))
    kind = str(data.get("kind", "control"))
    if step not in VALID_STEPS:
        raise ValueError(f"{case_dir}: invalid step {step!r} (expected one of {VALID_STEPS})")
    if kind not in VALID_KINDS:
        raise ValueError(f"{case_dir}: invalid kind {kind!r} (expected one of {VALID_KINDS})")
    return EvalCase(
        name=str(data.get("name", case_dir.name)),
        step=step,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        source=str(data.get("source", "")),
        params=dict(data.get("params") or {}),
        case_dir=case_dir,
    )


# ── Fixture loading ─────────────────────────────────────────────────────────


def load_fixture(case: EvalCase) -> Fixture:
    """Load step inputs for *case* from its ``fixture/`` directory."""
    fixture_dir = case.case_dir / FIXTURE_DIR
    if case.step == "plan":
        return Fixture(segments=_load_plan_fixture(fixture_dir))
    if case.step == "translate":
        return _load_translate_fixture(fixture_dir)
    raise ValueError(f"unsupported step: {case.step}")


def _load_plan_fixture(fixture_dir: Path) -> list[Segment]:
    """Rebuild timed segments from ``segment.json`` + ``words.json``."""
    segment_path = fixture_dir / artifacts.SEGMENT_JSON
    words_path = fixture_dir / "words.json"
    words = []
    if words_path.is_file():
        with open(words_path, encoding="utf-8") as f:
            words = [word_from_dict(w) for w in json.load(f)]
    return artifacts.read_segment_units(segment_path, words)


def _load_translate_fixture(fixture_dir: Path) -> Fixture:
    """Load plan units plus optional glossary / summary sidecars."""
    units = artifacts.read_plan_units(fixture_dir / artifacts.PLAN_JSON)
    return Fixture(
        segments=units,
        glossary=_read_optional_json(fixture_dir / "glossary.json"),
        summary=_read_optional_json(fixture_dir / "summary.json"),
    )


def _read_optional_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Annotation loading ──────────────────────────────────────────────────────


def load_annotation(case: EvalCase) -> Annotation | None:
    """Parse ``annotation.yaml``; None when absent."""
    path = case.case_dir / ANNOTATION_YAML
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return Annotation.from_dict(data)
