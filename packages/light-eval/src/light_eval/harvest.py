"""Harvest eval candidates from real pipeline runs.

Scans a run-output directory (default ``output/``) and identifies
material usable as eval fixtures:

- ``plan`` candidates: any word-level transcript (``transcript.json`` or
  ``*.transcript.json``, format ``light-transcript.v1`` with words).
- ``translate`` candidates: runs with ``plan/plan.json`` (plus optional
  ``context/glossary.json`` / ``context/summary.json``).

Runs come in two layouts, both supported:

- *flat*: renamed sidecars only (``xxx_p1.transcript.json``, ``.srt`` …)
  — plan candidates only, one per transcript part.
- *full*: canonical ``transcript.json`` plus artifact subdirs
  (``segment/``, ``plan/``, ``context/`` …) — plan and translate.

:func:`create_case` materializes a candidate into the case-suite layout
consumed by :mod:`light_eval.loader`.  A case covers either the whole video
or a contiguous unit range (``start_unit`` … ``end_unit``, inclusive), so
one video can yield several segment cases.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from light_models import Segment, word_from_dict, word_to_dict
from light_subtitle import artifacts
from light_subtitle import segment as segment_step
from light_subtitle.export.transcript import export_segments

from .loader import CASE_YAML, FIXTURE_DIR
from .models import VALID_KINDS, VALID_STEPS

TRANSCRIPT_FORMAT = "light-transcript.v1"
_LANG_SIDECAR_RE = re.compile(r"\.([a-z]{2}(?:-[a-zA-Z]{2,})?)\.srt$")


# ── Candidate model ─────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """One harvestable fixture source inside a run directory."""

    name: str  # transcript stem (parts) or run dir name (canonical)
    run: str  # run directory name
    run_dir: Path
    transcript: Path | None = None  # plan-step source (None for translate-only runs)
    duration_s: float = 0.0
    steps: list[str] = field(default_factory=list)  # subset of VALID_STEPS
    source_lang: str = ""
    target_langs: list[str] = field(default_factory=list)
    n_words: int = 0
    n_units: int = 0  # plan.json unit count (translate input scale)

    @property
    def lang_pair(self) -> str:
        """Human-readable language pair, e.g. ``en→zh`` (``en→?`` when unknown)."""
        src = self.source_lang or "?"
        tgt = ",".join(self.target_langs) if self.target_langs else "?"
        return f"{src}→{tgt}"

    @property
    def plan_json(self) -> Path:
        """``plan/plan.json`` inside the run dir (may not exist)."""
        return self.run_dir / "plan" / artifacts.PLAN_JSON

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run": self.run,
            "run_dir": str(self.run_dir),
            "transcript": str(self.transcript) if self.transcript else None,
            "duration_s": round(self.duration_s, 3),
            "steps": self.steps,
            "source_lang": self.source_lang,
            "target_langs": self.target_langs,
            "lang_pair": self.lang_pair,
            "n_words": self.n_words,
            "n_units": self.n_units,
        }


# ── Scanning ────────────────────────────────────────────────────────────────


def scan_candidates(root: str | Path) -> list[Candidate]:
    """Scan *root* for harvestable candidates, sorted by (run, name)."""
    root = Path(root)
    candidates: list[Candidate] = []
    for run_dir in _iter_run_dirs(root):
        candidates.extend(_scan_run_dir(run_dir))
    return sorted(candidates, key=lambda c: (c.run, c.name))


def _iter_run_dirs(root: Path) -> list[Path]:
    """Run directories under *root*; *root* itself when it is one run."""
    if _is_run_dir(root):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and _is_run_dir(p))


def _is_run_dir(path: Path) -> bool:
    """A run dir holds at least one transcript or a ``plan/plan.json``."""
    if (path / "plan" / artifacts.PLAN_JSON).is_file():
        return True
    return any(path.glob("*.transcript.json")) or (path / "transcript.json").is_file()


def _scan_run_dir(run_dir: Path) -> list[Candidate]:
    plan_json = run_dir / "plan" / artifacts.PLAN_JSON
    has_translate = plan_json.is_file()
    n_units = _count_plan_units(plan_json) if has_translate else 0
    candidates = []
    for transcript in _find_transcripts(run_dir):
        data = _read_transcript(transcript)
        if data is None:
            continue
        words = data["words"]
        candidates.append(
            Candidate(
                name=_transcript_name(run_dir, transcript),
                run=run_dir.name,
                run_dir=run_dir,
                transcript=transcript,
                duration_s=max((float(w.get("end", 0.0)) for w in words), default=0.0),
                steps=["plan"] + (["translate"] if has_translate else []),
                source_lang=str(data.get("language") or ""),
                target_langs=_target_langs(run_dir, exclude=str(data.get("language") or "")),
                n_words=len(words),
                n_units=n_units,
            )
        )
    if not candidates and has_translate:
        # translate-only run: no usable transcript, but plan.json exists
        candidates.append(
            Candidate(
                name=run_dir.name,
                run=run_dir.name,
                run_dir=run_dir,
                duration_s=_plan_duration(plan_json),
                steps=["translate"],
                target_langs=_target_langs(run_dir),
                n_units=n_units,
            )
        )
    return candidates


def _find_transcripts(run_dir: Path) -> list[Path]:
    """Transcript files to harvest from one run dir.

    The canonical ``transcript.json`` wins when present (full-layout runs
    also keep a slug-named duplicate sidecar); otherwise every
    ``*.transcript.json`` part is its own candidate.
    """
    canonical = run_dir / "transcript.json"
    if canonical.is_file() and _read_transcript(canonical) is not None:
        return [canonical]
    return sorted(run_dir.glob("*.transcript.json"))


def _read_transcript(path: Path) -> dict | None:
    """Parsed transcript JSON; None unless usable ``light-transcript.v1``."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("format") != TRANSCRIPT_FORMAT:
        return None
    words = data.get("words")
    if not isinstance(words, list) or not words:
        return None
    return data


def _transcript_name(run_dir: Path, transcript: Path) -> str:
    if transcript.name == "transcript.json":
        return run_dir.name
    return transcript.name[: -len(".transcript.json")]


def _target_langs(run_dir: Path, *, exclude: str = "") -> list[str]:
    """Target languages inferred from ``*.<lang>.srt`` sidecar names.

    *exclude* drops the source language itself (runs often keep an
    ``en.srt`` source-language sidecar next to the translated ones).
    """
    langs = set()
    for srt in run_dir.glob("*.srt"):
        match = _LANG_SIDECAR_RE.search(srt.name)
        if match and match.group(1) != exclude:
            langs.add(match.group(1))
    return sorted(langs)


def _count_plan_units(plan_json: Path) -> int:
    try:
        with open(plan_json, encoding="utf-8") as f:
            return len(json.load(f).get("units", []))
    except (OSError, json.JSONDecodeError):
        return 0


def _plan_duration(plan_json: Path) -> float:
    try:
        with open(plan_json, encoding="utf-8") as f:
            units = json.load(f).get("units", [])
    except (OSError, json.JSONDecodeError):
        return 0.0
    return max((float(u.get("end", 0.0)) for u in units), default=0.0)


# ── Unit scanning (range picking) ───────────────────────────────────────────


def scan_candidate_units(candidate: Candidate, step: str) -> list[dict[str, Any]]:
    """Unit sequence of *candidate* for *step*: ``[{unit_id, start, end, text}]``.

    ``translate`` reads the run's ``plan/plan.json``; ``plan`` reuses the
    pipeline's ``segment/segment.json`` (full layout) or regenerates segments
    from the word timeline (flat layout), same source as :func:`create_case`.
    """
    if step not in VALID_STEPS:
        raise ValueError(f"invalid step {step!r} (expected one of {VALID_STEPS})")
    if step not in candidate.steps:
        raise ValueError(f"candidate {candidate.name!r} has no material for step {step!r}")
    if step == "translate":
        if not candidate.plan_json.is_file():
            raise ValueError(f"candidate {candidate.name!r} has no plan/plan.json")
        with open(candidate.plan_json, encoding="utf-8") as f:
            data = json.load(f)
        return [
            {
                "unit_id": str(u.get("unit_id", "")),
                "start": u.get("start", 0.0),
                "end": u.get("end", 0.0),
                "text": u.get("text", ""),
            }
            for u in data.get("units", [])
        ]
    return [
        {"unit_id": s.unit_id, "start": s.start, "end": s.end, "text": s.source_text}
        for s in _candidate_segments(candidate)
    ]


def _candidate_segments(candidate: Candidate) -> list[Segment]:
    """Plan-step segments: pipeline ``segment.json`` when present, else regenerated."""
    if candidate.transcript is None:
        raise ValueError(f"candidate {candidate.name!r} has no transcript")
    data = _read_transcript(candidate.transcript)
    if data is None:
        raise ValueError(f"unusable transcript: {candidate.transcript}")
    words = [word_from_dict(w) for w in data["words"]]
    seg_json = candidate.run_dir / "segment" / artifacts.SEGMENT_JSON
    if candidate.transcript.name == "transcript.json" and seg_json.is_file():
        return artifacts.read_segment_units(seg_json, words)
    return segment_step.run(words)


# ── Case creation ───────────────────────────────────────────────────────────


def create_case(
    candidate: Candidate,
    step: str,
    kind: str,
    dest_root: str | Path,
    *,
    name: str | None = None,
    unit_ids: list[str] | None = None,
) -> Path:
    """Materialize *candidate* as one case under ``<dest_root>/<step>/<name>/``.

    With *unit_ids* the fixture keeps only those units (in original order),
    the case is named ``<video>__<first>-<last>`` (numbered on collision),
    and ``case.yaml`` records the selection.  Without *unit_ids* the whole
    video is used.

    Returns the created case directory; raises ``FileExistsError`` when it
    already exists (whole-video / explicit-name cases), and ``ValueError``
    when the candidate lacks the material *step* needs or any unit_id is
    unknown.
    """
    if step not in VALID_STEPS:
        raise ValueError(f"invalid step {step!r} (expected one of {VALID_STEPS})")
    if kind not in VALID_KINDS:
        raise ValueError(f"invalid kind {kind!r} (expected one of {VALID_KINDS})")
    if step not in candidate.steps:
        raise ValueError(f"candidate {candidate.name!r} has no material for step {step!r}")
    if unit_ids is not None and not unit_ids:
        raise ValueError("unit_ids must not be empty")
    if unit_ids is not None:
        unit_ids = sorted(unit_ids)

    if name is not None:
        case_name = _slugify(name)
    elif unit_ids is not None:
        case_name = _slugify(f"{candidate.name}__{unit_ids[0]}-{unit_ids[-1]}")
    else:
        case_name = _slugify(candidate.name)
    case_dir = Path(dest_root) / step / case_name
    if case_dir.exists():
        if name is not None or unit_ids is None:
            raise FileExistsError(f"case already exists: {case_dir}")
        case_dir = _numbered_case_dir(case_dir)
        case_name = case_dir.name
    fixture_dir = case_dir / FIXTURE_DIR
    fixture_dir.mkdir(parents=True)

    params: dict[str, Any] = {}
    try:
        if step == "plan":
            _write_plan_fixture(candidate, fixture_dir, unit_ids)
        else:
            _write_translate_fixture(candidate, fixture_dir, unit_ids)
            if candidate.target_langs:
                params["target_lang"] = candidate.target_langs[0]
    except Exception:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise

    case_yaml: dict[str, Any] = {
        "name": case_name,
        "step": step,
        "kind": kind,
        "source": str(candidate.run_dir),
        "params": params,
    }
    if unit_ids is not None:
        case_yaml["range"] = {"unit_ids": unit_ids}
    (case_dir / CASE_YAML).write_text(yaml.safe_dump(case_yaml, allow_unicode=True), encoding="utf-8")
    return case_dir


def _numbered_case_dir(case_dir: Path) -> Path:
    """First free ``<name>_2``, ``<name>_3`` … sibling of *case_dir*."""
    n = 2
    while case_dir.with_name(f"{case_dir.name}_{n}").exists():
        n += 1
    return case_dir.with_name(f"{case_dir.name}_{n}")


def _filter_segment_units(units: list[dict], unit_ids: list[str]) -> list[dict]:
    """Keep only units whose unit_id is in *unit_ids*, sorted by the original order in *units*."""
    unit_set = frozenset(unit_ids)
    unknown = sorted(set(unit_ids) - {str(u.get("unit_id", "")) for u in units})
    if unknown:
        raise ValueError(f"unknown unit_ids: {unknown}")
    return [u for u in units if str(u.get("unit_id", "")) in unit_set]


def _in_span(w_start: float, w_end: float, start: float, end: float) -> bool:
    """Word-coverage test with the same tolerance as ``artifacts._slice_words_for_unit``."""
    return w_start >= start - 0.05 and w_end <= end + 0.05


def _write_plan_fixture(candidate: Candidate, fixture_dir: Path, unit_ids: list[str] | None = None) -> None:
    """``fixture/segment.json`` + ``fixture/words.json`` from a transcript.

    Full-layout runs reuse the pipeline's own ``segment/segment.json``;
    flat runs regenerate segments from the word timeline via the real
    :func:`light_subtitle.segment.run`.  With *unit_ids*, both files are
    filtered to the selected segments and the words they cover.
    """
    if candidate.transcript is None:
        raise ValueError(f"candidate {candidate.name!r} has no transcript")
    data = _read_transcript(candidate.transcript)
    if data is None:
        raise ValueError(f"unusable transcript: {candidate.transcript}")
    words = [word_from_dict(w) for w in data["words"]]

    seg_json = candidate.run_dir / "segment" / artifacts.SEGMENT_JSON
    if candidate.transcript.name == "transcript.json" and seg_json.is_file():
        if unit_ids is None:
            shutil.copyfile(seg_json, fixture_dir / artifacts.SEGMENT_JSON)
        else:
            start, end = _write_filtered_segment_json(seg_json, fixture_dir / artifacts.SEGMENT_JSON, unit_ids)
            words = [w for w in words if _in_span(w.start, w.end, start, end)]
    else:
        segments = segment_step.run(words)
        if unit_ids is not None:
            segments = _filter_segment_list(segments, unit_ids)
            words = [w for w in words if _in_span(w.start, w.end, segments[0].start, segments[-1].end)]
        export_segments(words, segments, str(fixture_dir / artifacts.SEGMENT_JSON))
    (fixture_dir / "words.json").write_text(
        json.dumps([word_to_dict(w) for w in words], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _filter_segment_list(segments: list[Segment], unit_ids: list[str]) -> list[Segment]:
    """Keep only segments whose unit_id is in *unit_ids*, in original order."""
    unit_set = frozenset(unit_ids)
    unknown = sorted(set(unit_ids) - {s.unit_id for s in segments})
    if unknown:
        raise ValueError(f"unknown unit_ids: {unknown}")
    return [s for s in segments if s.unit_id in unit_set]


def _write_filtered_segment_json(src: Path, dest: Path, unit_ids: list[str]) -> tuple[float, float]:
    """Write *src* segment.json filtered to *unit_ids*; returns the kept time span."""
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    units = data.get("units", [])
    kept = _filter_segment_units(units, unit_ids)
    start, end = float(kept[0].get("start", 0.0)), float(kept[-1].get("end", 0.0))
    kept_words = [
        w for w in data.get("words", []) if _in_span(float(w.get("start", 0.0)), float(w.get("end", 0.0)), start, end)
    ]
    out = dict(data, units=kept)
    if "words" in out:
        out["words"] = kept_words
    if "total_units" in out:
        out["total_units"] = len(kept)
    if "total_words" in out:
        out["total_words"] = len(kept_words)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return start, end


def _write_translate_fixture(candidate: Candidate, fixture_dir: Path, unit_ids: list[str] | None = None) -> None:
    """``fixture/plan.json`` + optional glossary / summary sidecars.

    With *unit_ids*, plan.json keeps its schema but only the selected units;
    glossary / summary are always copied whole.
    """
    plan_json = candidate.plan_json
    if not plan_json.is_file():
        raise ValueError(f"candidate {candidate.name!r} has no plan/plan.json")
    if unit_ids is None:
        shutil.copyfile(plan_json, fixture_dir / artifacts.PLAN_JSON)
    else:
        with open(plan_json, encoding="utf-8") as f:
            data = json.load(f)
        units = data.get("units", [])
        kept = _filter_segment_units(units, unit_ids)
        out = dict(data, units=kept)
        (fixture_dir / artifacts.PLAN_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for sidecar in ("glossary.json", "summary.json"):
        src = candidate.run_dir / "context" / sidecar
        if src.is_file():
            shutil.copyfile(src, fixture_dir / sidecar)


def _slugify(name: str) -> str:
    """Filesystem-safe case name (runs already use safe slugs; be defensive)."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "case"


# ── Console rendering ───────────────────────────────────────────────────────


def format_duration(seconds: float) -> str:
    """``mm:ss`` (or ``h:mm:ss``) for the candidates table."""
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def format_table(candidates: list[Candidate]) -> str:
    """Plain-text candidate table for ``light-eval harvest``."""
    headers = ("NAME", "DURATION", "STEPS", "LANG", "SCALE")
    rows = [
        (
            c.name,
            format_duration(c.duration_s),
            ",".join(c.steps),
            c.lang_pair,
            f"{c.n_words}w/{c.n_units}u",
        )
        for c in candidates
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(headers))]
    lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return "\n".join(lines)
