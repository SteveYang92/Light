"""Eval workbench — local FastAPI app for case harvesting and annotation.

Serves a single-page vanilla-JS UI (``web/index.html``, shipped as package
data) plus a small JSON API:

- ``GET    /api/output-dirs``                    list manually-added scan directories
- ``POST   /api/output-dirs``                    add a scan directory
- ``DELETE /api/output-dirs``                    remove a scan directory
- ``GET    /api/candidates``                     harvest candidates from all registered dirs
- ``GET    /api/candidates/{name}?step=``        candidate detail (subtitle list filtered by step)
- ``POST   /api/cases``                          create a case from a candidate (selected unit_ids)
- ``GET    /api/cases``                          list existing cases (with annotation status)
- ``GET    /api/cases/{step}/{name}/output``        step output for the detail view
- ``POST   /api/cases/{step}/{name}/run``           (re-)run the runner for one case
- ``POST   /api/cases/{step}/{name}/judge``         LLM pre-judge of the persisted output
- ``GET/PUT /api/cases/{step}/{name}/annotation`` read / save ``annotation.yaml``

Cases are addressed as ``<step>/<name>`` (e.g. ``plan/my_case``).  Run outputs are
persisted to ``<case_dir>/.eval_run/output.json`` so the detail view survives
restarts.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import loader
from .harvest import create_case, scan_candidate_units, scan_candidates
from .judges.llm import LLMJudge, suggest_overall
from .models import PROBLEM_TYPES, VALID_KINDS, VALID_STEPS, EvalCase, StepOutput
from .runner import build_llm_client, run_case

OVERALL_CHOICES = ("pass", "borderline", "fail")

_RUN_OUTPUT = Path(".eval_run") / "output.json"
_RUN_JUDGE = Path(".eval_run") / "judge.json"


# ── Request models ──────────────────────────────────────────────────────────


class AddDirRequest(BaseModel):
    path: str


class CreateCaseRequest(BaseModel):
    candidate: str
    step: str
    kind: str = "control"
    unit_ids: list[str] | None = None
    name: str | None = None


class DefectIn(BaseModel):
    unit_id: str = ""
    problem_type: str = ""
    note: str = ""
    confirmed: bool | None = None


class AnnotationRequest(BaseModel):
    defects: list[DefectIn] = []
    overall: str = ""
    judge_suggestion: dict | None = None
    reviewed_by: str = ""


# ── App factory ─────────────────────────────────────────────────────────────


def create_app(
    suite_dir: str | Path = "tests/eval",
    output_dirs: list[str | Path] | None = None,
) -> FastAPI:
    """Build the workbench app bound to *suite_dir* (cases) and *output_dirs* (candidates)."""
    suite_dir = Path(suite_dir)
    dirs: list[Path] = [Path(d) for d in (output_dirs or []) if Path(d).is_dir()]
    app = FastAPI(title="light-eval workbench")
    app.state.output_dirs = dirs

    # ── UI ──────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("light_eval").joinpath("web/index.html").read_text(encoding="utf-8")

    # ── Output directories ─────────────────────────────────────────────────

    @app.get("/api/output-dirs")
    def list_output_dirs() -> dict:
        return {"dirs": [str(d) for d in app.state.output_dirs]}

    @app.post("/api/output-dirs")
    def add_output_dir(req: AddDirRequest) -> dict:
        path = Path(req.path).resolve()
        if not path.is_dir():
            raise HTTPException(400, f"not a directory: {path}")
        if any(path == d or path in d.parents for d in app.state.output_dirs):
            raise HTTPException(409, f"directory already added or is a parent: {path}")
        if any(d == path or d in path.parents for d in app.state.output_dirs):
            raise HTTPException(409, f"a parent of this directory is already added: {path}")
        app.state.output_dirs.append(path)
        return {"ok": True, "dirs": [str(d) for d in app.state.output_dirs]}

    @app.delete("/api/output-dirs")
    def remove_output_dir(path: str = Query(...)) -> dict:
        p = Path(path).resolve()
        matches = [d for d in app.state.output_dirs if d == p]
        if not matches:
            raise HTTPException(404, f"directory not found: {path}")
        for m in matches:
            app.state.output_dirs.remove(m)
        return {"ok": True, "dirs": [str(d) for d in app.state.output_dirs]}

    # ── Candidates ──────────────────────────────────────────────────────────

    @app.get("/api/candidates")
    def list_candidates() -> dict:
        all_candidates = []
        seen: set[tuple[str, str]] = set()
        for d in app.state.output_dirs:
            if not d.is_dir():
                continue
            for c in scan_candidates(d):
                key = (c.name, c.run)
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(c)
        all_candidates.sort(key=lambda c: (c.run, c.name))
        return {
            "dirs": [str(d) for d in app.state.output_dirs],
            "candidates": [
                {
                    "name": c.name,
                    "run": c.run,
                    "duration_s": round(c.duration_s, 3),
                    "steps": c.steps,
                    "lang_pair": c.lang_pair,
                }
                for c in all_candidates
            ],
        }

    @app.get("/api/candidates/{name}")
    def candidate_detail(name: str, step: str | None = None) -> dict:
        """Candidate subtitle list for the detail view (``?step=plan|translate``)."""
        match = _find_candidate(name)
        if match is None:
            raise HTTPException(404, f"candidate not found: {name!r}")
        chosen = step
        if chosen not in match.steps:
            chosen = match.steps[0]
        if chosen not in VALID_STEPS:
            raise HTTPException(422, f"invalid step {chosen!r}")
        try:
            units = scan_candidate_units(match, chosen)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        target_texts: dict[str, str] = {}
        if chosen == "translate" and match.target_langs:
            trans_path = match.run_dir / "translations" / "raw.json"
            if trans_path.is_file():
                try:
                    with open(trans_path, encoding="utf-8") as f:
                        trans_data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    pass
                else:
                    trans_units = []
                    if isinstance(trans_data, dict):
                        trans_units = trans_data.get("units") or trans_data.get("output") or []
                    elif isinstance(trans_data, list):
                        trans_units = trans_data
                    target_texts = {str(u.get("unit_id", "")): u.get("text", "") for u in trans_units}

        return {
            "candidate": {
                "name": match.name,
                "run": match.run,
                "duration_s": round(match.duration_s, 3),
                "steps": match.steps,
                "source_lang": match.source_lang,
                "target_langs": match.target_langs,
                "lang_pair": match.lang_pair,
            },
            "step": chosen,
            "units": [{**u, "target_text": target_texts.get(str(u["unit_id"]), "")} for u in units],
        }

    # ── Cases ───────────────────────────────────────────────────────────────

    @app.post("/api/cases", status_code=201)
    def create_case_api(req: CreateCaseRequest) -> dict:
        if req.step not in VALID_STEPS:
            raise HTTPException(422, f"invalid step {req.step!r}")
        if req.kind not in VALID_KINDS:
            raise HTTPException(422, f"invalid kind {req.kind!r}")
        match = _find_candidate(req.candidate)
        if match is None:
            raise HTTPException(404, f"candidate not found: {req.candidate!r}")
        try:
            case_dir = create_case(
                match,
                req.step,
                req.kind,
                suite_dir,
                unit_ids=req.unit_ids,
                name=req.name or None,
            )
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"case": _case_dict(loader.load_case(case_dir))}

    @app.get("/api/cases")
    def list_cases() -> dict:
        cases = loader.discover_cases(suite_dir) if suite_dir.is_dir() else []
        return {"suite_dir": str(suite_dir), "cases": [_case_dict(c) for c in cases]}

    # ── Case detail / run ───────────────────────────────────────────────────

    @app.get("/api/cases/{step}/{name}/output")
    def case_output(step: str, name: str) -> dict:
        case = _resolve_case(step, name)
        path = case.case_dir / _RUN_OUTPUT
        if not path.is_file():
            return {"case": _case_dict(case), "has_output": False, "items": []}
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
        return {
            "case": _case_dict(case),
            "has_output": True,
            "error": stored.get("error"),
            "skipped": stored.get("skipped", False),
            "usage": stored.get("usage"),
            "duration_s": stored.get("duration_s", 0.0),
            "items": _shape_items(case, stored.get("output", [])),
        }

    @app.post("/api/cases/{step}/{name}/run")
    def run_case_api(step: str, name: str) -> dict:
        case = _resolve_case(step, name)
        llm = build_llm_client()
        if case.step == "translate" and llm is None:
            raise HTTPException(409, "translate 步骤需要 LLM：请设置环境变量 DEEPSEEK_API_KEY 后重启 serve")
        fixture = loader.load_fixture(case)
        step_output = run_case(case, fixture, llm=llm)
        out_path = case.case_dir / _RUN_OUTPUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(step_output.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "case": _case_dict(case),
            "has_output": True,
            "error": step_output.error,
            "skipped": step_output.skipped,
            "usage": step_output.usage,
            "duration_s": round(step_output.duration_s, 3),
            "items": _shape_items(case, step_output.output),
        }

    @app.post("/api/cases/{step}/{name}/judge")
    def judge_case_api(step: str, name: str) -> dict:
        case = _resolve_case(step, name)
        out_path = case.case_dir / _RUN_OUTPUT
        if not out_path.is_file():
            raise HTTPException(409, "尚无步骤输出 — 请先「运行 / 重跑」再预评")
        llm = build_llm_client()
        if llm is None:
            raise HTTPException(409, "AI 预评需要 LLM：请设置环境变量 DEEPSEEK_API_KEY 后重启 serve")
        fixture = loader.load_fixture(case)
        step_output = StepOutput.from_dict(json.loads(out_path.read_text(encoding="utf-8")))
        judge_path = case.case_dir / _RUN_JUDGE
        judge_path.unlink(missing_ok=True)
        scores = LLMJudge(llm).score(case, fixture, step_output)
        if not scores:
            raise HTTPException(409, "步骤输出不可用（出错 / 跳过 / 为空），无法预评")
        (case.case_dir / loader.ANNOTATION_YAML).unlink(missing_ok=True)
        result = {
            "problem_types": {
                s.problem_type: {
                    "error_count": s.error_count,
                    "warning_count": s.warning_count,
                    "passed": s.passed,
                    "issues": s.issues,
                    "evidence": s.evidence,
                }
                for s in scores
            },
            "suggested_overall": suggest_overall(scores),
        }
        judge_path.parent.mkdir(parents=True, exist_ok=True)
        judge_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # ── Annotation ──────────────────────────────────────────────────────────

    @app.get("/api/cases/{step}/{name}/annotation")
    def get_annotation(step: str, name: str) -> dict:
        case = _resolve_case(step, name)
        annotation = loader.load_annotation(case)
        empty = {"defects": [], "overall": ""}
        return annotation.to_dict() if annotation is not None else empty

    @app.put("/api/cases/{step}/{name}/annotation")
    def put_annotation(step: str, name: str, req: AnnotationRequest) -> dict:
        case = _resolve_case(step, name)
        valid_types = set(PROBLEM_TYPES.get(case.step, {}))
        if req.overall and req.overall not in OVERALL_CHOICES:
            raise HTTPException(422, f"overall must be one of {OVERALL_CHOICES}")
        for d in req.defects:
            if d.problem_type and d.problem_type not in valid_types:
                raise HTTPException(422, f"unknown problem_type {d.problem_type!r} for step {case.step!r}")
        data: dict[str, Any] = {
            "defects": [
                {
                    k: v
                    for k, v in {
                        "unit_id": d.unit_id,
                        "problem_type": d.problem_type,
                        "note": d.note,
                        "confirmed": d.confirmed,
                    }.items()
                    if v is not False or k == "confirmed"
                }
                for d in req.defects
                if d.problem_type or d.unit_id
            ],
            "overall": req.overall,
        }
        if req.judge_suggestion is not None:
            data["judge_suggestion"] = req.judge_suggestion
        if req.reviewed_by:
            data["reviewed_by"] = req.reviewed_by
        (case.case_dir / loader.ANNOTATION_YAML).write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return data

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _find_candidate(name: str):
        """Candidate named *name* from a fresh scan of all registered dirs, or None."""
        for d in app.state.output_dirs:
            if not d.is_dir():
                continue
            found = next((c for c in scan_candidates(d) if c.name == name), None)
            if found is not None:
                return found
        return None

    def _resolve_case(step: str, name: str) -> EvalCase:
        case_dir = suite_dir / step / name
        if step not in VALID_STEPS or not name or not (case_dir / loader.CASE_YAML).is_file():
            raise HTTPException(404, f"case not found: {step}/{name!r}")
        return loader.load_case(case_dir)

    def _case_dict(case: EvalCase) -> dict:
        annotation = loader.load_annotation(case)
        return {
            "id": f"{case.step}/{case.name}",
            "name": case.name,
            "step": case.step,
            "kind": case.kind,
            "source": case.source,
            "params": case.params,
            "annotated": annotation is not None,
            "has_judge_suggestion": annotation is not None and annotation.judge_suggestion is not None,
            "has_output": (case.case_dir / _RUN_OUTPUT).is_file(),
            "problem_types": PROBLEM_TYPES[case.step],
        }

    def _shape_items(case: EvalCase, output: list[dict]) -> list[dict]:
        if case.step == "plan":
            return [
                {
                    "unit_id": str(u.get("unit_id", "")),
                    "start": u.get("start", 0.0),
                    "end": u.get("end", 0.0),
                    "text": u.get("text", ""),
                }
                for u in output
            ]
        source_by_unit = {}
        try:
            source_by_unit = {seg.unit_id: seg.source_text for seg in loader.load_fixture(case).segments}
        except (OSError, ValueError, KeyError):
            pass
        return [
            {
                "unit_id": str(c.get("unit_id", "")),
                "start": c.get("start", 0.0),
                "end": c.get("end", 0.0),
                "source_text": source_by_unit.get(str(c.get("unit_id", "")), ""),
                "text": c.get("text", ""),
            }
            for c in output
        ]

    return app
