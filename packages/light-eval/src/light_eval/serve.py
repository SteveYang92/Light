"""Eval workbench — local FastAPI app for case harvesting and annotation.

Serves a single-page vanilla-JS UI (``web/index.html``, shipped as package
data) plus a small JSON API:

- ``GET  /api/candidates``                    harvest candidates from the output dir
- ``GET  /api/candidates/{name}/units``       unit sequence of one candidate (range picking)
- ``POST /api/cases``                         create a case from a candidate (whole video or unit range)
- ``GET  /api/cases``                         list existing cases (with annotation status)
- ``GET  /api/cases/{step}/{name}/output``        step output for the detail view
- ``POST /api/cases/{step}/{name}/run``           (re-)run the runner for one case
- ``POST /api/cases/{step}/{name}/judge``         LLM pre-judge of the persisted output
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
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import loader
from .harvest import create_case, scan_candidate_units, scan_candidates
from .judges.llm import LLMJudge, suggest_overall
from .models import VALID_KINDS, VALID_STEPS, EvalCase, StepOutput
from .runner import build_llm_client, run_case

# Annotation dimensions per step (drives both validation and the UI form).
ANNOTATION_DIMENSIONS: dict[str, list[str]] = {
    "plan": ["boundary_quality", "split_necessity"],
    "translate": ["faithfulness", "naturalness", "unit_integrity", "terminology"],
}

OVERALL_CHOICES = ("pass", "borderline", "fail")

_RUN_OUTPUT = Path(".eval_run") / "output.json"
_RUN_JUDGE = Path(".eval_run") / "judge.json"


# ── Request models ──────────────────────────────────────────────────────────


class CreateCaseRequest(BaseModel):
    candidate: str  # candidate name from GET /api/candidates
    step: str
    kind: str = "control"
    start_unit: str | None = None  # optional unit range (both or neither)
    end_unit: str | None = None


class Defect(BaseModel):
    unit_id: str = ""
    issue: str = ""
    severity: str = ""  # must_fix | minor；空 = 未分级（读取侧按 minor 兼容处理）


class AnnotationRequest(BaseModel):
    dimensions: dict[str, int] = {}
    defects: list[Defect] = []
    overall: str = ""
    judge_suggestion: dict | None = None  # raw judge-endpoint JSON, echoed back for audit
    reviewed_by: str = ""


# ── App factory ─────────────────────────────────────────────────────────────


def create_app(suite_dir: str | Path = "tests/eval", output_dir: str | Path = "output") -> FastAPI:
    """Build the workbench app bound to *suite_dir* (cases) and *output_dir* (candidates)."""
    suite_dir = Path(suite_dir)
    output_dir = Path(output_dir)
    app = FastAPI(title="light-eval workbench")

    # ── UI ──────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("light_eval").joinpath("web/index.html").read_text(encoding="utf-8")

    # ── Candidates ──────────────────────────────────────────────────────────

    @app.get("/api/candidates")
    def list_candidates() -> dict:
        candidates = scan_candidates(output_dir) if output_dir.is_dir() else []
        return {"output_dir": str(output_dir), "candidates": [c.to_dict() for c in candidates]}

    @app.get("/api/candidates/{name}/units")
    def candidate_units(name: str, step: str | None = None) -> dict:
        """Unit sequence of one candidate for range picking (``?step=plan|translate``)."""
        match = _find_candidate(name)
        if match is None:
            raise HTTPException(404, f"candidate not found: {name!r}")
        chosen = step or ("translate" if "translate" in match.steps else "plan")
        try:
            units = scan_candidate_units(match, chosen)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"candidate": name, "step": chosen, "units": units}

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
                match, req.step, req.kind, suite_dir, start_unit=req.start_unit, end_unit=req.end_unit
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
        """LLM pre-judge of the persisted step output; persists to .eval_run/judge.json.

        Re-running starts the review from a clean slate: the stale judge.json
        is deleted up front (a failed run leaves nothing behind), and on
        success the saved annotation.yaml is deleted too (it is kept when
        judging fails).
        """
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
        judge_path.unlink(missing_ok=True)  # 重跑先删旧评审——失败也不残留过期结果
        scores = LLMJudge(llm).score(case, fixture, step_output)
        if not scores:
            raise HTTPException(409, "步骤输出不可用（出错 / 跳过 / 为空），无法预评")
        # 重跑即重来：评审成功后删除旧标注，本轮从干净状态开始（失败则保留旧标注）
        (case.case_dir / loader.ANNOTATION_YAML).unlink(missing_ok=True)
        result = {
            "dimensions": {s.dimension: {"score": s.score, "summary": s.detail, "issues": s.issues} for s in scores},
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
        empty = {"dimensions": {}, "defects": [], "overall": ""}
        return annotation.to_dict() if annotation is not None else empty

    @app.put("/api/cases/{step}/{name}/annotation")
    def put_annotation(step: str, name: str, req: AnnotationRequest) -> dict:
        case = _resolve_case(step, name)
        allowed = set(ANNOTATION_DIMENSIONS[case.step])
        unknown = sorted(set(req.dimensions) - allowed)
        if unknown:
            raise HTTPException(422, f"unknown dimensions for step {case.step!r}: {unknown}")
        for dim, value in req.dimensions.items():
            if not 1 <= value <= 5:
                raise HTTPException(422, f"dimension {dim!r} must be 1-5, got {value}")
        if req.overall and req.overall not in OVERALL_CHOICES:
            raise HTTPException(422, f"overall must be one of {OVERALL_CHOICES}")
        data: dict[str, Any] = {
            "dimensions": req.dimensions,
            "defects": [
                {k: v for k, v in {"unit_id": d.unit_id, "issue": d.issue, "severity": d.severity}.items() if v}
                for d in req.defects
                if d.issue or d.unit_id
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
        """Candidate named *name* from a fresh scan, or None."""
        if not output_dir.is_dir():
            return None
        return next((c for c in scan_candidates(output_dir) if c.name == name), None)

    def _resolve_case(step: str, name: str) -> EvalCase:
        """Load the case identified by ``<step>/<name>`` or 404."""
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
            "annotation_dimensions": ANNOTATION_DIMENSIONS[case.step],
        }

    def _shape_items(case: EvalCase, output: list[dict]) -> list[dict]:
        """View-model rows: plan units, or source/translation pairs for translate."""
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
