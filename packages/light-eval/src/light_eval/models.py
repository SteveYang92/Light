"""Eval data model — case, step output, dimension score, and report.

All models serialize to plain JSON dicts (``to_dict`` / ``from_dict``) so
reports are stable, diff-able artifacts.  ``StepOutput.output`` holds the
*serializable* form of the step result (plan units / translated cues);
judges work on that plus the re-loaded fixture, never on live objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

StepName = Literal["plan", "translate"]
CaseKind = Literal["control", "edge", "boundary"]

VALID_STEPS: tuple[str, ...] = ("plan", "translate")
VALID_KINDS: tuple[str, ...] = ("control", "edge", "boundary")


# ── Case ────────────────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    """One eval case discovered under ``<suite>/<step>/<case_name>/``."""

    name: str
    step: StepName
    kind: CaseKind
    source: str  # provenance: which real run / output dir this case was harvested from
    params: dict[str, Any] = field(default_factory=dict)  # e.g. target_lang, duration thresholds
    case_dir: Path = Path()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "step": self.step,
            "kind": self.kind,
            "source": self.source,
            "params": self.params,
        }


@dataclass
class Annotation:
    """Human annotation from ``annotation.yaml`` (used to calibrate judges later).

    ``judge_suggestion`` holds the raw LLM pre-judge JSON (from the workbench
    ``judge`` endpoint) when the human reviewed an AI pre-score; ``reviewed_by``
    records who finalized the annotation.  Both are optional — old
    ``annotation.yaml`` files without them load with defaults.
    """

    dimensions: dict[str, int] = field(default_factory=dict)  # dimension → 1-5 human score
    defects: list[dict[str, str]] = field(default_factory=list)  # [{"unit_id": ..., "issue": ...}]
    overall: str = ""
    judge_suggestion: dict | None = None  # raw judge endpoint JSON {dimensions, suggested_overall}
    reviewed_by: str = ""

    def to_dict(self) -> dict:
        # Omit the optional fields when unset so the base schema stays stable.
        data: dict[str, Any] = {"dimensions": self.dimensions, "defects": self.defects, "overall": self.overall}
        if self.judge_suggestion is not None:
            data["judge_suggestion"] = self.judge_suggestion
        if self.reviewed_by:
            data["reviewed_by"] = self.reviewed_by
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Annotation:
        defects = [
            {"unit_id": str(d.get("unit_id", "")), "issue": str(d.get("issue", ""))}
            for d in data.get("defects") or []
            if isinstance(d, dict)
        ]
        return cls(
            dimensions={str(k): int(v) for k, v in (data.get("dimensions") or {}).items()},
            defects=defects,
            overall=str(data.get("overall", "")),
            judge_suggestion=data.get("judge_suggestion"),
            reviewed_by=str(data.get("reviewed_by", "")),
        )


# ── Step output ─────────────────────────────────────────────────────────────


@dataclass
class StepOutput:
    """Result of running one pipeline step on one case."""

    case: str
    output: list[dict] = field(default_factory=list)  # serialized plan units / translated cues
    usage: dict | None = None
    duration_s: float = 0.0
    error: str | None = None
    skipped: bool = False  # e.g. translate without an LLM client

    def summary(self) -> dict:
        """Compact, report-friendly view (no full payload)."""
        return {
            "n_items": len(self.output),
            "duration_s": round(self.duration_s, 3),
            "usage": self.usage,
            "error": self.error,
            "skipped": self.skipped,
        }

    def to_dict(self) -> dict:
        return {
            "case": self.case,
            "output": self.output,
            "usage": self.usage,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StepOutput:
        return cls(
            case=data["case"],
            output=data.get("output", []),
            usage=data.get("usage"),
            duration_s=data.get("duration_s", 0.0),
            error=data.get("error"),
            skipped=data.get("skipped", False),
        )


# ── Dimension score ─────────────────────────────────────────────────────────


@dataclass
class DimensionScore:
    """One judge dimension verdict for one case.

    ``score`` is the computed metric value for rule judges (ratio / count);
    LLM judges will emit 1-5.  ``passed`` is the boolean gate; ``evidence``
    lists offending ids (unit_id / cue_id / word text).  ``issues`` holds the
    LLM judge's structured per-unit findings (``{"unit_id", "problem"}``);
    it stays empty for rule judges.
    """

    dimension: str
    score: float
    passed: bool
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        data: dict[str, Any] = {
            "dimension": self.dimension,
            "score": round(self.score, 4),
            "passed": self.passed,
            "detail": self.detail,
            "evidence": self.evidence,
        }
        if self.issues:  # omit when empty so the base schema stays stable
            data["issues"] = self.issues
        return data

    @classmethod
    def from_dict(cls, data: dict) -> DimensionScore:
        issues = [
            {"unit_id": str(i.get("unit_id", "")), "problem": str(i.get("problem", ""))}
            for i in data.get("issues") or []
            if isinstance(i, dict)
        ]
        return cls(
            dimension=data["dimension"],
            score=data.get("score", 0.0),
            passed=data.get("passed", False),
            detail=data.get("detail", ""),
            evidence=list(data.get("evidence", [])),
            issues=issues,
        )


# ── Report ──────────────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    """Per-case report entry: scores + step-output summary."""

    case: EvalCase
    scores: list[DimensionScore]
    output: StepOutput

    @property
    def passed(self) -> bool:
        """True only when the step ran (not skipped/errored) and every dimension passed."""
        return self.output.error is None and not self.output.skipped and all(s.passed for s in self.scores)

    def to_dict(self) -> dict:
        return {
            "case": self.case.to_dict(),
            "passed": self.passed,
            "scores": [s.to_dict() for s in self.scores],
            "output": self.output.to_dict(),
        }


@dataclass
class EvalReport:
    """Full suite report: per-case results + aggregate stats."""

    step: str
    cases: list[CaseResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def aggregate(self) -> dict:
        """Aggregate stats: case counts + per-dimension pass totals."""
        dimensions: dict[str, dict[str, int]] = {}
        for result in self.cases:
            for score in result.scores:
                bucket = dimensions.setdefault(score.dimension, {"total": 0, "passed": 0})
                bucket["total"] += 1
                bucket["passed"] += int(score.passed)
        return {
            "n_cases": len(self.cases),
            "n_passed": sum(1 for c in self.cases if c.passed),
            "n_errored": sum(1 for c in self.cases if c.output.error),
            "n_skipped": sum(1 for c in self.cases if c.output.skipped),
            "dimensions": dimensions,
        }

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "created_at": self.created_at,
            "aggregate": self.aggregate(),
            "cases": [c.to_dict() for c in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path
