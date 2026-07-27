"""Eval data model — case, step output, problem-type stats, and report.

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

# ── Problem types ──────────────────────────────────────────────────────────
# Each problem type has a label (Chinese name) and a default severity
# (error | warning).  The severity determines whether the type gates pass/fail:
# any confirmed defect of an error-severity type fails the case.

PROBLEM_TYPES: dict[str, dict[str, dict[str, str]]] = {
    "plan": {
        "semantic_boundary": {"label": "语义边界不当", "severity": "error"},
        "over_fragmentation": {"label": "过碎可合并", "severity": "warning"},
        "over_long_unit": {"label": "过长应拆分", "severity": "warning"},
        "dangling_word": {"label": "孤词/悬垂词尾", "severity": "error"},
        "empty_unit": {"label": "空单元", "severity": "error"},
        "flash_unit": {"label": "闪帧单元", "severity": "error"},
    },
    "translate": {
        "missing_content": {"label": "漏译", "severity": "error"},
        "extra_content": {"label": "增译/幻译", "severity": "error"},
        "semantic_drift": {"label": "语义偏移/歪曲原意", "severity": "error"},
        "translation_ese": {"label": "翻译腔/不自然", "severity": "warning"},
        "unit_mismatch": {"label": "单元串位/错位", "severity": "error"},
        "terminology_inconsistent": {"label": "术语不一致", "severity": "warning"},
        "word_choice": {"label": "用词不当", "severity": "warning"},
        "bad_line_break": {"label": "断行不当", "severity": "warning"},
    },
}


def problem_type_severity(step: str, problem_type: str) -> str:
    return PROBLEM_TYPES.get(step, {}).get(problem_type, {}).get("severity", "warning")


def problem_type_label(step: str, problem_type: str) -> str:
    return PROBLEM_TYPES.get(step, {}).get(problem_type, {}).get("label", problem_type)


# ── Case ────────────────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    """One eval case discovered under ``<suite>/<step>/<case_name>/``."""

    name: str
    step: StepName
    kind: CaseKind
    source: str
    params: dict[str, Any] = field(default_factory=dict)
    case_dir: Path = Path()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "step": self.step,
            "kind": self.kind,
            "source": self.source,
            "params": self.params,
        }


# ── Defect ──────────────────────────────────────────────────────────────────


@dataclass
class Defect:
    """One defect record — a specific problem found on a unit.

    *confirmed* is a tri-state: None = undecided, True = confirmed valid issue,
    False = dismissed by human reviewer.
    """

    unit_id: str
    problem_type: str
    note: str = ""
    confirmed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"unit_id": self.unit_id, "problem_type": self.problem_type}
        if self.note:
            d["note"] = self.note
        if self.confirmed is not None:
            d["confirmed"] = self.confirmed
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Defect:
        confirmed = data.get("confirmed")
        if confirmed is not None:
            confirmed = bool(confirmed)
        return cls(
            unit_id=str(data.get("unit_id", "")),
            problem_type=str(data.get("problem_type", "")),
            note=str(data.get("note", "")),
            confirmed=confirmed,
        )


# ── Annotation ──────────────────────────────────────────────────────────────


@dataclass
class Annotation:
    """Human annotation from ``annotation.yaml``.

    ``defects`` holds the reviewed defect list.  ``overall`` is the human
    final verdict (pass / borderline / fail).  ``judge_suggestion`` holds the
    raw AI pre-judge JSON when the human reviewed an AI suggestion.
    """

    defects: list[Defect] = field(default_factory=list)
    overall: str = ""
    judge_suggestion: dict | None = None
    reviewed_by: str = ""

    def to_dict(self) -> dict:
        data: dict[str, Any] = {
            "defects": [d.to_dict() for d in self.defects],
            "overall": self.overall,
        }
        if self.judge_suggestion is not None:
            data["judge_suggestion"] = self.judge_suggestion
        if self.reviewed_by:
            data["reviewed_by"] = self.reviewed_by
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Annotation:
        defects = [Defect.from_dict(d) for d in (data.get("defects") or []) if isinstance(d, dict)]
        return cls(
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
    output: list[dict] = field(default_factory=list)
    usage: dict | None = None
    duration_s: float = 0.0
    error: str | None = None
    skipped: bool = False

    def summary(self) -> dict:
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


# ── Problem-type stats ──────────────────────────────────────────────────────


@dataclass
class ProblemTypeStats:
    """Per-problem-type verdict for one case.

    ``error_count`` / ``warning_count`` are the raw issue counts returned by
    the judge.  ``passed`` indicates whether the type has no error-severity
    issues (i.e. no blocking problems).  ``issues`` holds the structured
    per-unit findings from the LLM judge (rule judges leave it empty).
    """

    problem_type: str
    error_count: int = 0
    warning_count: int = 0
    passed: bool = True
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return self.error_count + self.warning_count

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "problem_type": self.problem_type,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "passed": self.passed,
        }
        if self.detail:
            d["detail"] = self.detail
        d["evidence"] = self.evidence
        if self.issues:
            d["issues"] = self.issues
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ProblemTypeStats:
        issues = []
        for i in data.get("issues") or []:
            if not isinstance(i, dict):
                continue
            issue = {"unit_id": str(i.get("unit_id", "")), "problem": str(i.get("problem", ""))}
            if i.get("severity"):
                issue["severity"] = str(i["severity"])
            issues.append(issue)
        return cls(
            problem_type=data.get("problem_type", ""),
            error_count=data.get("error_count", 0),
            warning_count=data.get("warning_count", 0),
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
    scores: list[ProblemTypeStats]
    output: StepOutput

    @property
    def passed(self) -> bool:
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
        types: dict[str, dict[str, int]] = {}
        for result in self.cases:
            for score in result.scores:
                bucket = types.setdefault(score.problem_type, {"total": 0, "passed": 0})
                bucket["total"] += 1
                bucket["passed"] += int(score.passed)
        return {
            "n_cases": len(self.cases),
            "n_passed": sum(1 for c in self.cases if c.passed),
            "n_errored": sum(1 for c in self.cases if c.output.error),
            "n_skipped": sum(1 for c in self.cases if c.output.skipped),
            "problem_types": types,
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
