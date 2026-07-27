"""LLM judge — problem-type defect detection.

The judge inspects step outputs and classifies defects into predefined
problem types, each carrying a default severity (error / warning).  A case
passes only when it has zero confirmed error-severity defects.

Per-dimension scores are gone; the output contract is now a flat ``issues``
array with ``{unit_id, problem_type, note}`` entries.  Outputs longer than
``batch_size`` are judged in batches; issues are merged and deduplicated.
"""

from __future__ import annotations

import json
import logging
import re

from light_llm.client import OpenAIClient
from light_llm.json_extract import extract_json_object
from light_subtitle.config import PlanConfig

from ..loader import Fixture
from ..models import PROBLEM_TYPES, EvalCase, ProblemTypeStats, StepOutput, problem_type_severity
from ..prompts import render_prompt

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_ATTEMPTS = 2
_TEMPERATURE = 0.1
_SOFT_CAP_RATIO = 1.15
_MERGE_HINT = re.compile(r"合并|merge", re.IGNORECASE)

_SEVERITY_ORDER = {"error": 0, "warning": 1}


def suggest_overall(scores: list[ProblemTypeStats]) -> str:
    """Derived verdict: any error → fail; only warnings → borderline; empty → pass."""
    if any(s.error_count > 0 for s in scores):
        return "fail"
    if any(s.warning_count > 0 for s in scores):
        return "borderline"
    return "pass"


# ── Judge ───────────────────────────────────────────────────────────────────


class LLMJudge:
    """Inspects step outputs and reports defects by problem type."""

    def __init__(self, client: OpenAIClient | None, *, batch_size: int = BATCH_SIZE):
        self._client = client
        self._batch_size = batch_size

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[ProblemTypeStats]:
        if self._client is None or output.skipped or output.error or not output.output:
            return []
        step_types = PROBLEM_TYPES.get(case.step, {})
        if not step_types:
            return []
        batches = [output.output[i : i + self._batch_size] for i in range(0, len(output.output), self._batch_size)]
        all_issues: list[dict[str, str]] = []
        for index, batch in enumerate(batches):
            batch_issues = self._judge_batch(case, fixture, batch, step_types)
            if batch_issues is None:
                logger.warning(
                    "LLM judge: batch %d/%d of case %s unparseable after retry — skipped",
                    index + 1,
                    len(batches),
                    case.name,
                )
                continue
            all_issues.extend(batch_issues)
        return _merge_issues(all_issues, case.step)

    # ── One batch ───────────────────────────────────────────────────────────

    def _judge_batch(
        self, case: EvalCase, fixture: Fixture, batch: list[dict], step_types: dict[str, dict[str, str]]
    ) -> list[dict[str, str]] | None:
        prompt = _render_batch_prompt(case, fixture, batch)
        assert self._client is not None
        for attempt in range(MAX_ATTEMPTS):
            content, _usage = self._client.chat(
                [{"role": "system", "content": prompt}],
                temperature=_TEMPERATURE,
            )
            issues = _parse_issues_flat(content, step_types)
            if issues is not None:
                return _filter_issues(issues, case, batch)
            logger.info("LLM judge: unparseable response (attempt %d/%d)", attempt + 1, MAX_ATTEMPTS)
        return None


# ── Prompt rendering ────────────────────────────────────────────────────────


def _format_segment_line(unit_id: str, start: float, end: float, text: str, speaker: str = "") -> str:
    label = f"[{speaker}] " if speaker else ""
    return f"{unit_id} | {start:.2f}-{end:.2f} | {label}{text}"


def _batch_source_segments(step: str, fixture: Fixture, batch: list[dict]) -> list:
    if not batch:
        return fixture.segments
    if step == "plan":
        lo = min(float(item.get("start", 0.0)) for item in batch)
        hi = max(float(item.get("end", 0.0)) for item in batch)
        indices = [i for i, seg in enumerate(fixture.segments) if seg.end >= lo and seg.start <= hi]
        if not indices:
            return fixture.segments
        first, last = max(0, indices[0] - 1), min(len(fixture.segments) - 1, indices[-1] + 1)
        return fixture.segments[first : last + 1]
    referenced = {str(item.get("unit_id", "")) for item in batch} | {
        str(m) for item in batch for m in item.get("merged_from") or []
    }
    by_id = [seg for seg in fixture.segments if seg.unit_id in referenced]
    return by_id or fixture.segments


def _render_batch_prompt(case: EvalCase, fixture: Fixture, batch: list[dict]) -> str:
    source_lines = "\n".join(
        _format_segment_line(seg.unit_id, seg.start, seg.end, seg.source_text, seg.speaker or "")
        for seg in _batch_source_segments(case.step, fixture, batch)
    )
    if case.step == "plan":
        unit_lines = "\n".join(
            _format_segment_line(
                str(unit.get("unit_id", "")),
                float(unit.get("start", 0.0)),
                float(unit.get("end", 0.0)),
                str(unit.get("text", "")),
                str(unit.get("speaker") or ""),
            )
            for unit in batch
        )
        return render_prompt(
            "judge_plan.j2",
            source_lines=source_lines,
            unit_lines=unit_lines,
            min_duration=case.params.get("min_duration", PlanConfig().min_duration),
            max_duration=case.params.get("max_duration", PlanConfig().max_duration),
        )
    cue_lines = "\n".join(_format_cue_line(cue) for cue in batch)
    return render_prompt(
        "judge_translate.j2",
        source_lines=source_lines,
        cue_lines=cue_lines,
        glossary=fixture.glossary,
        summary=fixture.summary,
        target_lang=str(case.params.get("target_lang", "zh")),
    )


def _format_cue_line(cue: dict) -> str:
    unit_ref = str(cue.get("unit_id", ""))
    merged = cue.get("merged_from") or []
    if merged:
        unit_ref = f"{unit_ref} (+{', '.join(str(m) for m in merged)})"
    return _format_segment_line(
        f"{cue.get('cue_id', '')}->{unit_ref}",
        float(cue.get("start", 0.0)),
        float(cue.get("end", 0.0)),
        str(cue.get("text", "")),
        str(cue.get("speaker") or ""),
    )


# ── Parsing ────────────────────────────────────────────────────────────────


def _parse_issues_flat(content: str, step_types: dict[str, dict[str, str]]) -> list[dict[str, str]] | None:
    """Parse the judge JSON flat issues array; None when unusable."""
    fragment = extract_json_object(content)
    if fragment is None:
        return None
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("issues")
    if not isinstance(raw, list):
        return None
    valid_types = frozenset(step_types)
    issues: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        unit_id = item.get("unit_id")
        problem_type = item.get("problem_type")
        note = item.get("note", "")
        if not isinstance(unit_id, str) or not unit_id:
            continue
        if not isinstance(problem_type, str) or problem_type not in valid_types:
            continue
        issues.append(
            {
                "unit_id": unit_id,
                "problem_type": problem_type,
                "note": str(note),
            }
        )
    return issues or None


# ── Post-filter ─────────────────────────────────────────────────────────────


def _filter_issues(issues: list[dict[str, str]], case: EvalCase, batch: list[dict]) -> list[dict[str, str]]:
    batch_ids = {str(item.get("unit_id", "")) for item in batch}
    if case.step != "plan":
        for item in batch:
            batch_ids.add(str(item.get("cue_id", "")))
            batch_ids.update(str(m) for m in item.get("merged_from") or [])
    batch_ids.discard("")
    soft_cap = float(case.params.get("max_duration", PlanConfig().max_duration)) * _SOFT_CAP_RATIO
    by_id = {str(u.get("unit_id", "")): u for u in batch}
    kept = []
    for issue in issues:
        if issue["unit_id"] not in batch_ids:
            logger.info("LLM judge: dropping out-of-batch issue on %r (%s)", issue["unit_id"], case.name)
            continue
        if case.step == "plan" and _is_infeasible_merge(issue, by_id, soft_cap):
            logger.info("LLM judge: dropping infeasible merge suggestion on %r (%s)", issue["unit_id"], case.name)
            continue
        kept.append(issue)
    return kept


def _is_infeasible_merge(issue: dict[str, str], batch_by_id: dict[str, dict], soft_cap: float) -> bool:
    if not _MERGE_HINT.search(issue.get("note", "")):
        return False
    named = _mentioned_ids(issue["note"], batch_by_id)
    units = [batch_by_id[i] for i in dict.fromkeys([issue["unit_id"], *named]) if i in batch_by_id]
    if len(units) < 2:
        return False
    merged = max(float(u.get("end", 0.0)) for u in units) - min(float(u.get("start", 0.0)) for u in units)
    return merged > soft_cap


def _mentioned_ids(text: str, batch_by_id: dict[str, dict]) -> list[str]:
    if not batch_by_id:
        return []
    pattern = re.compile("|".join(re.escape(i) for i in sorted(batch_by_id, key=len, reverse=True)))
    return pattern.findall(text)


def _natural_key(text: str) -> tuple:
    return tuple(int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text))


# ── Issue merging ───────────────────────────────────────────────────────────


def _merge_issues(all_issues: list[dict[str, str]], step: str) -> list[ProblemTypeStats]:
    """Merge batch issues: deduplicate, then aggregate into per-type ProblemTypeStats."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for issue in all_issues:
        key = (issue["unit_id"], issue["problem_type"], issue["note"])
        if key not in seen:
            seen.add(key)
            deduped.append(issue)
    deduped.sort(key=lambda i: _natural_key(i["unit_id"]))

    type_buckets: dict[str, dict[str, list]] = {}
    step_types = PROBLEM_TYPES.get(step, {})
    for issue in deduped:
        pt = issue["problem_type"]
        bucket = type_buckets.setdefault(pt, {"errors": [], "warnings": []})
        severity = problem_type_severity(step, pt)
        if severity == "error":
            bucket["errors"].append(issue)
        else:
            bucket["warnings"].append(issue)

    result = []
    for pt_name in step_types:
        bucket = type_buckets.get(pt_name, {"errors": [], "warnings": []})
        e_count = len(bucket["errors"])
        w_count = len(bucket["warnings"])
        all_items = bucket["errors"] + bucket["warnings"]
        result.append(
            ProblemTypeStats(
                problem_type=pt_name,
                error_count=e_count,
                warning_count=w_count,
                passed=e_count == 0,
                evidence=list(dict.fromkeys(i["unit_id"] for i in all_items)),
                issues=all_items,
            )
        )
    return result
