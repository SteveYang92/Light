"""LLM judge — rubric-based 1-5 scoring via an OpenAI-compatible chat model.

One judge instance scores one case's step output along the rubric
dimensions for its step:

- ``plan``: ``boundary_quality`` (boundaries land on clause / phrase-group
  edges) and ``split_necessity`` (neither over-fragmented nor over-merged).
- ``translate``: ``faithfulness`` (no omission / addition),
  ``naturalness`` (spoken, not translationese), ``unit_integrity`` (cue ↔
  source unit correspondence) and ``terminology`` (glossary consistency).

Outputs longer than ``batch_size`` items are judged in time-ordered
batches; the per-dimension score is the *median* across batches (lower
middle for even counts, so one lenient batch cannot sink a large case),
the summary comes from the lowest-scoring batch, and per-unit issues are
merged across batches with duplicates dropped.  Raw verdicts pass through
a deterministic post-filter (:func:`_filter_verdicts`) that drops issues
the rubric cannot enforce: references to units outside the judged batch,
issues attached to a clean 5-score, and merge suggestions whose merged
duration would exceed the soft cap.  When the client is ``None`` (no API
key) the judge returns an empty list — reports then show the LLM judge as
skipped.
"""

from __future__ import annotations

import json
import logging
import re

from light_llm.client import OpenAIClient
from light_llm.json_extract import extract_json_object
from light_subtitle.config import PlanConfig

from ..loader import Fixture
from ..models import DimensionScore, EvalCase, StepOutput
from ..prompts import render_prompt

logger = logging.getLogger(__name__)

PLAN_DIMENSIONS: tuple[str, ...] = ("boundary_quality", "split_necessity")
TRANSLATE_DIMENSIONS: tuple[str, ...] = ("faithfulness", "naturalness", "unit_integrity", "terminology")

BATCH_SIZE = 50  # items per judge call; larger outputs are batched in time order
PASS_THRESHOLD = 4  # 1-5 rubric score at or above which a dimension passes
MAX_ATTEMPTS = 2  # initial call + 1 retry on unparseable JSON
_TEMPERATURE = 0.1  # judging should be near-deterministic
_SOFT_CAP_RATIO = 1.15  # mirrors plan.planner._SOFT_MAX_RATIO (validator tolerance)
_MERGE_HINT = re.compile(r"合并|merge", re.IGNORECASE)  # merge suggestions get a feasibility re-check


# ── Judge ───────────────────────────────────────────────────────────────────


class LLMJudge:
    """Scores step outputs with a rubric-prompted chat model (1-5 per dimension)."""

    def __init__(self, client: OpenAIClient | None, *, batch_size: int = BATCH_SIZE):
        self._client = client
        self._batch_size = batch_size

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[DimensionScore]:
        if self._client is None or output.skipped or output.error or not output.output:
            return []
        dimensions = PLAN_DIMENSIONS if case.step == "plan" else TRANSLATE_DIMENSIONS
        batches = [output.output[i : i + self._batch_size] for i in range(0, len(output.output), self._batch_size)]
        per_batch: list[dict[str, DimensionScore]] = []
        for index, batch in enumerate(batches):
            verdicts = self._judge_batch(case, fixture, batch, dimensions)
            if verdicts is None:
                logger.warning(
                    "LLM judge: batch %d/%d of case %s unparseable after retry — skipped",
                    index + 1,
                    len(batches),
                    case.name,
                )
                continue
            per_batch.append(verdicts)
        return _merge_batches(per_batch, dimensions)

    # ── One batch ───────────────────────────────────────────────────────────

    def _judge_batch(
        self,
        case: EvalCase,
        fixture: Fixture,
        batch: list[dict],
        dimensions: tuple[str, ...],
    ) -> dict[str, DimensionScore] | None:
        """Judge one batch; None when the model output stays unparseable."""
        prompt = _render_batch_prompt(case, fixture, batch)
        assert self._client is not None  # guarded by score()
        for attempt in range(MAX_ATTEMPTS):
            content, _usage = self._client.chat(
                [{"role": "system", "content": prompt}],
                temperature=_TEMPERATURE,
            )
            verdicts = _parse_verdicts(content, dimensions)
            if verdicts is not None:
                return _filter_verdicts(verdicts, case, batch)
            logger.info("LLM judge: unparseable response (attempt %d/%d)", attempt + 1, MAX_ATTEMPTS)
        return None


# ── Prompt rendering ────────────────────────────────────────────────────────


def _format_segment_line(unit_id: str, start: float, end: float, text: str, speaker: str = "") -> str:
    label = f"[{speaker}] " if speaker else ""  # speaker visible so the judge won't hallucinate switches
    return f"{unit_id} | {start:.2f}-{end:.2f} | {label}{text}"


def _batch_source_segments(step: str, fixture: Fixture, batch: list[dict]) -> list:
    """Fixture segments relevant to *batch* — full-context dumps only add noise.

    For plan, segments overlapping the batch's time range, plus one neighbor
    on each side as boundary context.  For translate, only the segments the
    batch's cues actually reference (via ``unit_id`` / ``merged_from``).
    """
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
        f"{cue.get('cue_id', '')}→{unit_ref}",
        float(cue.get("start", 0.0)),
        float(cue.get("end", 0.0)),
        str(cue.get("text", "")),
        str(cue.get("speaker") or ""),
    )


# ── Response parsing ────────────────────────────────────────────────────────


def _parse_verdicts(content: str, dimensions: tuple[str, ...]) -> dict[str, DimensionScore] | None:
    """Parse the judge JSON into per-dimension scores; None when unusable.

    A response counts as usable when at least one expected dimension parses
    with a numeric score; malformed dimensions are dropped individually.
    The current rubric schema is ``{score, summary, issues[]}``; the legacy
    ``reason`` / ``evidence`` keys are still honored as fallbacks.
    """
    fragment = extract_json_object(content)
    if fragment is None:
        return None
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    verdicts: dict[str, DimensionScore] = {}
    for dimension in dimensions:
        entry = data.get(dimension)
        if not isinstance(entry, dict):
            continue
        try:
            score = int(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        score = max(1, min(5, score))  # clamp to the rubric range
        issues = _parse_issues(entry.get("issues"))
        if score == 5:
            issues = []  # rubric contract: a clean score carries no issues
        if issues:
            evidence = list(dict.fromkeys(issue["unit_id"] for issue in issues))
        else:  # legacy fallback: old schema carried a bare evidence list
            evidence = [str(item) for item in entry.get("evidence") or []]
        verdicts[dimension] = DimensionScore(
            dimension=dimension,
            score=float(score),
            passed=score >= PASS_THRESHOLD,
            detail=str(entry.get("summary") or entry.get("reason") or ""),
            evidence=evidence,
            issues=issues,
        )
    return verdicts or None


def _parse_issues(raw: object) -> list[dict[str, str]]:
    """Validate the issues array; malformed entries are dropped individually."""
    if not isinstance(raw, list):
        return []
    issues: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        unit_id, problem = item.get("unit_id"), item.get("problem")
        if isinstance(unit_id, str) and isinstance(problem, str):
            issues.append({"unit_id": unit_id, "problem": problem})
    return issues


# ── Deterministic post-filter ────────────────────────────────────────────────


def _filter_verdicts(
    verdicts: dict[str, DimensionScore], case: EvalCase, batch: list[dict]
) -> dict[str, DimensionScore]:
    """Drop issues the rubric cannot enforce, before they reach the report.

    - *Out-of-batch references*: an issue may only name ids visible in this
      batch (plan: unit ids; translate: also cue ids and merge-chain ids).
    - *Infeasible merge suggestions* (plan only): when the problem text
      proposes merging units and names them, the merged duration is
      recomputed from the batch timestamps; beyond the soft cap the
      suggestion is impossible by design, so the issue is noise.
    """
    ids = {str(item.get("unit_id", "")) for item in batch}
    if case.step != "plan":
        for item in batch:
            ids.add(str(item.get("cue_id", "")))
            ids.update(str(m) for m in item.get("merged_from") or [])
    ids.discard("")
    soft_cap = float(case.params.get("max_duration", PlanConfig().max_duration)) * _SOFT_CAP_RATIO
    by_id = {str(u.get("unit_id", "")): u for u in batch}
    for dimension, score in verdicts.items():
        kept = []
        for issue in score.issues:
            if issue["unit_id"] not in ids:
                logger.info("LLM judge: dropping out-of-batch issue on %r (%s)", issue["unit_id"], case.name)
                continue
            if case.step == "plan" and _is_infeasible_merge(issue, by_id, soft_cap):
                logger.info("LLM judge: dropping infeasible merge suggestion on %r (%s)", issue["unit_id"], case.name)
                continue
            kept.append(issue)
        if kept != score.issues:  # rebuild so evidence matches the surviving issues
            verdicts[dimension] = DimensionScore(
                dimension=score.dimension,
                score=score.score,
                passed=score.passed,
                detail=score.detail,
                evidence=list(dict.fromkeys(issue["unit_id"] for issue in kept)),
                issues=kept,
            )
    return verdicts


def _is_infeasible_merge(issue: dict[str, str], batch_by_id: dict[str, dict], soft_cap: float) -> bool:
    """True when a merge suggestion names ≥2 batch units whose merged duration exceeds the soft cap."""
    if not _MERGE_HINT.search(issue["problem"]):
        return False
    named = _mentioned_ids(issue["problem"], batch_by_id)
    units = [batch_by_id[i] for i in dict.fromkeys([issue["unit_id"], *named]) if i in batch_by_id]
    if len(units) < 2:
        return False  # not verifiable from the batch — keep the issue
    merged = max(float(u.get("end", 0.0)) for u in units) - min(float(u.get("start", 0.0)) for u in units)
    return merged > soft_cap


def _mentioned_ids(text: str, batch_by_id: dict[str, dict]) -> list[str]:
    """Batch unit ids explicitly named in *text* (longest id wins at each position)."""
    if not batch_by_id:
        return []
    pattern = re.compile("|".join(re.escape(i) for i in sorted(batch_by_id, key=len, reverse=True)))
    return pattern.findall(text)


# ── Batch merging ───────────────────────────────────────────────────────────


def _merge_batches(per_batch: list[dict[str, DimensionScore]], dimensions: tuple[str, ...]) -> list[DimensionScore]:
    """Combine batch verdicts per dimension.

    The score is the median across batches — the lower middle value for even
    counts, so one lenient batch cannot sink a large case the way ``min`` did.
    The summary still comes from the lowest-scoring batch (worst problems stay
    visible); issues are merged across batches and deduplicated on
    ``(unit_id, problem)``.
    """
    merged: list[DimensionScore] = []
    for dimension in dimensions:
        batch_scores = [verdicts[dimension] for verdicts in per_batch if dimension in verdicts]
        if not batch_scores:
            continue
        ordered = sorted(batch_scores, key=lambda s: s.score)
        median = ordered[(len(ordered) - 1) // 2]
        worst = ordered[0]  # lowest-scoring batch owns the summary
        issues: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        evidence: list[str] = []
        for score in batch_scores:
            for issue in score.issues:
                key = (issue["unit_id"], issue["problem"])
                if key not in seen:
                    seen.add(key)
                    issues.append(issue)
            for item in score.evidence:
                if item not in evidence:
                    evidence.append(item)
        merged.append(
            DimensionScore(
                dimension=dimension,
                score=median.score,
                passed=median.score >= PASS_THRESHOLD,
                detail=worst.detail,
                evidence=evidence,
                issues=issues,
            )
        )
    return merged
