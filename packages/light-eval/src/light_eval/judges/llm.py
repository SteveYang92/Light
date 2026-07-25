"""LLM judge — rubric-based 1-5 scoring via an OpenAI-compatible chat model.

One judge instance scores one case's step output along the rubric
dimensions for its step:

- ``plan``: ``boundary_quality`` (boundaries land on clause / phrase-group
  edges) and ``split_necessity`` (neither over-fragmented nor over-merged).
- ``translate``: ``faithfulness`` (no omission / addition),
  ``naturalness`` (spoken, not translationese), ``unit_integrity`` (cue ↔
  source unit correspondence) and ``terminology`` (glossary consistency).

Outputs longer than ``batch_size`` items are judged in time-ordered
batches; the per-dimension score is the *minimum* across batches
(conservative), the summary comes from the lowest-scoring batch, and
per-unit issues are merged across batches with duplicates dropped.  When
the client is ``None`` (no API key) the judge returns an empty list —
reports then show the LLM judge as skipped.
"""

from __future__ import annotations

import json
import logging

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
                return verdicts
            logger.info("LLM judge: unparseable response (attempt %d/%d)", attempt + 1, MAX_ATTEMPTS)
        return None


# ── Prompt rendering ────────────────────────────────────────────────────────


def _format_segment_line(unit_id: str, start: float, end: float, text: str) -> str:
    return f"{unit_id} | {start:.2f}-{end:.2f} | {text}"


def _render_batch_prompt(case: EvalCase, fixture: Fixture, batch: list[dict]) -> str:
    source_lines = "\n".join(
        _format_segment_line(seg.unit_id, seg.start, seg.end, seg.source_text) for seg in fixture.segments
    )
    if case.step == "plan":
        unit_lines = "\n".join(
            _format_segment_line(
                str(unit.get("unit_id", "")),
                float(unit.get("start", 0.0)),
                float(unit.get("end", 0.0)),
                str(unit.get("text", "")),
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


# ── Batch merging ───────────────────────────────────────────────────────────


def _merge_batches(per_batch: list[dict[str, DimensionScore]], dimensions: tuple[str, ...]) -> list[DimensionScore]:
    """Combine batch verdicts per dimension.

    The score is the minimum across batches (conservative); the summary comes
    from the lowest-scoring batch rather than being concatenated; issues are
    merged across batches and deduplicated on ``(unit_id, problem)``.
    """
    merged: list[DimensionScore] = []
    for dimension in dimensions:
        batch_scores = [verdicts[dimension] for verdicts in per_batch if dimension in verdicts]
        if not batch_scores:
            continue
        best = min(batch_scores, key=lambda s: s.score)
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
                score=best.score,
                passed=best.score >= PASS_THRESHOLD,
                detail=best.detail,
                evidence=evidence,
                issues=issues,
            )
        )
    return merged
