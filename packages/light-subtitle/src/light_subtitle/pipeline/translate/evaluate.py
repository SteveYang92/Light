"""Translation quality evaluator — LLM-driven scoring with multi-dimensional assessment.

Runs after initial translation to identify low-quality segments for revision.
Scores each translation on accuracy, fluency, consistency, and compression.

Usage::

    from .evaluate import evaluate_translations

    scores = evaluate_translations(translated_cues, source_segments, config)
    low_score_ids = {s.unit_id for s in scores if s.overall < config.quality_threshold}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from light_models import Segment, SubtitleCue, covered_source_text

if TYPE_CHECKING:
    from ...config import SubtitleConfig
from ... import logger
from ...llm.client import OpenAIClient, merge_token_usage
from ...llm.prompts import render_prompt

# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class QualityScore:
    """Per-cue translation quality assessment."""

    unit_id: str
    accuracy: float  # 0.0–1.0: semantic fidelity, no information loss
    fluency: float  # 0.0–1.0: natural language, no translationese
    consistency: float  # 0.0–1.0: term consistency within batch
    compression: float  # 0.0–1.0: fits time window without losing meaning
    overall: float  # weighted average
    issues: list[str] = field(default_factory=list)  # diagnostic messages
    suggestion: str = ""  # recommended fix from evaluator


# ── Configuration ───────────────────────────────────────────────────────────

# Batch size for evaluation — balance between detail and cost.
EVAL_BATCH_SIZE = 25

# Weights for overall score calculation.
WEIGHTS = {
    "accuracy": 0.40,
    "fluency": 0.30,
    "consistency": 0.15,
    "compression": 0.15,
}


# ── Public API ──────────────────────────────────────────────────────────────


def evaluate_translations(
    translated_cues: list[SubtitleCue],
    source_segments: list[Segment],
    config: SubtitleConfig,
) -> tuple[list[QualityScore], dict | None]:
    """Evaluate translation quality for all translated cues.

    Splits into batches of EVAL_BATCH_SIZE, sends source+translation pairs
    to the LLM for scoring, and returns structured QualityScore objects.

    Returns empty list if evaluation is disabled or no LLM key is available.
    """
    if not config.evaluate_enabled or not config.llm_api_key:
        return [], None

    # Build lookup map: unit_id → source_text
    source_map: dict[str, str] = {s.unit_id: s.source_text for s in source_segments}

    # Pair cues with their source texts (only those with valid source).
    pairs: list[tuple[SubtitleCue, str]] = []
    for cue in translated_cues:
        src = covered_source_text(cue, source_map)
        if src:
            pairs.append((cue, src))

    if not pairs:
        return [], None

    all_scores: list[QualityScore] = []
    total_usage: dict = {}

    # Evaluate in batches.
    for batch_idx in range(0, len(pairs), EVAL_BATCH_SIZE):
        batch = pairs[batch_idx : batch_idx + EVAL_BATCH_SIZE]
        batch_scores, usage = _evaluate_batch(batch, config, batch_idx // EVAL_BATCH_SIZE)
        all_scores.extend(batch_scores)
        merge_token_usage(total_usage, usage)

    return all_scores, total_usage or None


# ── Batch evaluation ────────────────────────────────────────────────────────


def _evaluate_batch(
    pairs: list[tuple[SubtitleCue, str]],
    config: SubtitleConfig,
    batch_num: int,
) -> tuple[list[QualityScore], dict]:
    """Send a batch of source+translation pairs for LLM quality scoring."""
    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    prompt = _build_eval_prompt(pairs, config)
    messages = [{"role": "user", "content": prompt}]

    try:
        response, usage = client.chat(messages, temperature=0.1)
    except Exception as e:
        logger.warning(f"    ⚠ Evaluation batch {batch_num} failed: {e}")
        return [], {}

    return _parse_eval_response(response, pairs), usage


# ── Prompt construction ─────────────────────────────────────────────────────


def _build_eval_prompt(
    pairs: list[tuple[SubtitleCue, str]],
    config: SubtitleConfig,
) -> str:
    """Build the evaluation prompt with source + translation pairs."""
    items: list[dict] = []
    for cue, src_text in pairs:
        items.append(
            {
                "unit_id": cue.unit_id,
                "duration": round(cue.end - cue.start, 1),
                "source": src_text,
                "translation": cue.text.replace("\n", "\\n"),
            }
        )
    return render_prompt(
        "evaluate.j2",
        target_lang=config.target_lang,
        items=items,
        glossary=config.glossary,
        content_summary=config.content_summary,
    )


# ── Response parsing ────────────────────────────────────────────────────────


def _parse_eval_response(
    response: str,
    pairs: list[tuple[SubtitleCue, str]],
) -> list[QualityScore]:
    """Parse LLM evaluation response into QualityScore objects."""
    response = response.strip()

    # Extract JSON array from response.
    json_match = re.search(r"\[([\s\S]*)\]", response)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError):
            logger.warning("    ⚠ Evaluation: could not parse LLM JSON response, skipping")
            return []
    else:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("    ⚠ Evaluation: could not parse LLM response, skipping")
            return []

    if not isinstance(data, list):
        return []

    valid_ids = {cue.unit_id for cue, _ in pairs}
    scores: list[QualityScore] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("unit_id", ""))
        if uid not in valid_ids:
            continue

        accuracy = _clamp_float(item.get("accuracy", 0.5))
        fluency = _clamp_float(item.get("fluency", 0.5))
        consistency = _clamp_float(item.get("consistency", 0.5))
        compression = _clamp_float(item.get("compression", 0.5))

        overall = (
            accuracy * WEIGHTS["accuracy"]
            + fluency * WEIGHTS["fluency"]
            + consistency * WEIGHTS["consistency"]
            + compression * WEIGHTS["compression"]
        )

        raw_issues = item.get("issues", [])
        if isinstance(raw_issues, list):
            issues = [str(i) for i in raw_issues if str(i).strip()]
        else:
            issues = []

        scores.append(
            QualityScore(
                unit_id=uid,
                accuracy=round(accuracy, 2),
                fluency=round(fluency, 2),
                consistency=round(consistency, 2),
                compression=round(compression, 2),
                overall=round(overall, 2),
                issues=issues,
                suggestion=str(item.get("suggestion", "")).strip(),
            )
        )

    return scores


# ── Build score summary for refine ──────────────────────────────────────────


def get_low_score_cues(
    scores: list[QualityScore],
    threshold: float,
) -> set[str]:
    """Return unit_ids that scored below threshold."""
    return {s.unit_id for s in scores if s.overall < threshold}


def scores_to_dict(scores: list[QualityScore]) -> list[dict]:
    """Serialize scores for saving to quality.json."""
    return [
        {
            "unit_id": s.unit_id,
            "accuracy": s.accuracy,
            "fluency": s.fluency,
            "consistency": s.consistency,
            "compression": s.compression,
            "overall": s.overall,
            "issues": s.issues,
            "suggestion": s.suggestion,
        }
        for s in scores
    ]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _clamp_float(value) -> float:
    """Clamp a value to 0.0–1.0, defaulting to 0.5 on error."""
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (ValueError, TypeError):
        return 0.5
