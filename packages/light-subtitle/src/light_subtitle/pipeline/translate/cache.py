"""Translation cache & artifact persistence — raw/partial caches, plan resume, word re-attach."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from light_models import Segment, SubtitleCue

from ... import artifacts, logger
from ...config import SubtitleConfig
from ...usage.tracker import save_step_usage
from .. import export
from .. import plan as plan_pipeline
from .checkpoint import segment_graph_fingerprint


def plan_units(
    segments: list[Segment],
    config: SubtitleConfig,
    plan_dir: Path,
    *,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Segment], dict | None]:
    """Plan cue units with the LLM boundary planner; persist ``plan/plan.json``."""
    plan_dir.mkdir(parents=True, exist_ok=True)
    translation_segments, plan_usage = plan_pipeline.run(segments, config, plan_dir, progress=progress)
    if plan_usage:
        save_step_usage(plan_dir / artifacts.USAGE_JSON, plan_usage)
    return translation_segments, plan_usage


def load_plan_segments(plan_dir: Path, segments: list[Segment], config: SubtitleConfig) -> list[Segment]:
    """Rebuild translation units from ``plan/plan.json`` when resuming mid-translate.

    Falls back to re-running the planner (LLM calls) only when plan.json
    is absent.
    """
    units = plan_pipeline.load_plan_units(plan_dir)
    if units is not None:
        return units
    return plan_units(segments, config, plan_dir)[0]


def load_cached_translation(
    tx_dir: Path,
    config: SubtitleConfig,
    current_segments: list[Segment] | None = None,
) -> tuple[list[SubtitleCue], dict | None]:
    """Load translated cues and usage from cached raw.json / usage.json.

    When *current_segments* is provided, verifies the plan fingerprint
    against the stored one; returns ``([], None)`` on mismatch so the
    pipeline falls back to a full re-translation.

    If ``plan/segment_words.json`` exists, word timing is re-attached to each
    cue by matching ``unit_id``, enabling word-boundary alignment in the pace step.
    """
    raw_path = tx_dir / artifacts.RAW_JSON
    if not raw_path.exists():
        return [], None

    if current_segments is not None:
        fp_current = segment_graph_fingerprint(current_segments)
        fp_path = tx_dir / artifacts.FINGERPRINT_JSON
        if fp_path.exists():
            fp_stored = artifacts.read_fingerprint(fp_path)
            if fp_stored != fp_current:
                logger.warning("  Cached translation stale (plan changed) — re-translating")
                return [], None

    translated_cues = artifacts.read_raw_cues(raw_path, default_lang=config.target_lang)

    plan_dir = artifacts.plan_dir(tx_dir.parent)
    attach_words_to_cues(translated_cues, plan_dir)
    _attach_speakers_from_plan(translated_cues, plan_dir)

    usage: dict | None = None
    usage_path = tx_dir / artifacts.USAGE_JSON
    if usage_path.exists():
        with open(usage_path) as f:
            usage = json.load(f)
    logger.info(f"  Translation (cached): {len(translated_cues)} cues from raw.json")
    return translated_cues, usage


def save_artifacts(
    translated_cues: list[SubtitleCue],
    source_cues: list[SubtitleCue],
    usage: dict | None,
    tx_dir: Path,
    breakdown: dict[str, dict] | None = None,
    segments: list[Segment] | None = None,
) -> None:
    """Save raw.json, source.json, and usage.json artifacts.

    When *segments* is provided, also writes a plan fingerprint so resume
    can detect plan changes and discard stale cached translations.
    """
    export.export_raw_cues(translated_cues, str(tx_dir / artifacts.RAW_JSON))
    export.export_raw_cues(source_cues, str(tx_dir / artifacts.SOURCE_JSON))
    if usage:
        payload = dict(usage)
        if breakdown:
            payload["breakdown"] = breakdown
        logger.info(
            f"  Tokens: {payload.get('total_tokens', '?')} "
            f"(prompt: {payload.get('prompt_tokens', '?')}, "
            f"completion: {payload.get('completion_tokens', '?')})"
        )
        export.export_json_file(payload, str(tx_dir / artifacts.USAGE_JSON))
    if segments:
        fp = segment_graph_fingerprint(segments)
        artifacts.write_fingerprint(tx_dir / artifacts.FINGERPRINT_JSON, fp)


def attach_words_to_cues(cues: list[SubtitleCue], plan_dir: Path, *, prefer_joined: bool = True) -> None:
    """Re-attach word timing from ``plan/segment_words[.joined].json``.

    Uses ``unit_id`` plus ``merged_from`` (in chain order) so cues that
    absorbed other units get the full ASR span.  ``prefer_joined`` reads
    the joined graph when present (post-join resumes); the join pass
    itself must attach from the original graph (``prefer_joined=False``)
    because its 1:1 cues still reference the original unit ids.
    """
    seg_words_map = artifacts.load_segment_words_map(plan_dir, prefer_joined=prefer_joined)
    if seg_words_map is None:
        return
    for cue in cues:
        unit_chain = [cue.unit_id, *cue.merged_from]
        words = artifacts.words_from_unit_chain(unit_chain, seg_words_map)
        if words:
            cue.words = words


def attach_words_original(cues: list[SubtitleCue], plan_dir: Path) -> None:
    """Attach word timing from the ORIGINAL (pre-join) graph — join pass input."""
    attach_words_to_cues(cues, plan_dir, prefer_joined=False)


def save_segment_words(translation_segments: list[Segment], plan_dir: Path) -> None:
    """Save per-unit word-level timing to ``plan/segment_words.json``."""
    plan_dir.mkdir(parents=True, exist_ok=True)
    artifacts.write_segment_words(artifacts.resolve_segment_words_path(plan_dir), translation_segments)


def _attach_speakers_from_plan(cues: list[SubtitleCue], plan_dir: Path) -> None:
    """Fill empty cue speakers from plan metadata on resume (joined graph preferred)."""
    plan_path = plan_dir / artifacts.PLAN_JOINED_JSON
    if not plan_path.exists():
        plan_path = plan_dir / artifacts.PLAN_JSON
    if not plan_path.exists():
        return
    with open(plan_path, encoding="utf-8") as f:
        data = json.load(f)
    speaker_by_unit = {
        item["unit_id"]: item.get("speaker", "") for item in data.get("units", []) if isinstance(item, dict)
    }
    for cue in cues:
        if not cue.speaker:
            cue.speaker = speaker_by_unit.get(cue.unit_id, "")
