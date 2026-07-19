"""Translate steps — compose (cue planning), translate, retry, evaluate, join, save."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .. import artifacts, logger
from ..cue_builder import build_source_cues
from ..pipeline import translate as translate_pipeline
from ..pipeline.translate.join import join_cues, save_joined_units
from ..pipeline.translate.translate import run as translate_live
from ..reporting import StageStatus
from ..state_hydrate import hydrate_partial_cues, hydrate_plan_segments, sync_glossary
from ..usage.tracker import merge_token_usage, usage_delta
from .progress import STAGE_COMPOSE, STAGE_TRANSLATE

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _ensure_translate_ready(orch: Orchestrator) -> bool:
    if orch.config.target_lang is None:
        orch.emit_progress(STAGE_TRANSLATE, StageStatus.skipped, 1.0, "无需翻译")
        return False
    if not orch.config.llm_api_key:
        logger.warning("  Translation skipped (no LLM API key). Using source cues.")
        orch.emit_progress(STAGE_TRANSLATE, StageStatus.skipped, 1.0, "跳过翻译 (无 API key)")
        return False
    sync_glossary(orch)
    return True


def _translate_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_TRANSLATE, StageStatus.started, 0.0, "翻译中...")


def _translate_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_TRANSLATE, StageStatus.finished, 1.0, f"翻译完成 ({len(orch.state.translated_cues)} 条)")


def _plan_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_COMPOSE, StageStatus.started, 0.0, "规划字幕边界中...")


def _plan_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(
        STAGE_COMPOSE,
        StageStatus.finished,
        1.0,
        f"规划完成 ({len(orch.state.composed_segments)} 条 cue)",
    )


def _run_translate_compose(orch: Orchestrator) -> None:
    """Shared cue-planning step.

    Runs for both monolingual English and bilingual runs.  Builds
    ``orch.state.composed_segments`` from the pause-based ``segments`` via
    the LLM boundary planner, and rebuilds ``raw_source_cues`` from those
    planned units so the English track shares the same ``unit_id`` graph
    as the translated track.
    """
    plan_dir = artifacts.plan_dir(orch.config.output_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    if not orch.state.composed_segments:
        orch.state.composed_segments, plan_usage = translate_pipeline.plan_units(
            orch.state.segments,
            orch.config,
            plan_dir,
            progress=lambda f, m: orch.emit_progress(STAGE_COMPOSE, StageStatus.progress, f, m),
        )
        if plan_usage:
            orch.usage_tracker.record("translate.plan", plan_usage)
        # New plan → stale cached data no longer matches the unit graph.
        tx_dir = artifacts.translations_dir(orch.config.output_dir)
        for name in (artifacts.RAW_JSON, artifacts.PARTIAL_JSON, artifacts.USAGE_JSON):
            stale = tx_dir / name
            if stale.exists():
                stale.unlink()
                logger.info(f"  Discarded stale translation cache: {stale.name}")
        for name in (artifacts.SEGMENT_WORDS_JSON, artifacts.SEGMENT_WORDS_JOINED_JSON, artifacts.PLAN_JOINED_JSON):
            stale = plan_dir / name
            if stale.exists():
                stale.unlink()
                logger.info(f"  Discarded stale plan artifact: {stale.name}")
    translate_pipeline.save_segment_words(orch.state.composed_segments, plan_dir)
    orch.state.raw_source_cues = build_source_cues(orch.state.composed_segments, orch.state.source_lang)


def _run_translate_translate(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    tx_dir = artifacts.translations_dir(orch.config.output_dir)
    logger.info("  Translating...")
    orch.state.translated_cues, orch.state.translation_usage = translate_live(
        orch.state.composed_segments,
        orch.config,
        tx_dir,
        glossary=orch.state.merged_glossary,
        content_summary=orch.state.content_summary,
        progress=lambda f, m: orch.emit_progress(STAGE_TRANSLATE, StageStatus.progress, f, m),
    )
    if orch.state.translation_usage:
        orch.state.translation_usage_breakdown["translate.translate"] = dict(orch.state.translation_usage)
        orch.usage_tracker.record("translate.translate", orch.state.translation_usage)
    logger.info(f"  Translation: {len(orch.state.translated_cues)} translated cues")


def _run_translate_retry(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    before_usage = dict(orch.state.translation_usage or {})
    orch.state.translated_cues, orch.state.translation_usage = translate_pipeline.retry_missing(
        orch.state.translated_cues,
        orch.state.composed_segments,
        orch.config,
        orch.state.translation_usage,
        glossary=orch.state.merged_glossary,
        content_summary=orch.state.content_summary,
    )
    retry_usage = usage_delta(before_usage, orch.state.translation_usage)
    if retry_usage:
        orch.state.translation_usage_breakdown["translate.retry"] = retry_usage
        orch.usage_tracker.record("translate.retry", retry_usage)


def _run_translate_evaluate(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    orch.state.translated_cues, eval_breakdown = translate_pipeline.evaluate_and_refine(
        orch.state.translated_cues,
        orch.state.composed_segments,
        orch.config,
        artifacts.translations_dir(orch.config.output_dir),
        glossary=orch.state.merged_glossary,
        content_summary=orch.state.content_summary,
        progress=lambda f, m: orch.emit_progress(STAGE_TRANSLATE, StageStatus.progress, f, m),
    )
    if eval_breakdown:
        orch.state.translation_usage_breakdown.update(eval_breakdown)
        orch.usage_tracker.record_breakdown(eval_breakdown)
        for step_usage in eval_breakdown.values():
            merge_token_usage(orch.state.translation_usage, step_usage)


def _load_original_plan_units(orch: Orchestrator, plan_dir: Path) -> list:
    """Load the planner's ORIGINAL unit graph (plan.json + original word timing).

    The join pass must start from the un-joined graph: its input cues are
    1:1 with the original unit ids, and re-runs stay idempotent even when
    a previous join already wrote plan.joined.json.
    """
    units = translate_pipeline.load_plan_segments(plan_dir, orch.state.segments, orch.config)
    words_map = artifacts.load_segment_words_map(plan_dir, prefer_joined=False)
    if words_map:
        for u in units:
            word_dicts = words_map.get(u.unit_id)
            if word_dicts:
                u.words = [artifacts.word_from_dict(w) for w in word_dicts]
    return units


def _run_translate_join(orch: Orchestrator) -> None:
    """Join pass: LLM repairs dangling/flash translated cues (merge/shift)."""
    if not _ensure_translate_ready(orch):
        return
    plan_dir = artifacts.plan_dir(orch.config.output_dir)
    # Partial cues carry no words; attach EN timing from the ORIGINAL
    # graph (their unit ids predate any previous join run).
    translate_pipeline.attach_words_original(orch.state.translated_cues, plan_dir)
    result = join_cues(orch.state.translated_cues, _load_original_plan_units(orch, plan_dir), orch.config)
    orch.state.translated_cues = result.cues
    orch.state.composed_segments = result.units
    save_joined_units(result.units, plan_dir)
    if result.usage:
        orch.state.translation_usage_breakdown["translate.join"] = dict(result.usage)
        orch.usage_tracker.record("translate.join", result.usage)


def _run_translate_save(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    translate_pipeline.save_artifacts(
        orch.state.translated_cues,
        orch.state.raw_source_cues,
        orch.state.translation_usage,
        artifacts.translations_dir(orch.config.output_dir),
        breakdown=orch.state.translation_usage_breakdown or None,
        segments=orch.state.composed_segments,
    )


def _hydrate_translate_mid(orch: Orchestrator) -> None:
    hydrate_plan_segments(orch)
    hydrate_partial_cues(orch)
