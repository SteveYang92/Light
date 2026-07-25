"""Translate steps — compose (cue planning), translate, retry, evaluate, join, save."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from light_core import logger
from light_llm.usage.tracker import merge_token_usage, usage_delta
from light_models import word_from_dict
from light_subtitle import artifacts as sub_artifacts
from light_subtitle import translate as translate_pipeline
from light_subtitle.cue_builder import build_source_cues
from light_subtitle.translate.join import join_cues, save_joined_units
from light_subtitle.translate.translate import run as translate_live

from ..llm_client import client_from_config
from ..reporting import StageStatus
from ..state_hydrate import hydrate_partial_cues, hydrate_plan_segments, sync_glossary
from .progress import STAGE_COMPOSE, STAGE_TRANSLATE

if TYPE_CHECKING:
    from light_llm.client import OpenAIClient

    from ..orchestrator import Orchestrator


def _llm_or_none(orch: Orchestrator) -> OpenAIClient | None:
    """Explicit LLM client for subtitle-domain calls; None gates the LLM off."""
    return client_from_config(orch.config) if orch.config.llm_api_key else None


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
    plan_dir = sub_artifacts.plan_dir(orch.config.output_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    if not orch.state.composed_segments:
        orch.state.composed_segments, plan_usage = translate_pipeline.plan_units(
            orch.state.segments,
            orch.config.plan_config(),
            plan_dir,
            llm=_llm_or_none(orch),
            progress=lambda f, m: orch.emit_progress(STAGE_COMPOSE, StageStatus.progress, f, m),
        )
        if plan_usage:
            orch.usage_tracker.record("translate.plan", plan_usage)
        # New plan → stale cached data no longer matches the unit graph.
        tx_dir = sub_artifacts.translations_dir(orch.config.output_dir)
        for name in (sub_artifacts.RAW_JSON, sub_artifacts.PARTIAL_JSON, sub_artifacts.USAGE_JSON):
            stale = tx_dir / name
            if stale.exists():
                stale.unlink()
                logger.info(f"  Discarded stale translation cache: {stale.name}")
        for name in (
            sub_artifacts.SEGMENT_WORDS_JSON,
            sub_artifacts.SEGMENT_WORDS_JOINED_JSON,
            sub_artifacts.PLAN_JOINED_JSON,
        ):
            stale = plan_dir / name
            if stale.exists():
                stale.unlink()
                logger.info(f"  Discarded stale plan artifact: {stale.name}")
    translate_pipeline.save_segment_words(orch.state.composed_segments, plan_dir)
    orch.state.raw_source_cues = build_source_cues(orch.state.composed_segments, orch.state.source_lang)


def _run_translate_translate(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    tx_dir = sub_artifacts.translations_dir(orch.config.output_dir)
    logger.info("  Translating...")
    orch.state.translated_cues, orch.state.translation_usage = translate_live(
        orch.state.composed_segments,
        orch.config.translate_config(),
        tx_dir,
        llm=client_from_config(orch.config),
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
        orch.config.translate_config(),
        orch.state.translation_usage,
        llm=client_from_config(orch.config),
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
        orch.config.translate_config(),
        sub_artifacts.translations_dir(orch.config.output_dir),
        llm=client_from_config(orch.config),
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
    units = translate_pipeline.load_plan_segments(
        plan_dir, orch.state.segments, orch.config.plan_config(), llm=_llm_or_none(orch)
    )
    words_map = sub_artifacts.load_segment_words_map(plan_dir, prefer_joined=False)
    if words_map:
        for u in units:
            word_dicts = words_map.get(u.unit_id)
            if word_dicts:
                u.words = [word_from_dict(w) for w in word_dicts]
    return units


def _run_translate_join(orch: Orchestrator) -> None:
    """Join pass: LLM repairs dangling/flash translated cues (merge/shift)."""
    if not _ensure_translate_ready(orch):
        return
    plan_dir = sub_artifacts.plan_dir(orch.config.output_dir)
    # Partial cues carry no words; attach EN timing from the ORIGINAL
    # graph (their unit ids predate any previous join run).
    translate_pipeline.attach_words_original(orch.state.translated_cues, plan_dir)
    result = join_cues(
        orch.state.translated_cues,
        _load_original_plan_units(orch, plan_dir),
        orch.config.translate_config(),
        llm=client_from_config(orch.config),
    )
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
        sub_artifacts.translations_dir(orch.config.output_dir),
        breakdown=orch.state.translation_usage_breakdown or None,
        segments=orch.state.composed_segments,
    )


def _hydrate_translate_mid(orch: Orchestrator) -> None:
    hydrate_plan_segments(orch)
    hydrate_partial_cues(orch)
