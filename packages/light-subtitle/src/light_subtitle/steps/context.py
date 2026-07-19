"""Context step — glossary + content-summary extraction before translation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import logger
from ..pipeline import context_prep as context_prep_pipeline
from ..reporting import StageStatus
from ..state_hydrate import sync_glossary
from .progress import STAGE_CONTEXT

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _context_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_CONTEXT, StageStatus.started, 0.0, "提取翻译上下文中...")


def _context_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_CONTEXT, StageStatus.finished, 1.0, "翻译上下文就绪")


def _run_context(orch: Orchestrator) -> None:
    if orch.config.context_prep_enabled:
        result, usage = context_prep_pipeline.prepare_context(
            orch.state.segments,
            orch.config,
            orch.config.output_dir,
        )
        orch.state.auto_glossary = result.glossary
        orch.state.content_summary = result.summary
        if usage:
            orch.usage_tracker.record("context", usage)

    sync_glossary(orch, recompute=True)

    if orch.state.content_summary and not orch.config.speaker_names:
        speakers = orch.state.content_summary.get("speakers")
        if isinstance(speakers, dict):
            orch.state.speaker_names = {str(k): str(v) for k, v in speakers.items()}

    logger.info(f"  Translation context: {len(orch.state.merged_glossary)} glossary terms")
