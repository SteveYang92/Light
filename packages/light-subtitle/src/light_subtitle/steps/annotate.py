"""Annotate step — LLM-generated secondary subtitle annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import logger
from ..pipeline import annotate as annotate_pipeline
from ..reporting import StageStatus
from .progress import STAGE_ANNOTATE

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _annotate_progress_start(orch: Orchestrator) -> None:
    if orch.config.annotate and orch.state.translated_cues:
        orch.emit_progress(STAGE_ANNOTATE, StageStatus.started, 0.0, "生成注解中...")


def _annotate_progress_end(orch: Orchestrator) -> None:
    if not orch.config.annotate or not orch.state.translated_cues:
        orch.emit_progress(STAGE_ANNOTATE, StageStatus.finished, 1.0, "无需注解")
    else:
        orch.emit_progress(STAGE_ANNOTATE, StageStatus.finished, 1.0, f"注解完成 ({len(orch.state.annotations)} 条)")


def _run_annotate(orch: Orchestrator) -> None:
    if not orch.config.annotate or not orch.state.translated_cues:
        return

    logger.info("  Generating annotations...")
    orch.state.translated_cues, usage = annotate_pipeline.generate_annotations(
        orch.state.translated_cues,
        orch.state.composed_segments or orch.state.segments,
        orch.config,
        orch.config.output_dir,
        glossary=orch.state.merged_glossary,
        content_summary=orch.state.content_summary,
        progress=lambda f, m: orch.emit_progress(STAGE_ANNOTATE, StageStatus.progress, f, m),
    )
    if usage:
        orch.usage_tracker.record("annotate", usage)
    orch.state.annotations = {c.unit_id: c.annotation for c in orch.state.translated_cues if c.annotation}
    logger.info(f"  Annotations: {len(orch.state.annotations)} terms annotated")
