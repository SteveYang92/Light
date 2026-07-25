"""Annotate step — LLM-generated secondary subtitle annotations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from light_core import logger
from light_subtitle import annotate as annotate_pipeline
from light_subtitle import artifacts as sub_artifacts

from ..llm_client import client_from_config
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
    output_dir = Path(orch.config.output_dir)
    orch.state.translated_cues, usage = annotate_pipeline.generate_annotations(
        orch.state.translated_cues,
        orch.state.composed_segments or orch.state.segments,
        llm=client_from_config(orch.config) if orch.config.llm_api_key else None,
        glossary=orch.state.merged_glossary,
        content_summary=orch.state.content_summary,
        domain_context_path=output_dir / "transcript_correct" / "domain_context.json",
        usage_path=sub_artifacts.annotations_dir(output_dir) / sub_artifacts.USAGE_JSON,
        progress=lambda f, m: orch.emit_progress(STAGE_ANNOTATE, StageStatus.progress, f, m),
    )
    if usage:
        orch.usage_tracker.record("annotate", usage)
    orch.state.annotations = {c.unit_id: c.annotation for c in orch.state.translated_cues if c.annotation}
    sub_artifacts.save_annotations(orch.config.output_dir, orch.state.annotations)
    logger.info(f"  Annotations: {len(orch.state.annotations)} terms annotated")
