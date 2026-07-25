"""Segment step — pause-based semantic segmentation (+ segment.json export)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from light_core import logger
from light_subtitle import artifacts as sub_artifacts
from light_subtitle import export as export_module
from light_subtitle import segment
from light_subtitle.language import detect_source_lang

from ..reporting import StageStatus
from .progress import STAGE_SEGMENT

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _segment_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_SEGMENT, StageStatus.started, 0.0, "语义断句中...")


def _segment_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_SEGMENT, StageStatus.finished, 1.0, f"断句完成 ({len(orch.state.segments)} 段)")


def _run_segment(orch: Orchestrator) -> None:
    orch.state.source_lang = detect_source_lang(orch.state.words)
    logger.info(f"  Detected language: {orch.state.source_lang}")

    seg_config = orch.config.segment_config()
    orch.state.segments = segment.run(orch.state.words, seg_config.max_duration, seg_config.max_chars_per_line)
    logger.info(f"  Segment: {len(orch.state.segments)} segments")

    export_module.export_segments(
        orch.state.words,
        orch.state.segments,
        str(sub_artifacts.segment_json_path(orch.config.output_dir)),
    )
    # raw_source_cues is built by the compose step from composed units so
    # the English track shares the same unit graph as the translated track.
