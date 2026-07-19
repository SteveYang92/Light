"""Segment step — pause-based semantic segmentation (+ segment.json export)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import artifacts, logger
from ..language import detect_source_lang
from ..pipeline import export as export_module
from ..pipeline import segment
from .progress import STAGE_SEGMENT

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _segment_progress_start(orch: Orchestrator) -> None:
    orch._progress(STAGE_SEGMENT, 0.0, "语义断句中...")


def _segment_progress_end(orch: Orchestrator) -> None:
    orch._progress(STAGE_SEGMENT, 1.0, f"断句完成 ({len(orch.state.segments)} 段)")


def _run_segment(orch: Orchestrator) -> None:
    orch.state.source_lang = detect_source_lang(orch.state.words)
    logger.info(f"  Detected language: {orch.state.source_lang}")

    orch.state.segments = segment.run(orch.state.words, orch.config.max_duration)
    logger.info(f"  Segment: {len(orch.state.segments)} segments")

    export_module.export_segments(
        orch.state.words,
        orch.state.segments,
        str(artifacts.segment_json_path(orch.config.output_dir)),
    )
    # raw_source_cues is built by the compose step from composed units so
    # the English track shares the same unit graph as the translated track.
