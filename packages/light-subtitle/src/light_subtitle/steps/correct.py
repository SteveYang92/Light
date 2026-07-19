"""Transcript-correction step — LLM-based ASR error fixes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..pipeline.transcript_correct import correct_transcript
from ..reporting import StageStatus
from .progress import STAGE_CORRECT

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _correct_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_CORRECT, StageStatus.started, 0.0, "转录矫正中...")


def _correct_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_CORRECT, StageStatus.finished, 1.0, "转录矫正完成")


def _run_correct(orch: Orchestrator) -> None:
    orch.state.words, usage = correct_transcript(orch.state.words, orch.config, orch.config.output_dir)
    if usage and isinstance(usage.get("breakdown"), dict):
        orch.usage_tracker.record_breakdown(usage["breakdown"])
