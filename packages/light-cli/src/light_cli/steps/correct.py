"""Transcript-correction step — LLM-based ASR error fixes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from light_asr_polish import correct

from ..llm_client import client_from_config
from ..reporting import StageStatus
from .progress import STAGE_CORRECT

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _correct_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_CORRECT, StageStatus.started, 0.0, "转录矫正中...")


def _correct_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_CORRECT, StageStatus.finished, 1.0, "转录矫正完成")


def _run_correct(orch: Orchestrator) -> None:
    if not orch.state.words or not orch.config.llm_api_key or not orch.config.correct_enabled:
        return
    client = client_from_config(orch.config)
    orch.state.words, usage = correct(orch.state.words, client, orch.config.output_dir)
    if usage and isinstance(usage.get("breakdown"), dict):
        orch.usage_tracker.record_breakdown(usage["breakdown"])
