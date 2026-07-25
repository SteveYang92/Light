"""Punctuation-restoration step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from light_asr_polish import restore_punct

from ..llm_client import client_from_config
from ..reporting import StageStatus
from .progress import STAGE_PUNCT

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _punct_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_PUNCT, StageStatus.started, 0.0, "恢复标点中...")


def _punct_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_PUNCT, StageStatus.finished, 1.0, "标点恢复完成")


def _run_punct(orch: Orchestrator) -> None:
    if not orch.state.words or not orch.config.llm_api_key:
        return
    client = client_from_config(orch.config)
    orch.state.words, usage = restore_punct(orch.state.words, client, orch.config.output_dir)
    if usage:
        orch.usage_tracker.record("punct", usage)
