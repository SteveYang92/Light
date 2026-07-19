"""Punctuation-restoration step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..pipeline.punct_restore import restore_punctuation
from .progress import STAGE_PUNCT

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _punct_progress_start(orch: Orchestrator) -> None:
    orch._progress(STAGE_PUNCT, 0.0, "恢复标点中...")


def _punct_progress_end(orch: Orchestrator) -> None:
    orch._progress(STAGE_PUNCT, 1.0, "标点恢复完成")


def _run_punct(orch: Orchestrator) -> None:
    orch.state.words, usage = restore_punctuation(orch.state.words, orch.config, orch.config.output_dir)
    if usage:
        orch.usage_tracker.record("punct", usage)
