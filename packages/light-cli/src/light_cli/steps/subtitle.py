"""Subtitle step — format cues for display (pace/split/punct-strip).

Also hosts the bilingual-detection sniffer because the subtitle step
itself consults it when deciding which tracks to format; the export step
reuses the same verdict when writing files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from light_models import SubtitleCue
from light_subtitle import artifacts as sub_artifacts
from light_subtitle import subtitle
from light_subtitle import translate as translate_pipeline
from light_subtitle.subtitle import strip_punct

from ..llm_client import client_from_config
from ..reporting import StageStatus
from .progress import STAGE_FORMAT

if TYPE_CHECKING:
    from light_llm.client import OpenAIClient

    from ..orchestrator import Orchestrator


def _format_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_FORMAT, StageStatus.started, 0.0, "格式化字幕中...")


def _format_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_FORMAT, StageStatus.finished, 1.0, "格式化完成")


def _llm_or_none(orch: Orchestrator) -> OpenAIClient | None:
    """Explicit LLM client for the pace compression pass; None disables it."""
    return client_from_config(orch.config) if orch.config.llm_api_key else None


def _format_source(orch: Orchestrator) -> list[SubtitleCue]:
    formatted, _usage = subtitle.run(
        orch.state.raw_source_cues,
        orch.config.layout_config(),
        llm=_llm_or_none(orch),
        transcript_words=orch.state.words,
    )
    return strip_punct.strip_chinese_punct(formatted)


def _format_target(orch: Orchestrator) -> list[SubtitleCue]:
    if not orch.state.translated_cues:
        return []
    translate_pipeline.attach_words_to_cues(orch.state.translated_cues, sub_artifacts.plan_dir(orch.config.output_dir))
    formatted, compress_usage = subtitle.run(
        orch.state.translated_cues,
        orch.config.layout_config(),
        llm=_llm_or_none(orch),
        transcript_words=orch.state.words,
    )
    if compress_usage:
        orch.usage_tracker.record("subtitle.compress", compress_usage)
    return strip_punct.strip_chinese_punct(formatted)


def _wants_bilingual_exports(orch: Orchestrator) -> bool:
    """True when this run should emit en+zh bilingual artifacts.

    The disk sniffing (bare and slug-prefixed names) is load-bearing: on
    ``--resume-from subtitle/export`` the bilingual flag comes from the
    ORIGINAL invocation, which is not persisted, so a resumed run can only
    discover that the earlier run was bilingual from artifacts left behind.
    """
    if orch.config.bilingual:
        return True
    if orch.config.target_lang is None:
        return False
    out = Path(orch.config.output_dir)
    slug = orch.config.slug or ""
    names = ("bilingual.ass", "bilingual.vtt", "en.srt", "en.vtt")
    for name in names:
        if sub_artifacts.find_sidecar(out, name, slug) is not None:
            return True
    return False


def _run_subtitle(orch: Orchestrator) -> None:
    if orch.config.target_lang is None:
        orch.state.formatted_source_cues = _format_source(orch)
        orch.state.formatted_target_cues = None
    elif _wants_bilingual_exports(orch):
        orch.state.formatted_source_cues = _format_source(orch)
        orch.state.formatted_target_cues = _format_target(orch) if orch.state.translated_cues else []
    else:
        orch.state.formatted_source_cues = None
        orch.state.formatted_target_cues = _format_target(orch)
