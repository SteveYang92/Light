"""Declarative pipeline step registry — single source of truth for step IDs and behavior.

Only declarative assembly lives here: step ids, metadata, and the ordered
definition list.  The step run/progress implementations are grouped by
stage under :mod:`light_subtitle.steps`; resume-time state hydration lives
in :mod:`light_subtitle.state_hydrate`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from . import artifacts
from .config import AsrEngine, SubtitleConfig
from .pipeline.asr.artifacts import asr_words_path, audio_wav_path
from .state_hydrate import (
    hydrate_annotate_inputs,
    hydrate_asr_audio,
    hydrate_plan_segments,
    hydrate_segments_from_disk,
    hydrate_subtitle_export,
    hydrate_transcript_words,
    hydrate_words_after_correct,
    hydrate_words_after_punct,
)
from .steps import annotate, asr, context, correct, export, punct, segment, subtitle, translate

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

StepRunner = Callable[["Orchestrator"], None]
ArtifactFn = Callable[[SubtitleConfig], tuple[Path, ...]]


class StepId(StrEnum):
    ASR_EXTRACT = "asr.extract"
    ASR_TRANSCRIBE = "asr.transcribe"
    ASR_ALIGN = "asr.align"
    ASR_DIARIZE = "asr.diarize"
    CORRECT = "correct"
    PUNCT = "punct"
    SEGMENT = "segment"
    CONTEXT = "context"
    TRANSLATE_COMPOSE = "translate.compose"
    TRANSLATE_TRANSLATE = "translate.translate"
    TRANSLATE_RETRY = "translate.retry"
    TRANSLATE_EVALUATE = "translate.evaluate"
    TRANSLATE_JOIN = "translate.join"
    TRANSLATE_SAVE = "translate.save"
    ANNOTATE = "annotate"
    SUBTITLE = "subtitle"
    EXPORT = "export"


@dataclass(frozen=True)
class StepDefinition:
    """Metadata and handlers for one pipeline step.

    ``hydrate`` loads the state THIS step's input artifacts produce
    (named after the preceding steps' outputs — e.g. ``correct``'s
    hydrate loads ``transcript.json`` written by ASR).  On resume,
    ``hydrate_state`` replays handlers for all steps before the resume
    target, plus the target's own, so its inputs are in place.
    """

    id: StepId
    run: StepRunner
    artifacts: ArtifactFn
    progress_start: StepRunner | None = None
    progress_end: StepRunner | None = None
    hydrate: StepRunner | None = None
    enabled: Callable[[SubtitleConfig], bool] = lambda _c: True


def _out(config: SubtitleConfig) -> Path:
    return Path(config.output_dir)


def _subtitle_artifact_paths(config: SubtitleConfig) -> tuple[Path, ...]:
    if config.target_lang:
        return (
            artifacts.raw_cues_path(_out(config)),
            artifacts.segment_json_path(_out(config)),
            artifacts.plan_json_path(_out(config)),
        )
    return (
        artifacts.segment_json_path(_out(config)),
        artifacts.plan_json_path(_out(config)),
    )


# ── Step definition list ────────────────────────────────────────────────────────


def build_step_definitions(config: SubtitleConfig) -> list[StepDefinition]:
    """Return ordered step definitions for *config* (before enabled filtering)."""
    return [
        StepDefinition(
            id=StepId.ASR_EXTRACT,
            run=asr._run_asr_extract,
            artifacts=lambda _c: (),
            progress_start=asr._asr_progress_start,
            hydrate=hydrate_asr_audio,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.ASR_TRANSCRIBE,
            run=asr._run_asr_transcribe,
            artifacts=lambda c: (audio_wav_path(c.output_dir),),
            progress_end=asr._asr_progress_end,
            hydrate=hydrate_asr_audio,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.ASR_ALIGN,
            run=asr._run_asr_align,
            artifacts=lambda c: (asr_words_path(c), audio_wav_path(c.output_dir)),
            progress_end=asr._asr_progress_end,
            hydrate=asr._hydrate_asr_align,
            enabled=lambda c: c.asr == AsrEngine.WHISPER_CPP,
        ),
        StepDefinition(
            id=StepId.ASR_DIARIZE,
            run=asr._run_asr_diarize,
            artifacts=lambda c: (asr_words_path(c), audio_wav_path(c.output_dir)),
            progress_end=asr._asr_progress_end,
            hydrate=asr._hydrate_asr_align,
            enabled=lambda c: c.diarize,
        ),
        StepDefinition(
            id=StepId.CORRECT,
            run=correct._run_correct,
            artifacts=lambda c: (artifacts.transcript_path(_out(c)),),
            progress_start=correct._correct_progress_start,
            progress_end=correct._correct_progress_end,
            hydrate=hydrate_transcript_words,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.PUNCT,
            run=punct._run_punct,
            artifacts=lambda c: (artifacts.transcript_path(_out(c)),),
            progress_start=punct._punct_progress_start,
            progress_end=punct._punct_progress_end,
            hydrate=hydrate_words_after_correct,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.SEGMENT,
            run=segment._run_segment,
            artifacts=lambda c: (artifacts.transcript_path(_out(c)),),
            progress_start=segment._segment_progress_start,
            progress_end=segment._segment_progress_end,
            hydrate=hydrate_words_after_punct,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.CONTEXT,
            run=context._run_context,
            artifacts=lambda c: (artifacts.segment_json_path(_out(c)),),
            progress_start=context._context_progress_start,
            progress_end=context._context_progress_end,
            hydrate=hydrate_segments_from_disk,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_COMPOSE,
            run=translate._run_translate_compose,
            artifacts=lambda c: (artifacts.segment_json_path(_out(c)), artifacts.plan_json_path(_out(c))),
            progress_start=translate._plan_progress_start,
            progress_end=translate._plan_progress_end,
            hydrate=hydrate_plan_segments,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.TRANSLATE_TRANSLATE,
            run=translate._run_translate_translate,
            artifacts=lambda c: (artifacts.plan_json_path(_out(c)),),
            progress_start=translate._translate_progress_start,
            hydrate=hydrate_plan_segments,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_RETRY,
            run=translate._run_translate_retry,
            artifacts=lambda c: (artifacts.plan_json_path(_out(c)),),
            hydrate=translate._hydrate_translate_mid,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_EVALUATE,
            run=translate._run_translate_evaluate,
            artifacts=lambda c: (artifacts.partial_cues_path(_out(c)),),
            hydrate=translate._hydrate_translate_mid,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key and c.evaluate_enabled),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_JOIN,
            run=translate._run_translate_join,
            artifacts=lambda c: (artifacts.partial_cues_path(_out(c)),),
            hydrate=translate._hydrate_translate_mid,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_SAVE,
            run=translate._run_translate_save,
            artifacts=lambda c: (artifacts.partial_cues_path(_out(c)),),
            progress_end=translate._translate_progress_end,
            hydrate=translate._hydrate_translate_mid,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.ANNOTATE,
            run=annotate._run_annotate,
            artifacts=lambda c: (artifacts.raw_cues_path(_out(c)),),
            progress_start=annotate._annotate_progress_start,
            progress_end=annotate._annotate_progress_end,
            hydrate=hydrate_annotate_inputs,
            enabled=lambda c: bool(c.annotate and c.target_lang),
        ),
        StepDefinition(
            id=StepId.SUBTITLE,
            run=subtitle._run_subtitle,
            artifacts=_subtitle_artifact_paths,
            progress_start=subtitle._format_progress_start,
            hydrate=hydrate_subtitle_export,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.EXPORT,
            run=export._run_export,
            artifacts=_subtitle_artifact_paths,
            progress_end=subtitle._format_progress_end,
            hydrate=hydrate_subtitle_export,
            enabled=lambda _c: True,
        ),
    ]


def build_enabled_definitions(config: SubtitleConfig) -> list[StepDefinition]:
    """Return enabled steps in pipeline order."""
    return [d for d in build_step_definitions(config) if d.enabled(config)]


# ── ASR phase boundary ─────────────────────────────────────────────────────────

ASR_STEP_IDS: frozenset[StepId] = frozenset(
    {
        StepId.ASR_EXTRACT,
        StepId.ASR_TRANSCRIBE,
        StepId.ASR_ALIGN,
        StepId.ASR_DIARIZE,
    }
)
