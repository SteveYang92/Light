"""ASR steps — extract audio, transcribe, align, diarize (+ hydrate wrappers).

Thin orchestration over :mod:`light_asr`: builds an :class:`AsrConfig` from
:class:`SubtitleConfig`, calls the granular providers, and persists
checkpoints via :mod:`light_cli.artifacts` (which owns run-dir paths).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from light_asr import align as asr_align
from light_asr import diarize as asr_diarize
from light_asr import whisper_cpp, whisperx
from light_asr.audio import extract_audio
from light_asr.config import AsrConfig
from light_core import logger
from light_models import Word

from ..artifacts import asr_dir, save_asr_words, save_whisper_cpp_raw
from ..config import AsrEngine, SubtitleConfig
from ..reporting import StageStatus
from ..state_hydrate import hydrate_asr_audio, hydrate_asr_words
from .export import _export_transcript
from .progress import STAGE_ASR

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _asr_progress_start(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_ASR, StageStatus.started, 0.0, "提取音频中...")


def _asr_progress_end(orch: Orchestrator) -> None:
    orch.emit_progress(STAGE_ASR, StageStatus.finished, 1.0, f"ASR 完成 ({len(orch.state.words)} 个词)")


def _resolve_asr_lang(config: SubtitleConfig) -> str:
    return config.language if config.language != "auto" else "en"


def _asr_config(config: SubtitleConfig) -> AsrConfig:
    return AsrConfig(
        engine=config.asr,
        whisper_model=config.whisper_model,
        whisper_path=config.whisper_path,
        language=config.language,
        diarize=config.diarize,
        diarize_model=config.diarize_model,
        hf_token=config.hf_token,
    )


def _transcribe_words(config: SubtitleConfig, audio_path: str) -> list[Word]:
    if config.asr == AsrEngine.WHISPERX:
        return whisperx.run(audio_path, language=_resolve_asr_lang(config))
    work_dir = asr_dir(config.output_dir)
    words = whisper_cpp.transcribe(audio_path, _asr_config(config), work_dir)
    raw_src = work_dir / "whisper_output.json"
    if raw_src.exists():
        save_whisper_cpp_raw(config, raw_src)
    return words


def _run_asr_extract(orch: Orchestrator) -> None:
    orch.state.audio_path = extract_audio(orch.config.input_path, orch.config.output_dir)
    logger.info(f"  Extract: {orch.state.audio_path}")


def _run_asr_transcribe(orch: Orchestrator) -> None:
    orch.state.words = _transcribe_words(orch.config, orch.state.audio_path)
    save_asr_words(orch.config, orch.state.words)
    logger.info(f"  Transcribe: {len(orch.state.words)} words")
    _export_transcript(orch, Path(orch.config.output_dir))


def _run_asr_align(orch: Orchestrator) -> None:
    orch.state.words = asr_align.align_words(
        orch.state.words,
        orch.state.audio_path,
        language=_resolve_asr_lang(orch.config),
    )
    save_asr_words(orch.config, orch.state.words)
    logger.info(f"  Align: {len(orch.state.words)} words")
    _export_transcript(orch, Path(orch.config.output_dir))


def _run_asr_diarize(orch: Orchestrator) -> None:
    orch.state.words = asr_diarize.run(
        orch.state.words,
        orch.state.audio_path,
        hf_token=orch.config.hf_token,
        model_name=orch.config.diarize_model,
    )
    save_asr_words(orch.config, orch.state.words)
    logger.info("  Diarization done.")
    _export_transcript(orch, Path(orch.config.output_dir))


def _hydrate_asr_align(orch: Orchestrator) -> None:
    hydrate_asr_audio(orch)
    hydrate_asr_words(orch)
