"""ASR steps — extract audio, transcribe, align, diarize (+ hydrate wrappers)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .. import logger
from ..config import AsrEngine, SubtitleConfig
from ..pipeline.asr import align, diarize, extract_audio, transcribe, whisperx
from ..pipeline.asr.artifacts import save_asr_words, save_whisper_cpp_raw
from ..state_hydrate import hydrate_asr_audio, hydrate_asr_words
from .export import _export_transcript
from .progress import STAGE_ASR

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _asr_progress_start(orch: Orchestrator) -> None:
    orch._progress(STAGE_ASR, 0.0, "提取音频中...")


def _asr_progress_end(orch: Orchestrator) -> None:
    orch._progress(STAGE_ASR, 1.0, f"ASR 完成 ({len(orch.state.words)} 个词)")


def _resolve_asr_lang(config: SubtitleConfig) -> str:
    return config.language if config.language != "auto" else "en"


def _transcribe_words(config: SubtitleConfig, audio_path: str) -> list:
    if config.asr == AsrEngine.WHISPERX:
        return whisperx.run(audio_path, language=_resolve_asr_lang(config))
    words = transcribe.run(config, audio_path)
    raw_src = Path(config.output_dir) / "asr" / "whisper_output.json"
    if raw_src.exists():
        save_whisper_cpp_raw(config, raw_src)
    return words


def _run_asr_extract(orch: Orchestrator) -> None:
    orch.state.audio_path = extract_audio.run(orch.config)
    logger.info(f"  Extract: {orch.state.audio_path}")


def _run_asr_transcribe(orch: Orchestrator) -> None:
    orch.state.words = _transcribe_words(orch.config, orch.state.audio_path)
    save_asr_words(orch.config, orch.state.words)
    logger.info(f"  Transcribe: {len(orch.state.words)} words")
    _export_transcript(orch, Path(orch.config.output_dir))


def _run_asr_align(orch: Orchestrator) -> None:
    orch.state.words = align.run(
        orch.state.words,
        orch.state.audio_path,
        language=_resolve_asr_lang(orch.config),
    )
    save_asr_words(orch.config, orch.state.words)
    logger.info(f"  Align: {len(orch.state.words)} words")
    _export_transcript(orch, Path(orch.config.output_dir))


def _run_asr_diarize(orch: Orchestrator) -> None:
    orch.state.words = diarize.run(
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
