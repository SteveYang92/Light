"""ASR pipeline — extract_audio → transcribe → align → diarize."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

from light_models import Word

from ... import logger
from ...config import AsrEngine, SubtitleConfig
from . import align, diarize, extract_audio, transcribe, whisperx  # noqa: E402
from .artifacts import save_asr_words, save_whisper_cpp_raw

# Suppress pyannote's harmless torchcodec warning on Apple Silicon.
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")


@dataclass
class AsrResult:
    """Result of the ASR pipeline phase."""

    words: list[Word] = field(default_factory=list)


@dataclass
class AsrContext:
    """Shared state between ASR sub-steps."""

    audio_path: str = ""
    words: list[Word] = field(default_factory=list)


def run(config: SubtitleConfig) -> AsrResult:
    """Run full ASR pipeline (all sub-steps)."""
    ctx = AsrContext()
    _run_asr_step("extract", config, ctx)
    _run_asr_step("transcribe", config, ctx)
    if config.asr == AsrEngine.WHISPER_CPP:
        _run_asr_step("align", config, ctx)
    if config.diarize:
        _run_asr_step("diarize", config, ctx)
    return AsrResult(words=ctx.words)


def _run_asr_step(step: str, config: SubtitleConfig, ctx: AsrContext) -> None:
    """Execute a single ASR sub-step, mutating *ctx*."""
    match step:
        case "extract":
            ctx.audio_path = extract_audio.run(config)
            logger.info(f"  Extract: {ctx.audio_path}")
        case "transcribe":
            ctx.words = _transcribe(config, ctx.audio_path)
            save_asr_words(config, ctx.words)
            logger.info(f"  Transcribe: {len(ctx.words)} words")
        case "align":
            ctx.words = align.run(ctx.words, ctx.audio_path, language=_resolve_lang(config))
            save_asr_words(config, ctx.words)
            logger.info(f"  Align: {len(ctx.words)} words")
        case "diarize":
            ctx.words = diarize.run(
                ctx.words,
                ctx.audio_path,
                hf_token=config.hf_token,
                model_name=config.diarize_model,
            )
            save_asr_words(config, ctx.words)
            logger.info("  Diarization done.")
        case _:
            raise ValueError(f"Unknown ASR step: {step}")


def _transcribe(config: SubtitleConfig, audio_path: str) -> list[Word]:
    if config.asr == AsrEngine.WHISPERX:
        lang = config.language if config.language != "auto" else "en"
        return whisperx.run(audio_path, language=lang)
    words = transcribe.run(config, audio_path)
    raw_src = Path(config.output_dir) / "asr" / "whisper_output.json"
    if raw_src.exists():
        save_whisper_cpp_raw(config, raw_src)
    return words


def _resolve_lang(config: SubtitleConfig) -> str:
    return config.language if config.language != "auto" else "en"
