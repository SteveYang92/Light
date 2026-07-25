"""``light asr`` — standalone ASR: audio/video → transcript.json.

Thin wrapper over :mod:`light_asr` (extract_audio + api.transcribe).
Writes ``transcript.json`` (``light-transcript.v1``, same writer as the
pipeline) plus the ``asr/asr_<provider>.json`` word checkpoint, so the
output can feed ``light polish`` / ``light subtitle`` / ``light-qc``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from light_asr import (
    AsrConfig,
    AsrEngine,
    extract_audio,
    find_model,
    find_whisper,
    save_words_checkpoint,
    transcribe,
)
from light_core import logger
from light_subtitle import export as export_module

from ..cli import _validate_asr


def asr(
    input_path: Annotated[str, typer.Option("-i", "--input", help="Input audio/video file")],
    output_dir: Annotated[str, typer.Option("-o", "--output", help="Output directory")] = "./output",
    language: Annotated[str, typer.Option("-l", "--language")] = "auto",
    engine: Annotated[
        str,
        typer.Option("--asr", help="ASR engine: whisperx (default) or whisper-cpp", callback=_validate_asr),
    ] = "whisperx",
    whisper_model: Annotated[str, typer.Option("--whisper-model")] = "ggml-large-v3-turbo.bin",
    whisper_path: Annotated[
        str,
        typer.Option("--whisper-path", help="Path to whisper-cli (auto-detected from ~/whisper.cpp if not found)"),
    ] = "whisper-cli",
    diarize: Annotated[
        bool,
        typer.Option("--diarize/--no-diarize", help="Enable speaker diarization (requires HF token)"),
    ] = False,
    diarize_model: Annotated[str, typer.Option("--diarize-model")] = "pyannote/speaker-diarization-community-1",
    hf_token: Annotated[
        str,
        typer.Option("--hf-token", help="HuggingFace token for pyannote diarization (env: HF_TOKEN)"),
    ] = "",
):
    """Transcribe audio/video into transcript.json (word-level timestamps)."""
    out = Path(output_dir)
    audio_path = extract_audio(input_path, out)
    logger.info(f"  Extract: {audio_path}")

    resolved_engine = AsrEngine(engine)
    resolved_whisper_path = find_whisper(whisper_path)
    config = AsrConfig(
        engine=resolved_engine,
        whisper_model=find_model(whisper_model, resolved_whisper_path),
        whisper_path=resolved_whisper_path,
        language=language,
        diarize=diarize,
        diarize_model=diarize_model,
        hf_token=hf_token or os.environ.get("HF_TOKEN", ""),
    )

    work = out / "asr"
    words = transcribe(audio_path, config, work)
    save_words_checkpoint(words, work / f"asr_{resolved_engine.value}.json", resolved_engine.value)

    transcript = out / "transcript.json"
    source = f"{resolved_engine.value} {config.whisper_model}"
    export_module.export_transcript(words, [], str(transcript), source=source)
    logger.info(f"  Transcribe: {len(words)} words → {transcript}")
    typer.echo(f"transcript.json: {transcript} ({len(words)} words)")
