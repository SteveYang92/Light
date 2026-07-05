# ruff: noqa: B008
from __future__ import annotations

import os
from pathlib import Path

import typer

from .config import DEFAULT_MODEL, EngineMode, MixMode, TtsConfig
from .dub import run_dub, run_poc

app = typer.Typer(no_args_is_help=True, help="Subtitle dubbing (Qwen3-TTS or official IndexTTS).")


@app.command()
def dub(
    output_dir: str = typer.Argument(..., help="Pipeline output directory (needs translations/raw.json + video)"),
    lang: str = typer.Option("zh", "--lang", help="Cue language filter"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="HuggingFace mlx-community model id"),
    voices: str = typer.Option("", "--voices", help="Path to voices.yaml"),
    mix: MixMode = typer.Option(MixMode.DUCK, "--mix", help="Audio mix mode"),
    engine: EngineMode = typer.Option(
        EngineMode.MLX,
        "--engine",
        help="mlx | http | mock | indextts2 | indextts15 | indextts2_metal",
    ),
    mlx_url: str = typer.Option("", "--mlx-url", help="mlx_audio.server base URL (http engine)"),
    official_root: Path = typer.Option(
        Path("vendor/index-tts"),
        "--official-root",
        help="Cloned official index-tts repo (indextts2 / indextts15)",
    ),
    ref_audio: Path | None = typer.Option(None, "--ref-audio", help="Speaker reference WAV (IndexTTS)"),
    checkpoints: Path | None = typer.Option(None, "--checkpoints", help="IndexTTS checkpoints dir"),
    metal_root: Path = typer.Option(Path("vendor/index-tts2-metal"), "--metal-root", help="index-tts2-metal root"),
    metal_url: str = typer.Option("", "--metal-url", help="mtts HTTP base URL (indextts2_metal)"),
    metal_cfm_steps: int = typer.Option(16, "--metal-cfm-steps", help="CFM steps for indextts2_metal"),
    metal_manage_server: bool = typer.Option(False, "--metal-manage-server", help="Auto-start mtts server"),
    emotion: str = typer.Option("calm", "--emotion", help="IndexTTS2 emotion (indextts2 only)"),
    emotion_weight: float = typer.Option(0.6, "--emotion-weight", help="IndexTTS2 emotion weight"),
    num_beams: int = typer.Option(3, "--num-beams", help="IndexTTS2 GPT beam width"),
    cues: str = typer.Option("", "--cues", help="Path to translations/raw.json or pipeline output directory"),
    video: str = typer.Option("", "--video", help="Explicit path to source video"),
    max_cues: int | None = typer.Option(None, "--max-cues", help="Limit cues (debug)"),
    resume: bool = typer.Option(False, "--resume", help="Skip existing segment WAVs"),
    skip_mix: bool = typer.Option(False, "--skip-mix", help="Only produce tts/dub.wav"),
    trim: bool = typer.Option(False, "--trim", help="Hard-trim speech to cue window (may cut words)"),
    per_cue: bool = typer.Option(False, "--per-cue", help="Dub each display cue separately (legacy/debug)"),
    atempo_max: float = typer.Option(1.28, "--atempo-max", help="Max ffmpeg atempo speed-up"),
    preview: bool = typer.Option(False, "--preview", help="Write a short preview under tts/preview/"),
    preview_duration: float = typer.Option(180.0, "--preview-duration", help="Preview prefix duration in seconds"),
    qwen_chunk_chars: int = typer.Option(180, "--qwen-chunk-chars", help="Max chars per Qwen turn"),
    chunk_chars: int = typer.Option(160, "--chunk-chars", help="Max chars per IndexTTS turn"),
    chunk_min_chars: int = typer.Option(45, "--chunk-min-chars", help="Min chars before IndexTTS soft split"),
    qwen_max_tokens_min: int = typer.Option(512, "--qwen-max-tokens-min", help="Minimum Qwen max_tokens cap"),
    qwen_max_tokens_max: int = typer.Option(2048, "--qwen-max-tokens-max", help="Maximum Qwen max_tokens cap"),
    qwen_seed: int | None = typer.Option(None, "--qwen-seed", help="Optional Qwen seed if supported by mlx-audio"),
    temperature: float = typer.Option(0.6, "--temperature", help="Qwen sampling temperature"),
    top_k: int = typer.Option(50, "--top-k", help="Qwen top-k sampling"),
    top_p: float = typer.Option(1.0, "--top-p", help="Qwen top-p sampling"),
    repetition_penalty: float = typer.Option(1.05, "--repetition-penalty", help="Qwen repetition penalty"),
    indextts_verbose: bool = typer.Option(False, "--indextts-verbose", help="Verbose IndexTTS infer logs"),
) -> None:
    """Generate dubbed audio (and optional {slug}_dub.mp4) from subtitle cues."""
    cfg = TtsConfig(
        output_dir=output_dir,
        lang=lang,
        model=model,
        voices_path=voices or None,
        mix_mode=mix,
        engine_mode=engine,
        mlx_server_url=mlx_url or os.environ.get("MLX_AUDIO_URL", "http://127.0.0.1:8000"),
        indextts_official_root=str(official_root),
        indextts_checkpoints=str(checkpoints) if checkpoints else None,
        indextts_metal_root=str(metal_root),
        indextts_metal_url=metal_url or os.environ.get("MIT2_SERVER_URL", "http://127.0.0.1:3456"),
        indextts_metal_cfm_steps=metal_cfm_steps,
        indextts_metal_manage_server=metal_manage_server,
        indextts_ref_audio=str(ref_audio) if ref_audio else None,
        indextts_emotion=emotion,
        indextts_emotion_weight=emotion_weight,
        indextts_num_beams=num_beams,
        indextts_chunk_chars=chunk_chars,
        indextts_chunk_min_chars=chunk_min_chars,
        indextts_verbose=indextts_verbose,
        max_cues=max_cues,
        resume=resume,
        video=video or None,
        allow_trim=trim,
        atempo_max=atempo_max,
        per_cue=per_cue,
        preview=preview,
        preview_duration_s=preview_duration,
        qwen_chunk_chars=qwen_chunk_chars,
        qwen_max_tokens_min=qwen_max_tokens_min,
        qwen_max_tokens_max=qwen_max_tokens_max,
        qwen_seed=qwen_seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    cues_path = Path(cues) if cues else None
    run_dub(cfg, cues_path=cues_path, skip_mix=skip_mix)


@app.command("poc")
def poc_cmd(
    cues: str = typer.Option(..., "--cues", help="Path to translations/raw.json or pipeline output directory"),
    out: str = typer.Option(..., "--out", help="Output directory for WAV files"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Qwen3-TTS model id"),
    max_cues: int = typer.Option(3, "--max-cues", help="Max cues to synthesize"),
    engine: EngineMode = typer.Option(
        EngineMode.MLX,
        "--engine",
        help="mlx | http | mock | indextts2 | indextts15 | indextts2_metal",
    ),
) -> None:
    """Phase 0 POC: synthesize a few cues to WAV (no timeline mix)."""
    run_poc(
        Path(cues),
        Path(out),
        model=model,
        max_cues=max_cues,
        engine_mode=engine,
    )
