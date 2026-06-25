"""CLI definition — parse arguments, build config, dispatch to Orchestrator.

When ``--url`` is given the video is downloaded first and a semantic slug
is derived from the title.  When ``--input`` refers to a local file longer
than 45 minutes the video is automatically split at silence boundaries,
each segment processed independently, and the results merged.

Short local files and single short downloads run the standard pipeline
directly (backward-compatible with the legacy ``--input``-only path).
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

import typer

from .config import AsrEngine, SubtitleConfig
from .download import derive_slug_from_path, download_video, find_cached_download
from .runner import process_video
from .utils.whisper_utils import find_model, find_whisper

# ── Validation ──────────────────────────────────────────


def _validate_asr(value: str) -> str:
    valid = {e.value for e in AsrEngine}
    if value not in valid:
        raise typer.BadParameter(f"'{value}'.  Choose from: {', '.join(sorted(valid))}")
    return value


# ── CLI application ─────────────────────────────────────

app = typer.Typer()


def _default_resume_from_help() -> str:
    from .step_plan import list_step_ids

    steps = list_step_ids(SubtitleConfig(input_path="", output_dir="./output"))
    return f"Start from a specific step (e.g. {', '.join(steps[:4])}, …). Depends on --target-lang, --asr, etc."


@app.command()
def run(
    # ── Input (mutually exclusive: --url or --input) ───
    input_path: str = typer.Option(
        "",
        "-i",
        "--input",
        help="Input video/audio file (local path). Mutually exclusive with --url.",
    ),
    url: str = typer.Option(
        "",
        "--url",
        help="Video URL (YouTube, X, etc.) — downloads via yt-dlp. Mutually exclusive with --input.",
    ),
    # ── Output ──────────────────────────────────────────
    output_dir: str = typer.Option("./output", "-o", "--output", help="Output directory"),
    # ── ASR ─────────────────────────────────────────────
    whisper_model: str = typer.Option("ggml-large-v3-turbo.bin", "--whisper-model"),
    whisper_path: str = typer.Option(
        "whisper-cli",
        "--whisper-path",
        help="Path to whisper-cli (auto-detected from ~/whisper.cpp if not found)",
    ),
    language: str = typer.Option("auto", "-l", "--language"),
    asr: str = typer.Option(
        "whisperx",
        "--asr",
        help="ASR engine: whisperx (default) or whisper-cpp",
        callback=_validate_asr,
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize/--no-diarize",
        help="Enable speaker diarization (requires HF token)",
    ),
    diarize_model: str = typer.Option(
        "pyannote/speaker-diarization-community-1",
        "--diarize-model",
        help="Pyannote diarization model name",
    ),
    hf_token: str = typer.Option(
        "",
        "--hf-token",
        help="HuggingFace token for pyannote diarization (env: HF_TOKEN)",
    ),
    # ── Translation ─────────────────────────────────────
    target_lang: str = typer.Option(
        "", "--target-lang", help="Target language for translation (e.g. zh, en). Empty = source-only"
    ),
    bilingual: bool = typer.Option(False, "--bilingual", help="Output both source and translated subtitles"),
    # ── LLM ─────────────────────────────────────────────
    llm_base_url: str = typer.Option("https://api.deepseek.com", "--llm-base-url"),
    llm_model: str = typer.Option("deepseek-v4-flash", "--llm-model"),
    llm_api_key: str = typer.Option("", "--llm-api-key"),
    llm_temperature: float = typer.Option(0.4, "--llm-temperature"),
    # ── Formatting ──────────────────────────────────────
    cps_limit: int = typer.Option(9, "--cps-limit"),
    cps_limit_en: int = typer.Option(25, "--cps-limit-en"),
    max_lines: int = typer.Option(2, "--max-lines"),
    max_lines_zh: int = typer.Option(1, "--max-lines-zh", help="Max lines per cue for Chinese"),
    max_chars_per_line_zh: int = typer.Option(40, "--max-chars-zh"),
    max_chars_per_line_en: int = typer.Option(42, "--max-chars-en"),
    min_duration: float = typer.Option(0.8, "--min-duration"),
    max_duration: float = typer.Option(7.0, "--max-duration"),
    reading_padding: float = typer.Option(0.3, "--reading-padding"),
    # ── Advanced features ───────────────────────────────
    annotate: bool = typer.Option(
        False,
        "--annotate/--no-annotate",
        help="Generate LLM-powered secondary subtitle annotations",
    ),
    annotation_width: int = typer.Option(
        30,
        "--annotation-width",
        min=1,
        max=100,
        help="Annotation box width as % of screen (default 30)",
    ),
    evaluate: bool = typer.Option(
        False,
        "--evaluate/--no-evaluate",
        help="Enable LLM quality evaluation and refinement (adds ~2x cost)",
    ),
    quality_threshold: float = typer.Option(
        0.7,
        "--quality-threshold",
        min=0.0,
        max=1.0,
        help="Minimum quality score threshold for evaluation (default 0.7)",
    ),
    no_correct: bool = typer.Option(
        False,
        "--no-correct",
        help="Skip LLM-based transcript correction after ASR",
    ),
    no_context: bool = typer.Option(
        False,
        "--no-context",
        help="Skip glossary and content summary extraction before translation",
    ),
    glossary: str = typer.Option("", "--glossary", help="Path to YAML glossary"),
    config_file: str = typer.Option("", "-c", "--config", help="YAML config file"),
    # ── Long-video splitting ─────────────────────────
    split_threshold: float = typer.Option(
        2700.0,
        "--split-threshold",
        help=(
            "Seconds; videos longer than this are split at silence boundaries "
            "(default 2700 = 45 min). Lower to force splitting shorter videos."
        ),
    ),
    # ── Resume ──────────────────────────────────────────
    resume: bool = typer.Option(False, "--resume", help="Resume from failed/interrupted step in pipeline_run.json"),
    resume_from: str = typer.Option(
        "",
        "--resume-from",
        help=_default_resume_from_help(),
    ),
):
    # ═══════════════════════════════════════════════════════
    #  1. Validate mutually exclusive input
    # ═══════════════════════════════════════════════════════
    has_url = bool(url)
    has_input = bool(input_path)

    if not has_url and not has_input:
        raise typer.BadParameter("Either --input or --url must be provided.")
    if has_url and has_input:
        raise typer.BadParameter("--input and --url are mutually exclusive.")

    # ═══════════════════════════════════════════════════════
    #  2. Resolve input: download if URL
    # ═══════════════════════════════════════════════════════
    output_base = Path(output_dir)

    if has_url:
        cached = find_cached_download(url, output_base)
        if cached is not None:
            video_path, slug = cached
            print(f"  Using cached download: {video_path}")
        else:
            video_path, slug = download_video(url, output_base)
    else:
        video_path = Path(input_path).resolve()
        # Use parent directory name as slug only when the file is our generic
        # "video.*" (from yt-dlp download).  For uniquely named files the
        # stem carries the actual title.
        stem = video_path.stem
        parent_name = video_path.parent.name
        if stem == "video" and parent_name and parent_name not in (".", "..") and not parent_name.startswith(".seg"):
            slug = derive_slug_from_path(video_path.parent)
        else:
            slug = derive_slug_from_path(video_path)

    # ═══════════════════════════════════════════════════════
    #  3. Build config (shared across all paths)
    # ═══════════════════════════════════════════════════════
    resolved_whisper_path = find_whisper(whisper_path)
    resolved_whisper_model = find_model(whisper_model, resolved_whisper_path)

    if config_file:
        config = SubtitleConfig.from_yaml(config_file)
    else:
        glossary_dict: dict[str, str] = {}
        if glossary:
            import yaml

            with open(glossary) as f:
                glossary_dict = yaml.safe_load(f) or {}

        config = SubtitleConfig(
            input_path=str(video_path),
            output_dir=output_dir,
            url=url if has_url else None,
            slug=slug,
            bilingual=bilingual,
            whisper_model=resolved_whisper_model,
            whisper_path=resolved_whisper_path,
            language=language,
            target_lang=target_lang if target_lang else None,
            cps_limit=cps_limit,
            cps_limit_en=cps_limit_en,
            max_lines=max_lines,
            max_lines_zh=max_lines_zh,
            max_chars_per_line_zh=max_chars_per_line_zh,
            max_chars_per_line_en=max_chars_per_line_en,
            min_duration=min_duration,
            max_duration=max_duration,
            reading_padding=reading_padding,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            llm_temperature=llm_temperature,
            glossary=glossary_dict,
            asr=AsrEngine(asr),
            resume=resume,
            resume_from=resume_from if resume_from else None,
            diarize=diarize,
            diarize_model=diarize_model,
            hf_token=hf_token or os.environ.get("HF_TOKEN", ""),
            evaluate_enabled=evaluate,
            quality_threshold=quality_threshold,
            correct_enabled=not no_correct,
            context_prep_enabled=not no_context,
            annotate=annotate,
            annotation_width=annotation_width,
            split_threshold=split_threshold,
        )

    # ═══════════════════════════════════════════════════════
    #  4. Process: split if long, otherwise run directly
    # ═══════════════════════════════════════════════════════
    # ── Install SIGINT/SIGTERM handler (CLI only) ─────
    shutdown = threading.Event()

    def _on_sigint(_signum: int, _frame: object) -> None:
        print("\n  Shutting down...", flush=True)
        shutdown.set()

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # ═══════════════════════════════════════════════════════
    #  4. Run pipeline via shared runner
    # ═══════════════════════════════════════════════════════
    result = process_video(config)
    work_dir = result.output_dir

    # Rename generic outputs to slug-prefixed names for short videos.
    # Long videos are already named by the merge step.
    is_segment = work_dir.name.startswith((".seg", "chunk_"))
    if not is_segment:
        # Short video: outputs are zh.srt / cues.json etc.
        # Check if merge already produced slug-prefixed names
        slug_prefix = f"{slug}."
        has_slug_prefix = any(
            f.name.startswith(slug_prefix) and f.suffix in (".srt", ".vtt", ".json") for f in work_dir.iterdir()
        )
        if not has_slug_prefix:
            _rename_outputs(work_dir, slug)
    _cleanup_temp(work_dir)


# ═══════════════════════════════════════════════════════════
#  Output helpers
# ═══════════════════════════════════════════════════════════


def _rename_outputs(work_dir: Path, slug: str) -> None:
    """Rename pipeline outputs from generic names to ``{slug}.<ext>``."""
    import shutil

    mapping = {
        "zh.srt": f"{slug}.zh.srt",
        "zh.vtt": f"{slug}.zh.vtt",
        "cues.json": f"{slug}.cues.json",
        "annotations.ass": f"{slug}.annotations.ass",
        "annotations.vtt": f"{slug}.annotations.vtt",
    }

    # Copy transcript.json (not move) — resume depends on it.
    transcript_src = work_dir / "transcript.json"
    transcript_dst = work_dir / f"{slug}.transcript.json"
    if transcript_src.exists() and not transcript_dst.exists():
        import shutil as _shutil

        _shutil.copy2(str(transcript_src), str(transcript_dst))
    for src_name, dst_name in mapping.items():
        src = work_dir / src_name
        dst = work_dir / dst_name
        if src.exists():
            shutil.move(str(src), str(dst))

    # Rename the downloaded video file as well.
    for video_file in work_dir.glob("video.*"):
        dst = work_dir / f"{slug}{video_file.suffix}"
        if not dst.exists():
            shutil.move(str(video_file), str(dst))


def _cleanup_temp(work_dir: Path) -> None:
    """Remove intermediate audio files left by the ASR pipeline."""
    for name in ("audio_asr.wav", "audio_original.wav"):
        f = work_dir / name
        if f.exists():
            f.unlink()


# ── Pack command ─────────────────────────────────────────


@app.command()
def pack(
    output_dir: str = typer.Argument(..., help="Pipeline output directory containing video and subtitles"),
    font: str = typer.Option("PingFang SC", "--font", help="Font for main subtitle overlay"),
    encoder: str = typer.Option(
        "h264_videotoolbox",
        "--encoder",
        help="Video encoder: h264_videotoolbox (Apple hardware) or libx264 (software)",
    ),
    video: str = typer.Option(
        "",
        "--video",
        help="Explicit path to input video (auto-detected from output_dir if not set)",
    ),
):
    """Burn subtitles into video — produce a self-contained MP4.

    Discovers .zh.srt and optional .annotations.ass from OUTPUT_DIR,
    hard-burns both subtitle tracks into the video stream, and writes
    ``{slug}_pack.mp4`` alongside the original video.
    """
    from .pack import PackConfig, run_pack

    config = PackConfig(
        output_dir=output_dir,
        font=font,
        encoder=encoder,
        video=video if video else None,
    )
    run_pack(config)


def main():
    app()
