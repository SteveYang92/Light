"""CLI definition — parse arguments, build config, dispatch to Orchestrator.

When ``--url`` is given the video is downloaded first and a semantic slug
is derived from the title.  When ``--input`` refers to a local file longer
than 45 minutes the video is automatically split at silence boundaries,
each segment processed independently, and the results merged.

Short local files and single short downloads run the standard pipeline
directly (backward-compatible with the legacy ``--input``-only path).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import typer

from . import logger
from .config import AsrEngine, SubtitleConfig
from .download import derive_slug_from_path, download_video, find_cached_download
from .reporting import Reporter, RunEvent, RunKind
from .runner import process_video
from .usage.report import load_usage_from_dir
from .utils.whisper_utils import find_model, find_whisper
from .video_split import segment_tag

# ── Validation ──────────────────────────────────────────


def _validate_asr(value: str) -> str:
    valid = {e.value for e in AsrEngine}
    if value not in valid:
        raise typer.BadParameter(f"'{value}'.  Choose from: {', '.join(sorted(valid))}")
    return value


# ── CLI application ─────────────────────────────────────

app = typer.Typer()


def _non_default_cli_params(local_ns: dict) -> list[str]:
    """Pipeline params explicitly set on the CLI (i.e. differing from their
    typer defaults) — used to warn when ``--config`` makes them no-ops."""
    import inspect

    skip = {"ctx", "input_path", "url", "config_file", "style_config"}
    names = []
    for name, param in inspect.signature(run).parameters.items():
        if name in skip:
            continue
        default = getattr(param.default, "default", param.default)
        if local_ns.get(name) != default:
            names.append(name)
    return names


# NOTE: the --resume-from help text mirrors the first steps of the default
# plan (asr.extract, asr.transcribe, correct, punct); it is intentionally
# static so importing the CLI has no side effects (previously this was
# computed from a live step plan at import time).


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
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
    max_duration: float = typer.Option(5.0, "--max-duration"),
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
    font: str = typer.Option(
        "PingFang SC",
        "--font",
        help="Subtitle font for ASS export (system fallback chain when unavailable)",
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
    style_config: str = typer.Option(
        "",
        "--style-config",
        help="YAML style overrides for bilingual subtitle boxes (see light_subtitle/style/config.py)",
    ),
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
        help=(
            "Start from a specific step (e.g. asr.extract, asr.transcribe, correct, punct, …)."
            " Depends on --target-lang, --asr, etc."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="输出详细日志（默认只显示进度）",
    ),
):
    if ctx.invoked_subcommand is not None:
        return

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
    # Non-verbose runs show reporter progress only (logger echo off).
    logger.set_console_echo(verbose)
    output_base = Path(output_dir)

    if has_url:
        cached = find_cached_download(url, output_base)
        if cached is not None:
            video_path, slug = cached
            logger.info(f"  Using cached download: {video_path}")
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
        ignored = _non_default_cli_params(locals())
        if ignored:
            logger.warning(f"  ⚠ --config 生效，以下 CLI 参数被忽略: {', '.join(ignored)}")
    else:
        glossary_dict: dict[str, str] = {}
        if glossary:
            import yaml

            with open(glossary) as f:
                glossary_dict = yaml.safe_load(f) or {}

        # CLI params → SubtitleConfig: fields with a same-named parameter are
        # copied verbatim; *special* holds the exceptions (derived values,
        # renames, env fallbacks, enum conversion).
        params = locals()
        special = {
            "input_path": str(video_path),
            "url": url if has_url else None,
            "slug": slug,
            "whisper_model": resolved_whisper_model,
            "whisper_path": resolved_whisper_path,
            "target_lang": target_lang if target_lang else None,
            "glossary": glossary_dict,
            "asr": AsrEngine(asr),
            "resume_from": resume_from if resume_from else None,
            "llm_api_key": llm_api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            "hf_token": hf_token or os.environ.get("HF_TOKEN", ""),
            "evaluate_enabled": evaluate,
            "correct_enabled": not no_correct,
            "context_prep_enabled": not no_context,
        }
        config_kwargs = {
            f.name: params[f.name]
            for f in dataclasses.fields(SubtitleConfig)
            if f.name in params and f.name not in special
        }
        config = SubtitleConfig(**config_kwargs, **special)

    if style_config:
        from .style.config import SubtitleStyleConfig

        config.style = SubtitleStyleConfig.load_yaml(style_config)

    # ═══════════════════════════════════════════════════════
    #  4. Run pipeline via shared runner
    # ═══════════════════════════════════════════════════════
    reporter = _make_reporter(verbose)
    try:
        result = process_video(config, progress_callback=reporter)
        work_dir = result.output_dir

        # Rename generic outputs to slug-prefixed names for short videos.
        # Long videos are already named by the merge step.
        is_segment = bool(segment_tag(work_dir))
        if not is_segment and _has_generic_outputs(work_dir):
            # Always rename when bare names exist — e.g. after ``--resume-from
            # subtitle`` export writes ``zh.srt`` / ``bilingual.ass`` even if an
            # earlier run already created ``{slug}.zh.srt``.
            _rename_outputs(work_dir, slug)
        _cleanup_temp(work_dir)
        reporter.emit(RunEvent(RunKind.finished, _finished_payload(work_dir, slug)))
    except KeyboardInterrupt:
        reporter.emit(RunEvent(RunKind.failed, {"error": "已中断（Ctrl+C），可用 --resume 续跑", **_log_payload()}))
        raise
    except SystemExit:
        # Orchestrator's SIGINT/SIGTERM handler raises SystemExit(130).
        reporter.emit(RunEvent(RunKind.failed, {"error": "已中断（信号），可用 --resume 续跑", **_log_payload()}))
        raise
    except Exception as e:
        reporter.emit(RunEvent(RunKind.failed, {"error": f"{type(e).__name__}: {e}", **_log_payload()}))
        raise
    finally:
        reporter.close()
        logger.set_console_echo(True)


# ═══════════════════════════════════════════════════════════
#  Reporter helpers
# ═══════════════════════════════════════════════════════════


def _is_tty() -> bool:
    """True for an interactive terminal that supports rich rendering."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM", "") != "dumb"


def _make_reporter(verbose: bool) -> Reporter:
    """Pick the progress renderer.

    Verbose → plain text (full logger stream conflicts with a live view);
    TTY → Rich live view (rich imported lazily); otherwise → plain text.
    """
    if verbose or not _is_tty():
        from .reporting import PlainReporter

        return PlainReporter()
    from .reporting.rich_ui import RichReporter

    return RichReporter()


def _log_payload() -> dict:
    """``{"log": path}`` for terminal events when a pipeline log is bound."""
    path = logger.log_path()
    return {"log": str(path)} if path else {}


def _finished_payload(work_dir: Path, slug: str) -> dict:
    """Payload for the terminal RunEvent(finished) — artifacts, usage, log."""
    artifacts = sorted(
        p.name for p in work_dir.glob(f"{slug}.*") if p.suffix in {".srt", ".vtt", ".ass", ".json"} and p.is_file()
    )
    payload: dict = {"slug": slug, "output": str(work_dir), "artifacts": artifacts, **_log_payload()}
    usage = _usage_line(work_dir)
    if usage:
        payload["usage"] = usage
    return payload


def _usage_line(work_dir: Path) -> str:
    """Compact ``tokens: N, ≈$X`` summary from usage_report.json (multi-segment
    runs already aggregate per-segment reports into this file at merge time)."""
    report = load_usage_from_dir(work_dir)
    if report is None:
        return ""
    parts: list[str] = []
    total = report.totals.get("total_tokens", 0)
    if total:
        parts.append(f"tokens: {total}")
    if report.cost.total_usd is not None:
        parts.append(f"≈${report.cost.total_usd:.4f}")
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════
#  Output helpers
# ═══════════════════════════════════════════════════════════


_GENERIC_OUTPUT_NAMES = (
    "zh.srt",
    "zh.vtt",
    "en.srt",
    "en.vtt",
    "bilingual.ass",
    "bilingual.vtt",
    "cues.json",
    "annotations.ass",
    "annotations.vtt",
)


def _has_generic_outputs(work_dir: Path) -> bool:
    """True when the export step wrote bare filenames that need slug prefixing."""
    return any((work_dir / name).exists() for name in _GENERIC_OUTPUT_NAMES)


def _rename_outputs(work_dir: Path, slug: str) -> None:
    """Rename pipeline outputs from generic names to ``{slug}.<ext>``."""
    import shutil

    mapping = {
        "zh.srt": f"{slug}.zh.srt",
        "zh.vtt": f"{slug}.zh.vtt",
        "en.srt": f"{slug}.en.srt",
        "en.vtt": f"{slug}.en.vtt",
        "bilingual.ass": f"{slug}.bilingual.ass",
        "bilingual.vtt": f"{slug}.bilingual.vtt",
        "cues.json": f"{slug}.cues.json",
        "annotations.ass": f"{slug}.annotations.ass",
        "annotations.vtt": f"{slug}.annotations.vtt",
    }

    # Copy transcript.json (not move) — resume depends on it.
    transcript_src = work_dir / "transcript.json"
    transcript_dst = work_dir / f"{slug}.transcript.json"
    if transcript_src.exists() and not transcript_dst.exists():
        shutil.copy2(str(transcript_src), str(transcript_dst))
    for src_name, dst_name in mapping.items():
        src = work_dir / src_name
        dst = work_dir / dst_name
        if src.exists():
            if dst.exists():
                dst.unlink()
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
    font: str = typer.Option(
        "PingFang SC",
        "--font",
        help="Subtitle font; ASS tracks are patched before burn, SRT uses force_style",
    ),
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

    Auto-detects the main subtitle from OUTPUT_DIR: prefers ``bilingual.ass``
    (self-styled, 中上英下) when present, otherwise falls back to ``zh.srt``.
    ``--font`` applies to all paths (ASS Style patch or SRT force_style) with
    a built-in system fallback chain.  Optional ``.annotations.ass`` is
    overlaid as a secondary track.  Writes ``{slug}_pack.mp4`` alongside the
    original video.

    Run a bilingual pipeline first to get ``bilingual.ass``::

        uv run light-subtitle -i input.mp4 --target-lang zh --bilingual -o output
        uv run light-subtitle pack output
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
