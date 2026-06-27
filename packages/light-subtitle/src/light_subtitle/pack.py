"""Pack subtitle files (SRT + ASS annotations) into a video via hard-burn.

Independent from the subtitle pipeline — post-processing step to produce
a self-contained MP4 with subtitles baked into the video stream.

Requires ``ffmpeg-full`` (Homebrew) for libass support::

    brew install ffmpeg-full
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import logger
from .fonts import FontConfig, resolve_font, write_patched_ass

# ── Constants ───────────────────────────────────────────

FFMPEG_FULL_PATHS = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
]

FFPROBE_FULL_PATHS = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe",
    "/usr/local/opt/ffmpeg-full/bin/ffprobe",
]

FONT_SIZE = 20
MARGIN_V = 10
OUTPUT_SUFFIX = "_pack"


# ── Config ──────────────────────────────────────────────


@dataclass
class PackConfig:
    """Configuration for the subtitle-video packing step."""

    output_dir: str
    """Path to the pipeline output directory containing the video and subtitles."""

    font: str = "PingFang SC"
    """Preferred subtitle font; resolved via system fallback chain before burn."""

    encoder: str = "h264_videotoolbox"
    """Video encoder.  ``h264_videotoolbox`` (Apple hardware) or ``libx264`` (software)."""

    video: str | None = None
    """Explicit path to the input video file.  Auto-detected from *output_dir* if None."""


# ── Public API ──────────────────────────────────────────


def run_pack(config: PackConfig) -> None:
    """Burn subtitles into a video and write a self-contained MP4.

    Discovers the video, main subtitle, and optional annotation subtitle from
    *config.output_dir*, then encodes with ffmpeg-full.  The main subtitle is
    auto-detected: ``bilingual.ass`` (self-styled, burned via ``ass=``) is
    preferred when present; otherwise falls back to ``zh.srt`` (burned via
    ``subtitles=`` with ``force_style``).
    """
    output_dir = Path(config.output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # ── Locate ffmpeg-full ───────────────────────────
    ffmpeg_bin, ffprobe_bin = _find_ffmpeg_full()

    # ── Discover media files ─────────────────────────
    video_path, sub_path, sub_kind, annot_path = _discover_files(output_dir, config.video)
    resolved_font = resolve_font(FontConfig(primary=config.font))
    logger.info(f"  主字幕: {sub_path.name} ({'双语 ASS' if sub_kind == 'bilingual' else 'SRT'})")
    if resolved_font != config.font:
        logger.info(f"  字体: {config.font} → {resolved_font}")
    else:
        logger.info(f"  字体: {resolved_font}")

    # ── Probe original bitrate ───────────────────────
    original_bitrate = _probe_video_bitrate(ffprobe_bin, video_path)
    logger.info(f"  原始视频码率: {original_bitrate} kbps")

    # ── Derive output path ───────────────────────────
    slug = video_path.stem
    output_path = output_dir / f"{slug}{OUTPUT_SUFFIX}.mp4"

    # ── Build filter chain ───────────────────────────
    temp_dir = Path(tempfile.mkdtemp(prefix="light-pack-"))
    try:
        filters: list[str] = []
        if annot_path:
            patched_annot = temp_dir / "annotations.patched.ass"
            write_patched_ass(annot_path, resolved_font, patched_annot)
            filters.append(f"ass={patched_annot}")

        if sub_kind == "bilingual":
            patched_sub = temp_dir / "bilingual.patched.ass"
            write_patched_ass(sub_path, resolved_font, patched_sub)
            filters.append(f"ass={patched_sub}")
            logger.info(f"  双语 ASS 已应用 --font → {resolved_font}")
        else:
            filters.append(
                f"subtitles={sub_path}:force_style='Fontsize={FONT_SIZE},Fontname={resolved_font},MarginV={MARGIN_V}'"
            )
        filter_complex = ",".join(filters)

        # ── Run ffmpeg ───────────────────────────────────
        cmd = [
            str(ffmpeg_bin),
            "-i",
            str(video_path),
            "-filter_complex",
            f"[0:v]{filter_complex}[outv]",
            "-map",
            "[outv]",
            "-map",
            "0:a",
            "-c:v",
            config.encoder,
            "-b:v",
            f"{original_bitrate}k",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-y",
            str(output_path),
        ]

        logger.info(f"  编码中... (ffmpeg {' '.join(cmd[1:4])} ...)")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg 编码失败 (exit {e.returncode})") from e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Report ───────────────────────────────────────
    size_mb = output_path.stat().st_size / (1024 * 1024)
    mode_label = "双语 ASS" if sub_kind == "bilingual" else "中文字幕 SRT"
    logger.info(f"  打包完成: {output_path.name} ({size_mb:.0f} MB, {mode_label})")
    print(f"\n  ✅ {output_path}")


# ── Helpers ────────────────────────────────────────────


def _find_ffmpeg_full() -> tuple[Path, Path]:
    """Locate ffmpeg-full binaries; raise with install instructions if missing."""
    ffmpeg = _find_bin(FFMPEG_FULL_PATHS, "ffmpeg")
    ffprobe = _find_bin(FFPROBE_FULL_PATHS, "ffprobe")

    if ffmpeg is None or ffprobe is None:
        missing = []
        if ffmpeg is None:
            missing.append("ffmpeg")
        if ffprobe is None:
            missing.append("ffprobe")
        raise RuntimeError(
            f"未找到 ffmpeg-full ({', '.join(missing)} 缺失)。\n"
            "  ffmpeg-full 提供 libass 支持，用于硬烧 ASS 注释字幕。\n"
            "  安装方法:\n"
            "    brew install ffmpeg-full"
        )

    return ffmpeg, ffprobe


def _find_bin(paths: list[str], name: str) -> Path | None:
    """Search *paths*, then fall back to ``shutil.which``."""
    for p in paths:
        if Path(p).exists():
            return Path(p)
    found = shutil.which(name)
    return Path(found) if found else None


def _find_subtitle_in_dir(output_dir: Path, video_stem: str, suffix: str) -> Path | None:
    """Locate ``{stem}{suffix}``, bare ``{suffix}``, or a unique ``*{suffix}`` in *output_dir*."""
    bare_name = suffix if suffix.startswith(".") else f".{suffix}"
    for name in (f"{video_stem}{bare_name}", bare_name.lstrip(".")):
        path = output_dir / name
        if path.is_file():
            return path
    globbed = sorted(output_dir.glob(f"*{bare_name}"))
    if len(globbed) == 1:
        return globbed[0]
    return None


def _discover_files(output_dir: Path, video_override: str | None) -> tuple[Path, Path, str, Path | None]:
    """Discover video, main subtitle, subtitle kind, and optional annotation.

    Returns ``(video, sub_path, sub_kind, annot_path)`` where *sub_kind* is
    ``"bilingual"`` (ASS, self-styled — burn with ``ass=``) or ``"srt"``
    (needs ``force_style``).  Bilingual ASS is preferred when present; falls
    back to monolingual ``zh.srt`` for non-bilingual runs.
    """
    # ── Video ─────────────────────────────────────────
    if video_override:
        video_path = Path(video_override).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"视频文件未找到: {video_path}")
    else:
        mp4_files = sorted(p for p in output_dir.glob("*.mp4") if OUTPUT_SUFFIX not in p.stem)
        if not mp4_files:
            raise FileNotFoundError(f"在 {output_dir} 中未找到 .mp4 视频文件")
        if len(mp4_files) > 1:
            names = "\n".join(f"    {f.name}" for f in mp4_files)
            raise RuntimeError(f"找到多个视频文件:\n{names}\n  请使用 --video 参数指定要打包的视频。")
        video_path = mp4_files[0]

    # ── Annotation subtitle (.annotations.ass) ───────
    annot_path = _find_subtitle_in_dir(output_dir, video_path.stem, ".annotations.ass")

    # ── Main subtitle — bilingual.ass preferred, then zh.srt ──
    sub_path = _find_subtitle_in_dir(output_dir, video_path.stem, ".bilingual.ass")
    if sub_path is not None:
        return video_path, sub_path, "bilingual", annot_path
    sub_path = _find_subtitle_in_dir(output_dir, video_path.stem, ".zh.srt")
    if sub_path is not None:
        return video_path, sub_path, "srt", annot_path

    raise FileNotFoundError(
        "未找到双语字幕 (bilingual.ass) 或中文字幕 (zh.srt) 文件。\n"
        "  请确认已运行翻译管线 (--target-lang zh)，双语运行加 --bilingual。"
    )


def _probe_video_bitrate(ffprobe_bin: Path, video_path: Path) -> int:
    """Extract the video stream bitrate in kbps via ffprobe."""
    cmd = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    bitrate_str = result.stdout.strip()

    if bitrate_str and bitrate_str.isdigit():
        bitrate = int(bitrate_str) // 1000
        if bitrate > 0:
            return bitrate

    # Fallback: compute from format bitrate and duration
    cmd2 = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-show_entries",
        "format=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    fmt_bitrate = result2.stdout.strip()
    if fmt_bitrate and fmt_bitrate.isdigit():
        bitrate = int(fmt_bitrate) // 1000
        if bitrate > 0:
            return bitrate

    # Last resort: sensible default
    return 3000
