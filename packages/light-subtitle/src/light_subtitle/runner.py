"""Shared pipeline runner — download (URL) → split (long) → orchestrate → merge.

Usable by both the CLI and the web backend so that download, split, and
pipeline orchestration live in one place.

Usage::

    from light_subtitle.runner import process_video
    result = process_video(config, progress_callback=my_callback)
    # result.output_dir -> Path with merged subtitles
    # result.slug       -> semantic name
    # result.video_path -> original downloaded video
    # result.success    -> True when pipeline completed normally

Progress callback signature: ``(stage: str, progress: float, message: str)``
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import SubtitleConfig
from .download import download_video, find_cached_download
from .merge_outputs import merge_all
from .orchestrator import Orchestrator
from .video_split import (
    compute_split_points,
    find_existing_segments,
    find_existing_split_points,
    should_split,
    split_video,
)

ProgressCallback = Callable[[str, float, str], None] | None

_DEFAULT_SPLIT_THRESHOLD = 2700  # 45 minutes
_DEFAULT_OVERLAP = 10


@dataclass
class ProcessResult:
    """Summary returned after a pipeline run."""

    output_dir: Path  # work directory containing merged outputs
    slug: str  # semantic slug
    video_path: Path  # original (downloaded) video file
    success: bool = True  # False when segments failed or were interrupted


def process_video(
    config: SubtitleConfig,
    progress_callback: ProgressCallback = None,
) -> ProcessResult:
    """Run the full video → subtitles pipeline.

    Downloads the video if *config.url* is set (skips when a cached download
    exists), splits long videos at silence boundaries, processes each segment
    with pipelined ASR concurrency, and merges results.

    Returns a ``ProcessResult`` with the output directory, slug, and original
    video path.
    """
    prog = progress_callback or (lambda _s, _p, _m: None)

    # ── 1. Download (or reuse cached) ──
    if config.url:
        cached = find_cached_download(config.url, Path(config.output_dir))
        if cached is not None:
            video_path, slug = cached
            prog("download", 1.0, "复用已下载视频")
        else:
            prog("download", 0.0, "下载中…")
            video_path, slug = download_video(config.url, Path(config.output_dir))
            prog("download", 1.0, "下载完成")
        is_long = should_split(video_path, threshold=_DEFAULT_SPLIT_THRESHOLD)
    else:
        video_path = Path(config.input_path).resolve()
        slug = config.slug or _slugify(Path(config.input_path).stem)
        is_long = should_split(video_path, threshold=_DEFAULT_SPLIT_THRESHOLD)

    work_dir = video_path.parent if config.url else Path(config.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Split / process ──
    if is_long:
        success = _process_long(config, video_path, slug, work_dir, prog)
    else:
        _process_short(config, video_path, work_dir, prog)
        success = True

    return ProcessResult(output_dir=work_dir, slug=slug, video_path=video_path, success=success)


# ── Short video ─────────────────────────────────────────


def _process_short(
    config: SubtitleConfig,
    video_path: Path,
    work_dir: Path,
    prog: Callable[[str, float, str], None],
) -> None:
    """Run the pipeline directly on a short (≤45 min) video."""
    seg_config = config.clone_for_segment(
        input_path=str(video_path),
        output_dir=str(work_dir),
    )
    Orchestrator(seg_config, progress_callback=prog).run()
    prog("done", 1.0, "全部完成")


# ── Long video ──────────────────────────────────────────


def _process_long(
    config: SubtitleConfig,
    video_path: Path,
    slug: str,
    work_dir: Path,
    prog: Callable[[str, float, str], None],
) -> bool:
    """Split + pipeline + merge for videos longer than 45 minutes.

    ASR for segment N+1 runs concurrently with post-ASR (correct, translate,
    etc.) of segment N.  Only one ASR runs at a time, gated by a
    threading.Event.

    Returns True when all segments completed and were merged successfully.
    """
    overlap = _DEFAULT_OVERLAP

    # ── Split (or reuse existing segments) ──
    seg_dirs = find_existing_segments(work_dir)
    if seg_dirs is not None:
        points = find_existing_split_points(work_dir)
        if points is None:
            points = compute_split_points(video_path, target_duration=_DEFAULT_SPLIT_THRESHOLD)
        prog("split", 1.0, f"复用 {len(seg_dirs)} 个分段")
    else:
        prog("split", 0.0, "检测分块点…")
        points = compute_split_points(video_path, target_duration=_DEFAULT_SPLIT_THRESHOLD)
        seg_dirs = split_video(video_path, points, overlap=overlap, seg_dir_template=".seg")
        prog("split", 1.0, f"切分为 {len(seg_dirs)} 段")

    # ── Build per-segment configs ──
    seg_configs: list[SubtitleConfig] = []
    for seg_dir in seg_dirs:
        seg_video = next(seg_dir.glob("video.*"), None)
        if seg_video is None:
            continue
        seg_configs.append(
            config.clone_for_segment(
                input_path=str(seg_video),
                output_dir=str(seg_dir),
            )
        )

    if not seg_configs:
        return False

    # ── Pipelined concurrency ──
    shutdown = threading.Event()
    asr_ready = threading.Event()
    asr_ready.set()  # first segment can start ASR immediately

    futures: list = []
    segment_failed = False
    with ThreadPoolExecutor(max_workers=len(seg_configs) + 1) as executor:
        for _i, cfg in enumerate(seg_configs):
            asr_ready.wait()
            if shutdown.is_set():
                break
            asr_ready.clear()

            orch = Orchestrator(cfg, progress_callback=prog, on_asr_complete=asr_ready.set, shutdown_event=shutdown)
            futures.append(executor.submit(orch.run))

    for f in futures:
        try:
            f.result()
        except Exception:
            segment_failed = True

    if shutdown.is_set() or segment_failed:
        return False

    # ── Merge ──
    prog("merge", 0.0, "合并分段…")
    merge_all(seg_dirs[0].parent, slug, overlap=overlap)
    prog("merge", 1.0, "合并完成")
    prog("done", 1.0, "全部完成")
    return True


# ── helpers ─────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Derive a filesystem-safe slug from *text*."""
    import re

    cleaned = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:80]
