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

from . import logger
from .config import SubtitleConfig
from .download import download_video, find_cached_download
from .merge_outputs import merge_all
from .orchestrator import Orchestrator
from .reporting import (
    STAGE_DONE,
    STAGE_DOWNLOAD,
    STAGE_MERGE,
    STAGE_SPLIT,
    ProgressEvent,
    Reporter,
    RunEvent,
    RunKind,
    SegmentRef,
    StageStatus,
    as_reporter,
)
from .video_split import (
    compute_split_points,
    find_existing_segments,
    find_existing_split_points,
    segment_tag,
    should_split,
    split_video,
)

ProgressCallback = Callable[[str, float, str], None] | None

_DEFAULT_OVERLAP = 10


def _emit(reporter: Reporter, stage: str, status: StageStatus, progress: float, message: str) -> None:
    """Emit a run-level progress event (segment=None)."""
    reporter.emit(ProgressEvent(stage=stage, status=status, progress=progress, message=message, segment=None))


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
    reporter = as_reporter(progress_callback)

    # ── 1. Download (or reuse cached) ──
    if config.url:
        cached = find_cached_download(config.url, Path(config.output_dir))
        if cached is not None:
            video_path, slug = cached
            _emit(reporter, STAGE_DOWNLOAD, StageStatus.finished, 1.0, "复用已下载视频")
        else:
            _emit(reporter, STAGE_DOWNLOAD, StageStatus.started, 0.0, "下载中…")
            video_path, slug = download_video(
                config.url,
                Path(config.output_dir),
                progress=lambda f, m: _emit(reporter, STAGE_DOWNLOAD, StageStatus.progress, f, m),
            )
            _emit(reporter, STAGE_DOWNLOAD, StageStatus.finished, 1.0, "下载完成")
        is_long = should_split(video_path, threshold=config.split_threshold)
    else:
        video_path = Path(config.input_path).resolve()
        slug = config.slug or _slugify(Path(config.input_path).stem)
        is_long = should_split(video_path, threshold=config.split_threshold)

    work_dir = video_path.parent if config.url else Path(config.output_dir)

    reporter.emit(
        RunEvent(
            RunKind.started,
            {
                "slug": slug,
                "mode": (
                    "bilingual"
                    if config.bilingual
                    else (f"translate→{config.target_lang}" if config.target_lang else "source-only")
                ),
                "input": str(config.url or config.input_path),
                "output": str(work_dir),
            },
        )
    )

    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Split / process ──
    if is_long:
        success = _process_long(config, video_path, slug, work_dir, reporter)
    else:
        _process_short(config, video_path, work_dir, reporter)
        success = True

    return ProcessResult(output_dir=work_dir, slug=slug, video_path=video_path, success=success)


# ── Short video ─────────────────────────────────────────


def _process_short(
    config: SubtitleConfig,
    video_path: Path,
    work_dir: Path,
    reporter: Reporter,
) -> None:
    """Run the pipeline directly on a short (≤45 min) video."""
    seg_config = config.clone_for_segment(
        input_path=str(video_path),
        output_dir=str(work_dir),
    )
    Orchestrator(seg_config, progress_callback=reporter).run()
    _emit(reporter, STAGE_DONE, StageStatus.finished, 1.0, "全部完成")


# ── Long video ──────────────────────────────────────────


def _process_long(
    config: SubtitleConfig,
    video_path: Path,
    slug: str,
    work_dir: Path,
    reporter: Reporter,
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
            points = compute_split_points(video_path, target_duration=config.split_threshold)
        _emit(reporter, STAGE_SPLIT, StageStatus.finished, 1.0, f"复用 {len(seg_dirs)} 个分段")
    else:
        _emit(reporter, STAGE_SPLIT, StageStatus.started, 0.0, "检测分块点…")
        points = compute_split_points(video_path, target_duration=config.split_threshold)
        seg_dirs = split_video(video_path, points, overlap=overlap, seg_dir_template=".seg")
        _emit(reporter, STAGE_SPLIT, StageStatus.finished, 1.0, f"切分为 {len(seg_dirs)} 段")

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

            orch = Orchestrator(
                cfg,
                progress_callback=reporter,
                on_asr_complete=asr_ready.set,
                shutdown_event=shutdown,
                segment=SegmentRef(_i, len(seg_configs), segment_tag(cfg.output_dir)),
            )
            futures.append(executor.submit(orch.run))

    for f in futures:
        try:
            f.result()
        except Exception as e:
            segment_failed = True
            logger.warning(f"  Segment failed: {type(e).__name__}: {e}")

    if shutdown.is_set() or segment_failed:
        return False

    # ── Merge ──
    # Bind the merge phase to a pipeline log in work_dir (the per-segment
    # Orchestrators logged into their own .segN/ dirs; the main thread had
    # no file logger, so merge messages previously went nowhere on disk).
    logger.init(work_dir)
    _emit(reporter, STAGE_MERGE, StageStatus.started, 0.0, "合并分段…")
    merge_all(seg_dirs[0].parent, slug, overlap=overlap)
    _emit(reporter, STAGE_MERGE, StageStatus.finished, 1.0, "合并完成")
    _emit(reporter, STAGE_DONE, StageStatus.finished, 1.0, "全部完成")
    return True


# ── helpers ─────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Derive a filesystem-safe slug from *text*."""
    import re

    cleaned = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:80]
