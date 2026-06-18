"""Pipeline service — delegates video processing to light-subtitle's shared runner.

The backend is a thin layer: it receives a URL from the frontend, passes it
to ``light_subtitle.runner.process_video()`` (which handles download, split,
ASR, translation, and merge), then scans the output to populate the database
and generate a thumbnail.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from light_subtitle.config import SubtitleConfig
from light_subtitle.runner import process_video

from ..database import insert_chunk, insert_run, update_run, update_video

# ── Event queue registry ────────────────────────────────

_event_queues: dict[str, asyncio.Queue] = {}
_lock = threading.Lock()

_pipeline_threads: dict[str, threading.Thread] = {}

# Stored event loop reference for cross-thread event emission
_main_loop: asyncio.AbstractEventLoop | None = None


def _set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    if _main_loop is None:
        _main_loop = loop


def get_event_queue(video_id: str) -> asyncio.Queue:
    with _lock:
        if video_id not in _event_queues:
            _event_queues[video_id] = asyncio.Queue()
        return _event_queues[video_id]


def cleanup_event_queue(video_id: str) -> None:
    with _lock:
        _event_queues.pop(video_id, None)


def _get_video_title(url: str) -> str:
    """Extract video title via yt-dlp --dump-json."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
        title = data.get("title", "Untitled")
        title = re.sub(r"[^\w\s]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title[:120]
    except Exception:
        return "Untitled"


def _emit(
    video_id: str,
    stage: str,
    progress: float,
    message: str,
    chunk: int | None = None,
    total_chunks: int | None = None,
) -> None:
    """Thread-safe event emission via stored main event loop."""
    if _main_loop is None:
        return
    _main_loop.call_soon_threadsafe(
        lambda: get_event_queue(video_id).put_nowait(
            {
                "stage": stage,
                "progress": progress,
                "message": message,
                "chunk": chunk,
                "total_chunks": total_chunks,
            }
        )
    )


# ── Pipeline execution ──────────────────────────────────


def run_pipeline(
    video_id: str,
    config: SubtitleConfig,
    db_path: str,
    data_dir: str,
    video_format: str = "",
    cookies_from_browser: str = "",
    cookies_file: str = "",
) -> None:
    """Main pipeline entry point. Runs in a background thread."""
    try:
        _run_pipeline_body(
            video_id,
            config,
            db_path,
            data_dir,
            video_format,
            cookies_from_browser,
            cookies_file,
        )
    except Exception as e:
        _emit(video_id, "error", 1.0, f"管线失败: {e}")
        update_video(db_path, video_id, status="error")
        _cleanup_thread(video_id)


def _run_pipeline_body(
    video_id: str,
    config: SubtitleConfig,
    db_path: str,
    data_dir: str,
    video_format: str = "",
    cookies_from_browser: str = "",
    cookies_file: str = "",
) -> None:
    output_dir = os.path.join(data_dir, "videos", video_id)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Build SubtitleConfig for light-subtitle runner ──
    sub_config = SubtitleConfig(
        input_path=config.input_path,  # URL
        output_dir=output_dir,
        url=config.input_path,
        target_lang=config.target_lang,
        bilingual=config.bilingual,
        whisper_model=config.whisper_model,
        language=config.language,
        llm_model=config.llm_model,
        llm_base_url=config.llm_base_url,
        llm_api_key=config.llm_api_key,
        llm_temperature=config.llm_temperature,
        diarize=config.diarize,
        annotate=config.annotate,
        resume=config.resume,
        cps_limit=config.cps_limit,
        cps_limit_en=config.cps_limit_en,
        max_lines=config.max_lines,
        max_lines_zh=config.max_lines_zh,
        max_chars_per_line_zh=config.max_chars_per_line_zh,
        max_chars_per_line_en=config.max_chars_per_line_en,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        reading_padding=config.reading_padding,
        asr=config.asr,
        diarize_model=config.diarize_model,
        hf_token=config.hf_token,
        evaluate_enabled=config.evaluate_enabled,
        quality_threshold=config.quality_threshold,
        correct_enabled=config.correct_enabled,
        context_prep_enabled=config.context_prep_enabled,
        annotation_width=config.annotation_width,
        glossary=config.glossary,
    )

    # ── 2. Mark processing ──
    rid = insert_run(db_path, video_id)
    update_video(db_path, video_id, status="processing")

    # ── 3. Progress callback → SSE events ──
    def on_progress(stage: str, progress: float, msg: str):
        _emit(video_id, stage, progress, msg)

    # ── 4. Delegate to light-subtitle runner ──
    result = process_video(sub_config, progress_callback=on_progress)

    if not result.success:
        _emit(video_id, "error", 1.0, "管线未完成（可重试断点续跑）")
        update_run(db_path, rid, status="error", error_msg="管线未完成")
        update_video(db_path, video_id, status="error")
        _cleanup_thread(video_id)
        return

    # ── 5. Post-process: thumbnail + scan output + populate DB ──
    _post_process(video_id, result, db_path, data_dir, rid)


def _post_process(
    video_id: str,
    result,
    db_path: str,
    data_dir: str,
    rid: str,
) -> None:
    """Scan runner output, generate thumbnail, populate video/chunk/run in DB."""
    work_dir = result.output_dir
    slug = result.slug

    # ── Find merged video ──
    merged_video: Path | None = None
    for ext in (".mp4", ".webm", ".mkv"):
        candidate = work_dir / f"{slug}{ext}"
        if candidate.exists():
            merged_video = candidate
            break
    if merged_video is None:
        # Fallback: look for any video file (short local input keeps generic name)
        for ext in (".mp4", ".webm", ".mkv"):
            for f in work_dir.glob(f"*{ext}"):
                merged_video = f
                break
            if merged_video:
                break
    if merged_video is None:
        merged_video = result.video_path

    # ── Duration ──
    duration = _probe_duration(str(merged_video))

    # ── Thumbnail ──
    thumb_path = os.path.join(data_dir, "videos", video_id, "thumbnail.jpg")
    _make_thumbnail(str(merged_video), thumb_path, min((duration or 30) * 0.1, 30))

    # ── Scan subtitles ──
    subtitles: dict[str, str] = {}
    for f in work_dir.iterdir():
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix in (".srt", ".vtt", ".ass", ".json"):
            # Use key relative to slug prefix (e.g. "zh.srt", "annotations.ass")
            key = f.name
            if f.name.startswith(f"{slug}."):
                key = f.name[len(slug) + 1 :]
            subtitles[key] = str(f)

    # ── Update DB ──
    update_video(db_path, video_id, status="done", duration=duration, thumbnail=thumb_path)

    insert_chunk(
        db_path,
        video_id=video_id,
        chunk_index=0,
        video_path=str(merged_video),
        output_dir=str(work_dir),
        duration=duration,
        subtitles=subtitles,
    )

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    update_run(
        db_path,
        rid,
        status="done",
        progress=1.0,
        stage="done",
        current_chunk=1,
        total_chunks=1,
        finished_at=now,
    )

    _emit(video_id, "done", 1.0, "全部完成")
    _cleanup_thread(video_id)


# ── Helpers ─────────────────────────────────────────────


def _probe_duration(video_path: str) -> float | None:
    """Probe video duration via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _make_thumbnail(video_path: str, output_path: str, time_sec: float = 10.0) -> None:
    """Extract a frame at the given time for thumbnail."""
    try:
        subprocess.run(
            ["ffmpeg", "-ss", str(time_sec), "-i", video_path, "-vframes", "1", "-q:v", "2", "-y", output_path],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass


# ── Thread management ───────────────────────────────────


def _cleanup_thread(video_id: str) -> None:
    """Remove finished pipeline thread from registry."""
    with _lock:
        _pipeline_threads.pop(video_id, None)


def start_pipeline_thread(
    video_id: str,
    config: SubtitleConfig,
    db_path: str,
    data_dir: str,
    video_format: str = "",
    cookies_from_browser: str = "",
    cookies_file: str = "",
) -> None:
    """Start pipeline in a background thread."""
    thread = threading.Thread(
        target=run_pipeline,
        args=(video_id, config, db_path, data_dir, video_format, cookies_from_browser, cookies_file),
        daemon=True,
    )
    _pipeline_threads[video_id] = thread
    thread.start()
