from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from light_subtitle.config import SubtitleConfig
from light_subtitle.orchestrator import Orchestrator
from light_subtitle.utils.whisper_utils import find_model, find_whisper

from ..database import insert_chunk, insert_run, update_run, update_video

# ── Constants ───────────────────────────────────────────

CHUNK_TARGET_SEC = 45 * 60  # 45 minutes per chunk
SILENCE_DUR = 1.0
SILENCE_THRESH = -35

# ── Video format mappings ───────────────────────────────

format_map = {
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "best": "bestvideo+bestaudio/best",
}

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


def _emit(
    video_id: str, stage: str, progress: float, message: str, chunk: int | None = None, total_chunks: int | None = None
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


# ── yt-dlp download ─────────────────────────────────────


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


def _download_video(
    url: str, output_path: str, fmt: str = "", cookies_from_browser: str = "", cookies_file: str = ""
) -> str:
    """Download video via yt-dlp, return the resulting file path."""
    cmd = ["yt-dlp", "-o", output_path]
    if fmt:
        cmd.insert(1, "-f")
        cmd.insert(2, format_map.get(fmt, fmt))
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(url)
    subprocess.run(cmd, check=True, timeout=3600, capture_output=True, text=True)
    # Find the actual file (yt-dlp may add extensions)
    base = Path(output_path)
    if base.exists():
        return str(base)
    # Try with .mp4 extension
    for p in [Path(f"{output_path}.mp4"), Path(f"{output_path}.webm"), Path(f"{output_path}.mkv")]:
        if p.exists():
            return str(p)
    # Fallback: glob
    parent = base.parent
    for f in parent.iterdir():
        if f.stem == base.stem:
            return str(f)
    raise FileNotFoundError(f"Downloaded file not found at {output_path}")


# ── ffprobe metadata ────────────────────────────────────


def _get_duration(video_path: str) -> float:
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


def _generate_thumbnail(video_path: str, output_path: str, time_sec: float = 10.0) -> None:
    """Extract a frame at the given time for thumbnail."""
    subprocess.run(
        ["ffmpeg", "-ss", str(time_sec), "-i", video_path, "-vframes", "1", "-q:v", "2", output_path],
        capture_output=True,
        timeout=30,
    )


# ── Silence detection + chunking ────────────────────────


def _detect_silence(audio_path: str) -> list[dict]:
    """Use ffmpeg silencedetect, return list of {start, end, duration}."""
    cmd = [
        "ffmpeg",
        "-i",
        audio_path,
        "-af",
        f"silencedetect=noise={SILENCE_THRESH}dB:d={SILENCE_DUR}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    stderr = result.stderr
    silences = []
    pattern = re.compile(
        r"silence_start:\s*([\d.]+)\s*\n.*?silence_end:\s*([\d.]+)\s*\|.*?silence_duration:\s*([\d.]+)",
        re.DOTALL,
    )
    for m in pattern.finditer(stderr):
        silences.append(
            {
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "duration": float(m.group(3)),
            }
        )
    return silences


def _find_split_points(silences: list[dict], total_duration: float) -> list[float]:
    """Find split points at ~45min intervals on silence boundaries."""
    if total_duration <= CHUNK_TARGET_SEC:
        return [0.0, total_duration]

    points = [0.0]
    target = CHUNK_TARGET_SEC
    while target < total_duration - 60:  # leave at least 1min for last chunk
        # Find silence nearest to target, within 30s window
        candidates = [s for s in silences if abs(s["start"] - target) < 30]
        if candidates:
            best = min(candidates, key=lambda s: abs(s["start"] - target))
            points.append(best["start"])
        else:
            points.append(target)
        target += CHUNK_TARGET_SEC
    points.append(total_duration)
    return points


def _extract_audio(video_path: str, output_path: str) -> str:
    """Extract 16kHz mono WAV for silence detection."""
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output_path],
        check=True,
        capture_output=True,
        timeout=600,
    )
    return output_path


def _split_video(video_path: str, split_points: list[float], output_dir: str) -> list[dict]:
    """Split video at the given split points, return list of {path, start, duration}."""
    os.makedirs(output_dir, exist_ok=True)
    chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        duration = split_points[i + 1] - start
        chunk_path = os.path.join(output_dir, f"chunk_{i:03d}.mp4")
        cmd = [
            "ffmpeg",
            "-ss",
            str(start),
            "-i",
            video_path,
            "-t",
            str(duration),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            chunk_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        actual_dur = _get_duration(chunk_path)
        chunks.append({"path": chunk_path, "start": start, "duration": actual_dur, "index": i})
    return chunks


# ── Pipeline execution ──────────────────────────────────


def _scan_subtitles(output_dir: str) -> dict[str, str]:
    """Scan output dir for subtitle files, return {filename: path} (e.g. {"zh.srt": "/...", "zh.vtt": "/..."})."""
    subtitles: dict[str, str] = {}
    for f in Path(output_dir).iterdir():
        if f.is_file() and f.suffix.lower() in (".srt", ".vtt", ".ass"):
            subtitles[f.name] = str(f)
    return subtitles


def _run_pipeline_for_chunk(
    video_path: str,
    output_dir: str,
    config: SubtitleConfig,
    video_id: str,
    chunk_idx: int,
    total_chunks: int,
) -> None:
    """Run full subtitle pipeline for a single chunk."""
    whisper_exe = find_whisper(config.whisper_path)
    whisper_model_full = find_model(config.whisper_model, whisper_exe)
    chunk_config = SubtitleConfig(
        input_path=video_path,
        output_dir=output_dir,
        whisper_model=whisper_model_full,
        whisper_path=whisper_exe,
        language=config.language,
        target_lang=config.target_lang,
        bilingual=config.bilingual,
        cps_limit=config.cps_limit,
        cps_limit_en=config.cps_limit_en,
        max_lines=config.max_lines,
        max_lines_zh=config.max_lines_zh,
        max_chars_per_line_zh=config.max_chars_per_line_zh,
        max_chars_per_line_en=config.max_chars_per_line_en,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        reading_padding=config.reading_padding,
        llm_base_url=config.llm_base_url,
        llm_model=config.llm_model,
        llm_api_key=config.llm_api_key,
        llm_temperature=config.llm_temperature,
        asr=config.asr,
        diarize=config.diarize,
        hf_token=config.hf_token,
        diarize_model=config.diarize_model,
        evaluate_enabled=config.evaluate_enabled,
        quality_threshold=config.quality_threshold,
        correct_enabled=config.correct_enabled,
        context_prep_enabled=config.context_prep_enabled,
        annotate=config.annotate,
        annotation_width=config.annotation_width,
        glossary=config.glossary,
    )

    def prog(stage: str, progress: float, msg: str):
        _emit(video_id, "chunk", progress, msg, chunk=chunk_idx, total_chunks=total_chunks)

    orchestrator = Orchestrator(chunk_config, progress_callback=prog)
    orchestrator.run()


def _find_existing_video(video_dir: str) -> str | None:
    """Find an already-downloaded video file in the directory.

    Looks for 'original*' files or any mp4/webm/mkv in the dir root.
    """
    video_exts = {".mp4", ".webm", ".mkv"}
    # 1. Check for original* files
    for f in Path(video_dir).glob("original*"):
        if f.is_file() and f.suffix.lower() in video_exts:
            return str(f)
    # 2. Fallback: any video file in the directory root
    for f in Path(video_dir).iterdir():
        if f.is_file() and f.suffix.lower() in video_exts:
            return str(f)
    return None


def _download_error_line(exc: subprocess.CalledProcessError) -> str:
    """Extract the last stderr/stdout line from a failed subprocess."""
    for stream in (exc.stderr, exc.stdout):
        if stream:
            lines = stream.strip().splitlines()
            if lines:
                return lines[-1]
    return str(exc)


def run_pipeline(
    video_id: str,
    config: SubtitleConfig,
    db_path: str,
    data_dir: str,
    video_format: str = "",
    cookies_from_browser: str = "",
    cookies_file: str = "",
) -> None:
    """Main pipeline runner. Downloads video, splits into chunks, runs pipeline on each."""
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
    video_dir = os.path.join(data_dir, "videos", video_id)
    os.makedirs(video_dir, exist_ok=True)

    # Step 1: Download (or reuse existing)
    video_path = _find_existing_video(video_dir)
    if video_path:
        _emit(video_id, "download", 1.0, "复用已下载视频")
    else:
        _emit(video_id, "download", 0.0, "下载中...")
        dest_template = os.path.join(video_dir, "original")
        try:
            video_path = _download_video(
                config.input_path, dest_template, video_format, cookies_from_browser, cookies_file
            )
        except subprocess.CalledProcessError as e:
            _emit(video_id, "download", 1.0, f"下载失败: {_download_error_line(e)}")
            update_video(db_path, video_id, status="error")
            _cleanup_thread(video_id)
            return
    duration = _get_duration(video_path)

    # Step 1b: Thumbnail
    thumb_path = os.path.join(video_dir, "thumbnail.jpg")
    _generate_thumbnail(video_path, thumb_path, min(duration * 0.1, 30))
    _emit(video_id, "download", 1.0, "下载完成")

    # Step 2: Silence detection + splitting
    _emit(video_id, "split", 0.0, "检测静音点...")
    chunks_dir = os.path.join(video_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    if duration > CHUNK_TARGET_SEC:
        audio_path = os.path.join(video_dir, "audio_detect.wav")
        _extract_audio(video_path, audio_path)
        silences = _detect_silence(audio_path)
        split_points = _find_split_points(silences, duration)
        chunks = _split_video(video_path, split_points, chunks_dir)
        _emit(video_id, "split", 1.0, f"切分为 {len(chunks)} 段")
        # Clean up audio detection file
        if os.path.exists(audio_path):
            os.remove(audio_path)
    else:
        # Single chunk: copy original into chunks dir for a uniform layout
        single_path = os.path.join(chunks_dir, "chunk_000.mp4")
        shutil.copy2(video_path, single_path)
        chunks = [{"path": single_path, "start": 0.0, "duration": duration, "index": 0}]
        _emit(video_id, "split", 1.0, "无需切分")

    # Step 3: Run pipeline on each chunk
    total = len(chunks)
    rid = insert_run(db_path, video_id)
    update_video(db_path, video_id, status="processing", duration=duration, thumbnail=thumb_path)

    for i, chunk in enumerate(chunks):
        chunk_out_dir = os.path.join(chunks_dir, f"out_{i:03d}")
        os.makedirs(chunk_out_dir, exist_ok=True)
        _emit(video_id, "chunk", 0.0, f"处理第 {i + 1}/{total} 段", chunk=i, total_chunks=total)

        try:
            _run_pipeline_for_chunk(chunk["path"], chunk_out_dir, config, video_id, i, total)
        except Exception as e:
            _emit(video_id, "chunk", 1.0, f"第 {i + 1} 段失败: {e}", chunk=i, total_chunks=total)
            update_run(db_path, rid, status="error", error_msg=str(e), current_chunk=i + 1, total_chunks=total)
            update_video(db_path, video_id, status="error")
            return

        # Register chunk in DB
        subtitles = _scan_subtitles(chunk_out_dir)
        insert_chunk(
            db_path,
            video_id=video_id,
            chunk_index=i,
            video_path=chunk["path"],
            output_dir=chunk_out_dir,
            duration=chunk["duration"],
            subtitles=subtitles,
        )

    # Mark done
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    update_run(
        db_path,
        rid,
        status="done",
        progress=1.0,
        stage="done",
        current_chunk=total,
        total_chunks=total,
        finished_at=now,
    )
    update_video(db_path, video_id, status="done")
    _emit(video_id, "done", 1.0, "全部完成")
    _cleanup_thread(video_id)


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
