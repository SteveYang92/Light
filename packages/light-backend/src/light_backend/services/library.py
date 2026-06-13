from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..database import (
    get_video,
    insert_chunk,
    insert_video,
    list_chunks,
    update_video,
)


@dataclass
class _ChunkPair:
    """A matched video + subtitle group."""

    video_path: str
    subtitles: dict[str, str]
    segment_label: str  # e.g. "p1", "p2"


def _discover_chunks(output_dir: str, explicit_video: str = "") -> list[_ChunkPair]:
    """Discover video+subtitle pairs in the output directory.

    Handles two patterns:
      1. x-subtitle style: {slug}_p1.mp4, {slug}_p1.zh.srt, {slug}_p2.mp4, ...
      2. Single chunk: zh.srt, en.srt + optional separate video file
    """
    out = Path(output_dir)
    pairs: list[_ChunkPair] = []

    # ── Pattern 1: multi-part — find all video files, match subtitle files ──
    video_exts = ["*.mp4", "*.webm", "*.mkv"]
    all_videos: list[Path] = []
    for ext in video_exts:
        all_videos.extend(out.glob(ext))
    all_videos = sorted(all_videos)

    # Detect multi-part pattern: if any file has "_pN" suffix, exclude the
    # base (unsplit) video that lacks the suffix.
    has_segments = any(re.search(r"_p\d+$", v.stem) for v in all_videos)
    if has_segments:
        video_files = [v for v in all_videos if re.search(r"_p\d+$", v.stem)]
    else:
        video_files = all_videos

    if video_files:
        for video in video_files:
            base = video.stem  # e.g. "video", "Biggest_Mysteries_in_Physics_p1"
            segment = ""
            m = re.search(r"_p(\d+)$", base)
            if m:
                segment = f"p{m.group(1)}"

            subtitles: dict[str, str] = {}
            for f in out.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".srt", ".vtt", ".ass"):
                    continue
                if f.stem == base or f.stem.startswith(base + "."):
                    key = f.name
                    if f.stem.startswith(base + "."):
                        key = f.name[len(base) + 1 :]
                    subtitles[key] = str(f)

            # For single-video dirs: also grab root-level subtitle files
            if len(video_files) == 1:
                for f in out.iterdir():
                    if f.is_file() and f.suffix.lower() in (".srt", ".vtt"):
                        key = f.name
                        if key not in subtitles:
                            subtitles[key] = str(f)

            pairs.append(
                _ChunkPair(
                    video_path=str(video),
                    subtitles=subtitles,
                    segment_label=segment,
                )
            )
        return pairs

    # ── Pattern 2: no mp4 in dir — use explicit video path + subtitle files ──
    if explicit_video and Path(explicit_video).exists():
        subtitles: dict[str, str] = {}
        for f in out.iterdir():
            if f.is_file() and f.suffix.lower() in (".srt", ".vtt", ".ass"):
                subtitles[f.name] = str(f)
        pairs.append(
            _ChunkPair(
                video_path=explicit_video,
                subtitles=subtitles,
                segment_label="",
            )
        )
        return pairs

    # ── Pattern 3: no mp4 and no explicit video — scan for subtitles only ──
    subtitles: dict[str, str] = {}
    for f in out.iterdir():
        if f.is_file() and f.suffix.lower() in (".srt", ".vtt"):
            subtitles[f.name] = str(f)
    if subtitles:
        pairs.append(
            _ChunkPair(
                video_path="",
                subtitles=subtitles,
                segment_label="",
            )
        )

    return pairs


def import_existing_output(
    db_path: str,
    data_dir: str,
    output_dir: str,
    video_path: str = "",
    title: str = "",
) -> dict:
    """Import an existing light-subtitle output directory.

    Automatically discovers video+subtitle pairs. Handles multi-part
    videos (p1/p2/...) by creating one video with multiple chunks.
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    pairs = _discover_chunks(output_dir, video_path)
    if not pairs:
        raise ValueError(f"目录中未发现任何视频或字幕文件: {output_dir}")

    # ── Check: if any pair has no video_path, user must specify it ──
    for p in pairs:
        if not p.video_path:
            found = ", ".join(p.subtitles.keys())
            raise ValueError(f"发现字幕文件 ({found})，但未找到 mp4 视频。请在导入时指定视频文件路径。")

    # ── Determine title and total duration ──
    if not title:
        # Strip segment suffix (e.g. "_p1") and use common base
        base_name = Path(pairs[0].video_path).stem
        base_name = re.sub(r"_p\d+$", "", base_name)
        title = base_name

    total_duration: float | None = None
    chunk_durations: list[float] = []

    for p in pairs:
        dur = None
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
                    p.video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout.strip():
                dur = float(result.stdout.strip())
        except Exception:
            dur = None
        chunk_durations.append(dur or 0)

    total_duration = sum(chunk_durations) if chunk_durations else None

    # ── Create video record ──
    video = insert_video(
        db_path,
        title=title,
        source="import",
        duration=total_duration,
        status="done",
    )
    vid = video["id"]

    # ── Generate thumbnail from first segment ──
    thumb_dir = os.path.join(data_dir, "videos", vid)
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_path = os.path.join(thumb_dir, "thumbnail.jpg")
    try:
        first_dur = chunk_durations[0] if chunk_durations else 30
        subprocess.run(
            [
                "ffmpeg",
                "-ss",
                str(min(first_dur * 0.1, 30)),
                "-i",
                pairs[0].video_path,
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-y",
                thumb_path,
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        if os.path.getsize(thumb_path) > 0:
            update_video(db_path, vid, thumbnail=thumb_path)
    except Exception:
        pass

    # ── Create chunks ──
    for i, p in enumerate(pairs):
        insert_chunk(
            db_path,
            video_id=vid,
            chunk_index=i,
            video_path=p.video_path,
            output_dir=str(out_path),
            duration=chunk_durations[i] if i < len(chunk_durations) else None,
            subtitles=p.subtitles,
        )

    return get_video(db_path, vid)


def get_video_detail(db_path: str, vid: str) -> dict | None:
    """Get video with chunks."""
    video = get_video(db_path, vid)
    if video is None:
        return None
    video["chunks"] = list_chunks(db_path, vid)
    return video


def delete_video_and_files(db_path: str, data_dir: str, vid: str) -> None:
    """Delete video record. For URL-downloaded videos also remove files."""
    from ..database import delete_video as db_delete
    from ..database import get_video as db_get

    video = db_get(db_path, vid)
    if video is None:
        return
    db_delete(db_path, vid)
    # Only clean up data dir for non-imported videos (URL downloads)
    video_dir = os.path.join(data_dir, "videos", vid)
    if os.path.exists(video_dir) and video.get("source") != "import":
        import shutil

        shutil.rmtree(video_dir)
