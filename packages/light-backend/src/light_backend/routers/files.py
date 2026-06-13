from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..database import get_chunk, get_video
from ..state import get_config

router = APIRouter(prefix="/api/chunks", tags=["files"])

# Read in modest blocks so a single request never loads the whole file into RAM.
STREAM_BLOCK = 256 * 1024

# lang/fmt must be simple tokens — reject path traversal in URL segments.
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


class RangeNotSatisfiable(Exception):
    """Raised when a Range header cannot be satisfied."""

    def __init__(self, file_size: int) -> None:
        self.file_size = file_size


def _guess_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".srt": "text/plain; charset=utf-8",
        ".vtt": "text/vtt; charset=utf-8",
        ".ass": "text/plain; charset=utf-8",
        ".json": "application/json",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "application/octet-stream")


def _is_under_dir(candidate: str, base_dir: str) -> bool:
    """Return True when candidate resolves inside base_dir."""
    if not base_dir:
        return False
    try:
        resolved = Path(candidate).resolve()
        base = Path(base_dir).resolve()
        return resolved.is_relative_to(base)
    except (OSError, ValueError):
        return False


def _safe_existing_path(candidate: str, base_dir: str) -> str | None:
    """Return candidate only when it exists and stays within base_dir."""
    if not candidate or not os.path.exists(candidate):
        return None
    if _is_under_dir(candidate, base_dir):
        return candidate
    return None


def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int]:
    """Parse a single HTTP Range header (bytes=…) into inclusive start/end."""
    if not range_header.startswith("bytes="):
        raise ValueError("unsupported range unit")

    range_val = range_header[6:].strip()
    if "," in range_val:
        raise ValueError("multipart ranges are not supported")

    start_str, _, end_str = range_val.partition("-")

    try:
        if not start_str and end_str:
            # Suffix range: bytes=-N (last N bytes)
            suffix_len = int(end_str)
            if suffix_len <= 0:
                raise RangeNotSatisfiable(file_size)
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        elif start_str and not end_str:
            # Open-ended: bytes=N-
            start = int(start_str)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
    except ValueError as exc:
        raise ValueError("invalid range values") from exc

    end = min(end, file_size - 1)
    if start < 0 or start >= file_size or start > end:
        raise RangeNotSatisfiable(file_size)
    return start, end


def _read_block(path: str, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


async def _async_iter_file_range(path: str, start: int, end: int) -> AsyncIterator[bytes]:
    """Yield bytes [start, end] without blocking the event loop."""
    remaining = end - start + 1
    offset = start
    while remaining > 0:
        to_read = min(STREAM_BLOCK, remaining)
        block = await asyncio.to_thread(_read_block, path, offset, to_read)
        if not block:
            break
        remaining -= len(block)
        offset += len(block)
        yield block


def _resolve_subtitle_path(chunk: dict, lang: str, fmt: str) -> str | None:
    """Resolve subtitle file path from chunk metadata and output dir."""
    if not _SAFE_NAME.match(lang) or not _SAFE_NAME.match(fmt):
        return None

    output_dir = chunk.get("output_dir", "")
    if not output_dir:
        return None

    subtitles = chunk.get("subtitles", {})
    sub_key = f"{lang}.{fmt}"
    sub_path = subtitles.get(sub_key)

    if sub_path:
        safe = _safe_existing_path(sub_path, output_dir)
        if safe:
            return safe

    # Import-style keys, e.g. "video_p2.zh.srt" or "video_p2.annotations.vtt"
    if lang == "annotations":
        key_suffix = f".annotations.{fmt}"
    else:
        key_suffix = f".{lang}.{fmt}"
    for key, path in subtitles.items():
        if key.endswith(key_suffix):
            safe = _safe_existing_path(path, output_dir)
            if safe:
                return safe

    # Pipeline layout: out_NNN/zh.srt or annotations.vtt
    direct = os.path.join(output_dir, sub_key)
    safe = _safe_existing_path(direct, output_dir)
    if safe:
        return safe

    # Import layout: {video_stem}.zh.srt in shared output dir
    video_path = chunk.get("video_path", "")
    if video_path:
        stem = Path(video_path).stem
        if lang == "annotations":
            stemmed = os.path.join(output_dir, f"{stem}.annotations.{fmt}")
        else:
            stemmed = os.path.join(output_dir, f"{stem}.{lang}.{fmt}")
        safe = _safe_existing_path(stemmed, output_dir)
        if safe:
            return safe

    return None


@router.get("/{chunk_id}/stream")
async def stream_chunk(chunk_id: str, range_header: str | None = Header(default=None, alias="Range")):
    """Stream video chunk with Range header support."""
    cfg = get_config()
    chunk = get_chunk(cfg.db_path, chunk_id)
    if chunk is None:
        raise HTTPException(404, "Chunk not found")

    video_path = chunk["video_path"]
    if not os.path.exists(video_path):
        raise HTTPException(404, "Video file not found")

    file_size = os.path.getsize(video_path)
    mime = _guess_mime(video_path)

    if range_header:
        try:
            start, end = _parse_byte_range(range_header, file_size)
        except RangeNotSatisfiable as exc:
            raise HTTPException(
                416,
                "Range not satisfiable",
                headers={"Content-Range": f"bytes */{exc.file_size}"},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                416,
                "Range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"},
            ) from exc

        content_length = end - start + 1
        return StreamingResponse(
            _async_iter_file_range(video_path, start, end),
            status_code=206,
            media_type=mime,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(content_length),
                "Accept-Ranges": "bytes",
            },
        )

    return FileResponse(video_path, media_type=mime, headers={"Accept-Ranges": "bytes"})


@router.get("/{chunk_id}/subtitles/{lang}.{fmt}")
async def get_subtitle(chunk_id: str, lang: str, fmt: str):
    """Get subtitle file for a chunk."""
    cfg = get_config()
    chunk = get_chunk(cfg.db_path, chunk_id)
    if chunk is None:
        raise HTTPException(404, "Chunk not found")

    sub_path = _resolve_subtitle_path(chunk, lang, fmt)

    if not sub_path:
        raise HTTPException(404, f"Subtitle {lang}.{fmt} not found")

    mime = _guess_mime(sub_path)
    return FileResponse(sub_path, media_type=mime)


@router.get("/{chunk_id}/thumbnail")
async def get_chunk_thumbnail(chunk_id: str):
    """Get thumbnail for chunk's parent video."""
    cfg = get_config()
    chunk = get_chunk(cfg.db_path, chunk_id)
    if chunk is None:
        raise HTTPException(404, "Chunk not found")
    video = get_video(cfg.db_path, chunk["video_id"])
    if video and video.get("thumbnail"):
        thumb_path = video["thumbnail"]
        if os.path.exists(thumb_path):
            return FileResponse(thumb_path, media_type=_guess_mime(thumb_path))
    raise HTTPException(404, "Thumbnail not found")
