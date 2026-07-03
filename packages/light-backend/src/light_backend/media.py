from __future__ import annotations

from pathlib import Path

AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".opus", ".weba"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mkv"})
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

MEDIA_GLOBS = [f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS)]

_STREAM_MIME: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
    ".weba": "audio/webm",
    ".srt": "text/plain; charset=utf-8",
    ".vtt": "text/vtt; charset=utf-8",
    ".ass": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def media_kind(path: str) -> str:
    """Return ``audio`` or ``video`` based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "video"


def guess_stream_mime(path: str) -> str:
    """Guess Content-Type for streaming or static file serving."""
    ext = Path(path).suffix.lower()
    return _STREAM_MIME.get(ext, "application/octet-stream")
