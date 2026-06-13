"""Download videos via yt-dlp and derive semantic slugs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def download_video(url: str, output_dir: Path) -> tuple[Path, str]:
    """Download a video from *url* into *output_dir* and return (video_path, slug).

    The slug is derived from the video title (sanitised, 80 chars max).
    The video is saved as ``video.%(ext)s`` inside ``output_dir/<slug>/``.
    """

    # ── Probe title + slug ──
    info_json = _dump_json(url)
    title = info_json.get("title", "video")
    slug = _slugify(title)

    work_dir = output_dir / slug
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──
    cmd = [
        "yt-dlp",
        "-o",
        str(work_dir / "video.%(ext)s"),
        "--no-playlist",
        url,
    ]
    subprocess.run(cmd, check=True)

    # Find the downloaded file (extension may vary: .mp4, .webm, .mkv)
    candidates = list(work_dir.glob("video.*"))
    if not candidates:
        raise FileNotFoundError(f"No video file found in {work_dir} after download")
    video_path = candidates[0]
    return video_path, slug


def derive_slug_from_path(file_path: Path) -> str:
    """Derive a semantic slug from a local file path (stem only)."""
    return _slugify(file_path.stem)


# ── internal helpers ────────────────────────────────────


def _slugify(text: str) -> str:
    """Sanitise *text* into a filesystem-safe slug."""
    # Remove non-word characters (keep CJK, alphanumeric, spaces)
    cleaned = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:80]


def _dump_json(url: str) -> dict:
    """Run ``yt-dlp --dump-json`` and return the parsed dict."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-playlist", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)
