"""Download videos via yt-dlp and derive semantic slugs.

Supports cached downloads: once a URL has been downloaded, subsequent runs
skip yt-dlp entirely (both metadata probe and download) by looking up the
persistent URL → slug mapping.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_URL_SLUG_MAP = "url_slug_cache.json"


def download_video(url: str, output_dir: Path) -> tuple[Path, str]:
    """Download a video from *url* into *output_dir* and return (video_path, slug).

    The slug is derived from the video title (sanitised, 80 chars max).
    The video is saved as ``video.%(ext)s`` inside ``output_dir/<slug>/``.

    On success the URL → slug mapping is persisted so future runs can skip
    both the metadata probe and the download.
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

    # ── Cache the URL → slug mapping ──
    _save_url_slug(url, slug, output_dir)

    return video_path, slug


def find_cached_download(url: str, output_dir: Path) -> tuple[Path, str] | None:
    """Return ``(video_path, slug)`` if *url* has been downloaded before.

    Checks the persistent URL → slug mapping stored in *output_dir*.
    Returns ``None`` if the URL hasn't been seen, the slug directory is
    missing, or no ``video.*`` file exists inside it.
    """
    mapping = _load_url_slug_map(output_dir)
    slug = mapping.get(url)
    if slug is None:
        return None

    work_dir = output_dir / slug
    if not work_dir.is_dir():
        return None

    candidates = list(work_dir.glob("video.*"))
    if not candidates:
        return None

    return candidates[0], slug


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


def _load_url_slug_map(output_dir: Path) -> dict[str, str]:
    """Load the persistent URL → slug mapping from *output_dir*."""
    path = output_dir / _URL_SLUG_MAP
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_url_slug(url: str, slug: str, output_dir: Path) -> None:
    """Persist *url* → *slug* mapping so subsequent runs can skip yt-dlp."""
    mapping = _load_url_slug_map(output_dir)
    mapping[url] = slug
    path = output_dir / _URL_SLUG_MAP
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def _dump_json(url: str) -> dict:
    """Run ``yt-dlp --dump-json`` and return the parsed dict."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-playlist", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)
