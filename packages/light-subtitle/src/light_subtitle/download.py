"""Download videos via yt-dlp and derive semantic slugs.

Supports cached downloads: once a URL has been downloaded, subsequent runs
skip yt-dlp entirely (both metadata probe and download) by looking up the
persistent URL → slug mapping.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

_URL_SLUG_MAP = "url_slug_cache.json"

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError as _DownloadError
except ImportError:
    YoutubeDL = None  # type: ignore[assignment]
    _DownloadError = Exception


def _make_ydl(opts: dict[str, Any]) -> Any:
    """Create a yt-dlp YoutubeDL instance with the given options."""
    if YoutubeDL is None:
        raise ImportError("yt-dlp is not installed")
    return YoutubeDL(opts)


def download_video(
    url: str,
    output_dir: Path,
    *,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, str]:
    """Download a video from *url* into *output_dir* and return (video_path, slug).

    The slug is derived from the video title (sanitised, 80 chars max).
    The video is saved as ``video.%(ext)s`` inside ``output_dir/<slug>/``.

    On success the URL → slug mapping is persisted so future runs can skip
    both the metadata probe and the download.

    *progress* is called with (fraction, message) on each download status
    update (``"downloading"`` / ``"finished"`` status from yt-dlp hooks).
    When the total byte size is unknown, fraction is clamped to 0.0.
    """

    # ── Probe title + slug ──
    info_json = _dump_json(url)
    title = info_json.get("title", "video")
    slug = _slugify(title)

    work_dir = output_dir / slug
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Download (yt-dlp Python API) ──
    outtmpl = str(work_dir / "video.%(ext)s")

    def _hook(d: dict) -> None:
        if progress is None:
            return
        status = d.get("status")
        if status == "downloading":
            downloaded = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total and total > 0:
                frac = downloaded / total
                pct = round(frac * 100)
                progress(frac, f"下载中... {pct}%")
        elif status == "finished":
            progress(1.0, "下载完成")

    ydl_opts = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook],
    }

    try:
        from . import logger

        with logger.capture_external_output():
            ydl = _make_ydl(ydl_opts)
            ydl.download([url])
    except Exception as e:
        if isinstance(e, _DownloadError):
            raise RuntimeError(f"yt-dlp download failed: {e!s}") from e
        raise

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


def probe_slug(url: str) -> str:
    """Probe the video title from *url* and derive a slug (no download)."""
    info_json = _dump_json(url)
    title = info_json.get("title", "video")
    return _slugify(title)


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
