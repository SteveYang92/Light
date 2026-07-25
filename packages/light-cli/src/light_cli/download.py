"""Download videos via yt-dlp and derive semantic slugs.

Supports cached downloads: once a URL has been downloaded, subsequent runs
skip yt-dlp entirely (both metadata probe and download) by looking up the
persistent URL → slug mapping.

Video files stay as ``video.*`` inside ``output/<slug>/``.  Lookup also accepts
legacy ``{slug}.*`` renames so older runs resume without re-downloading.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_URL_SLUG_MAP = "url_slug_cache.json"
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}

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
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> tuple[Path, str]:
    """Download a video from *url* into *output_dir* and return (video_path, slug).

    The slug is derived from the video title (sanitised, 80 chars max).
    The video is saved as ``video.%(ext)s`` inside ``output_dir/<slug>/``.

    On success the URL → slug mapping is persisted so future runs can skip
    both the metadata probe and the download.  If a video file already exists
    in the slug directory (``video.*`` or legacy ``{slug}.*``), returns it
    without calling yt-dlp download.

    *progress* is called with (fraction, message) on each download status
    update (``"downloading"`` / ``"finished"`` status from yt-dlp hooks).
    When the total byte size is unknown, fraction is clamped to 0.0.

    *cookies_from_browser* / *cookies_file* are passed through to yt-dlp for
    age-restricted or login-gated sites (same as ``--cookies-from-browser`` /
    ``--cookies``).
    """

    # ── Probe title + slug (can take seconds; surface via progress) ──
    if progress is not None:
        progress(0.0, "获取视频信息…")
    info_json = _dump_json(
        url,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    title = info_json.get("title", "video")
    slug = _slugify(title)

    work_dir = output_dir / slug
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if video already exists (resume / legacy rename) ──
    existing = find_video_in_dir(work_dir, slug)
    if existing is not None:
        _save_url_slug(url, slug, output_dir)
        return existing, slug

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
                # Stage label + progress bar carry status; no redundant message.
                progress(downloaded / total, "")
        elif status == "finished":
            progress(1.0, "下载完成")

    ydl_opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook],
    }
    ydl_opts.update(_cookie_ydl_opts(cookies_from_browser, cookies_file))

    try:
        from light_core import logger

        with logger.capture_external_output():
            ydl = _make_ydl(ydl_opts)
            ydl.download([url])
    except Exception as e:
        if isinstance(e, _DownloadError):
            raise RuntimeError(f"yt-dlp download failed: {e!s}") from e
        raise

    video_path = find_video_in_dir(work_dir, slug)
    if video_path is None:
        raise FileNotFoundError(f"No video file found in {work_dir} after download")

    _save_url_slug(url, slug, output_dir)
    return video_path, slug


def find_cached_download(url: str, output_dir: Path) -> tuple[Path, str] | None:
    """Return ``(video_path, slug)`` if *url* has been downloaded before.

    Checks the persistent URL → slug mapping stored in *output_dir*.
    Returns ``None`` if the URL hasn't been seen, the slug directory is
    missing, or no video file exists inside it (accepts ``video.*`` or the
    legacy ``{slug}.*`` form).
    """
    mapping = _load_url_slug_map(output_dir)
    slug = mapping.get(url)
    if slug is None:
        slug = mapping.get(_canonical_url(url))
    if slug is None:
        canon = _canonical_url(url)
        for key, value in mapping.items():
            if _canonical_url(key) == canon:
                slug = value
                break
    if slug is None:
        return None

    work_dir = output_dir / slug
    if not work_dir.is_dir():
        return None

    video_path = find_video_in_dir(work_dir, slug)
    if video_path is None:
        return None

    return video_path, slug


def find_video_in_dir(work_dir: Path, slug: str | None = None) -> Path | None:
    """Return a video file in *work_dir*, preferring ``video.*`` over ``{slug}.*``."""
    if not work_dir.is_dir():
        return None
    for path in sorted(work_dir.glob("video.*")):
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES:
            return path
    if slug:
        for path in sorted(work_dir.glob(f"{slug}.*")):
            if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES:
                return path
    return None


def derive_slug_from_path(file_path: Path) -> str:
    """Derive a semantic slug from a local file path (stem only)."""
    return _slugify(file_path.stem)


def probe_slug(
    url: str,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> str:
    """Probe the video title from *url* and derive a slug (no download)."""
    info_json = _dump_json(
        url,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    title = info_json.get("title", "video")
    return _slugify(title)


# ── internal helpers ────────────────────────────────────


def _slugify(text: str) -> str:
    """Sanitise *text* into a filesystem-safe slug."""
    cleaned = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:80]


def _canonical_url(url: str) -> str:
    """Normalize common video URL variants for cache lookup.

    YouTube ``youtu.be/ID`` and ``watch?v=ID&t=…`` collapse to
    ``https://www.youtube.com/watch?v=ID``.  Other hosts keep query but drop
    the fragment.
    """
    raw = url.strip()
    try:
        parsed = urlparse(raw)
    except ValueError:
        return raw
    host = (parsed.netloc or "").lower().removeprefix("www.")
    path = parsed.path or ""

    video_id: str | None = None
    if host == "youtu.be":
        video_id = path.strip("/").split("/", 1)[0] or None
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if path.startswith("/watch"):
            video_id = (parse_qs(parsed.query).get("v") or [None])[0]
        elif path.startswith("/shorts/") or path.startswith("/embed/"):
            parts = path.strip("/").split("/", 2)
            video_id = parts[1] if len(parts) > 1 else None

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"

    return parsed._replace(fragment="").geturl()


def _cookie_ydl_opts(
    cookies_from_browser: str | None,
    cookies_file: str | None,
) -> dict[str, Any]:
    """Build yt-dlp API cookie options from CLI/env-style strings."""
    opts: dict[str, Any] = {}
    browser = (cookies_from_browser or "").strip()
    if browser:
        opts["cookiesfrombrowser"] = _parse_cookies_from_browser(browser)
    path = (cookies_file or "").strip()
    if path:
        opts["cookiefile"] = path
    return opts


def _parse_cookies_from_browser(spec: str) -> tuple[Any, ...]:
    """Parse ``BROWSER[+KEYRING][:PROFILE][::CONTAINER]`` into a yt-dlp tuple."""
    try:
        from yt_dlp import parse_options

        parsed = parse_options(["--cookies-from-browser", spec]).ydl_opts.get("cookiesfrombrowser")
        if parsed:
            return tuple(parsed)
    except Exception:
        pass
    # Fallback when parse_options is unavailable: browser name only.
    name = re.split(r"[+:]", spec, maxsplit=1)[0].strip().lower()
    return (name, None, None, None)


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
    canon = _canonical_url(url)
    if canon != url:
        mapping[canon] = slug
    path = output_dir / _URL_SLUG_MAP
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def _dump_json(
    url: str,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    """Run ``yt-dlp --dump-json`` and return the parsed dict."""
    cmd = ["yt-dlp", "--dump-json", "--no-playlist"]
    browser = (cookies_from_browser or "").strip()
    if browser:
        cmd.extend(["--cookies-from-browser", browser])
    path = (cookies_file or "").strip()
    if path:
        cmd.extend(["--cookies", path])
    cmd.append(url)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)
