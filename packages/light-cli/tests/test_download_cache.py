"""Tests for download cache lookup — bare video.* and legacy {slug}.* names."""

from __future__ import annotations

import json
from pathlib import Path

from light_cli.download import (
    _canonical_url,
    find_cached_download,
    find_video_in_dir,
)


def test_find_video_in_dir_prefers_video_star(tmp_path: Path) -> None:
    slug = "Demo"
    work = tmp_path / slug
    work.mkdir()
    (work / f"{slug}.webm").write_bytes(b"legacy")
    (work / "video.webm").write_bytes(b"canonical")
    found = find_video_in_dir(work, slug)
    assert found is not None
    assert found.name == "video.webm"


def test_find_video_in_dir_falls_back_to_slug_name(tmp_path: Path) -> None:
    slug = "Demo"
    work = tmp_path / slug
    work.mkdir()
    (work / f"{slug}.webm").write_bytes(b"legacy")
    found = find_video_in_dir(work, slug)
    assert found is not None
    assert found.name == f"{slug}.webm"


def test_find_cached_download_with_renamed_video(tmp_path: Path) -> None:
    """After legacy rename video.webm → {slug}.webm, cache lookup must still hit."""
    url = "https://www.youtube.com/watch?v=eAXxdtNlK04"
    slug = "Stop_Burning_Tokens"
    work = tmp_path / slug
    work.mkdir()
    (work / f"{slug}.webm").write_bytes(b"x")
    (tmp_path / "url_slug_cache.json").write_text(json.dumps({url: slug}), encoding="utf-8")

    cached = find_cached_download(url, tmp_path)
    assert cached is not None
    path, got_slug = cached
    assert got_slug == slug
    assert path.name == f"{slug}.webm"


def test_find_cached_download_canonicalizes_youtube_url(tmp_path: Path) -> None:
    slug = "Demo"
    work = tmp_path / slug
    work.mkdir()
    (work / "video.webm").write_bytes(b"x")
    (tmp_path / "url_slug_cache.json").write_text(
        json.dumps({"https://www.youtube.com/watch?v=abc123": slug}),
        encoding="utf-8",
    )

    cached = find_cached_download("https://youtu.be/abc123", tmp_path)
    assert cached is not None
    assert cached[1] == slug


def test_canonical_url_strips_youtube_timestamp() -> None:
    assert _canonical_url("https://www.youtube.com/watch?v=abc&t=30s") == ("https://www.youtube.com/watch?v=abc")
    assert _canonical_url("https://youtu.be/abc") == "https://www.youtube.com/watch?v=abc"
