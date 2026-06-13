from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from light_backend.database import insert_chunk, insert_video
from light_backend.main import app
from light_backend.routers.files import (
    RangeNotSatisfiable,
    _parse_byte_range,
    _resolve_subtitle_path,
    _safe_existing_path,
)
from light_backend.routers.videos import _chunk_to_out
from light_backend.services.library import _discover_chunks
from light_backend.state import get_config

# ── Range parsing ───────────────────────────────────────


class TestParseByteRange:
    def test_closed_range(self) -> None:
        assert _parse_byte_range("bytes=0-99", 1000) == (0, 99)

    def test_open_ended_range(self) -> None:
        assert _parse_byte_range("bytes=500-", 1000) == (500, 999)

    def test_suffix_range(self) -> None:
        assert _parse_byte_range("bytes=-100", 1000) == (900, 999)

    def test_suffix_longer_than_file(self) -> None:
        assert _parse_byte_range("bytes=-2000", 1000) == (0, 999)

    def test_multipart_rejected(self) -> None:
        with pytest.raises(ValueError, match="multipart"):
            _parse_byte_range("bytes=0-1,2-3", 1000)

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid range"):
            _parse_byte_range("bytes=abc-def", 1000)

    def test_unsupported_unit(self) -> None:
        with pytest.raises(ValueError, match="unsupported range unit"):
            _parse_byte_range("items=0-5", 1000)

    def test_start_beyond_eof(self) -> None:
        with pytest.raises(RangeNotSatisfiable) as exc:
            _parse_byte_range("bytes=1000-", 1000)
        assert exc.value.file_size == 1000


# ── Subtitle path safety ────────────────────────────────


class TestSubtitlePathGuard:
    def test_blocks_traversal_via_lang(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir)
        secret = tmp_path / "secret.vtt"
        secret.write_text("WEBVTT\n", encoding="utf-8")

        chunk = {
            "output_dir": output_dir,
            "video_path": os.path.join(output_dir, "video.mp4"),
            "subtitles": {},
        }
        # lang=../../../secret would join outside output_dir — must be rejected by _SAFE_NAME
        assert _resolve_subtitle_path(chunk, "../../../secret", "vtt") is None

    def test_resolves_pipeline_layout(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        sub = output_dir / "zh.vtt"
        sub.write_text("WEBVTT\n", encoding="utf-8")

        chunk = {
            "output_dir": str(output_dir),
            "video_path": str(output_dir / "chunk_000.mp4"),
            "subtitles": {},
        }
        assert _resolve_subtitle_path(chunk, "zh", "vtt") == str(sub)

    def test_rejects_subtitle_outside_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        outside = tmp_path / "evil.vtt"
        outside.write_text("WEBVTT\n", encoding="utf-8")

        chunk = {
            "output_dir": str(output_dir),
            "video_path": str(output_dir / "video.mp4"),
            "subtitles": {"zh.vtt": str(outside)},
        }
        assert _resolve_subtitle_path(chunk, "zh", "vtt") is None

    def test_safe_existing_path(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        inside = base / "ok.vtt"
        inside.write_text("x", encoding="utf-8")
        outside = tmp_path / "bad.vtt"
        outside.write_text("x", encoding="utf-8")

        assert _safe_existing_path(str(inside), str(base)) == str(inside)
        assert _safe_existing_path(str(outside), str(base)) is None


# ── Chunk discovery ─────────────────────────────────────


class TestDiscoverChunks:
    def test_single_video_with_subtitles(self, tmp_path: Path) -> None:
        (tmp_path / "clip.mp4").write_bytes(b"\x00")
        (tmp_path / "zh.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")

        pairs = _discover_chunks(str(tmp_path))
        assert len(pairs) == 1
        assert pairs[0].video_path.endswith("clip.mp4")
        assert "zh.srt" in pairs[0].subtitles

    def test_multipart_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "video_p1.mp4").write_bytes(b"\x00")
        (tmp_path / "video_p2.mp4").write_bytes(b"\x00")
        (tmp_path / "video_p1.zh.srt").write_text("sub1\n", encoding="utf-8")
        (tmp_path / "video_p2.zh.srt").write_text("sub2\n", encoding="utf-8")
        # Base unsplit video should be ignored when segments exist
        (tmp_path / "video.mp4").write_bytes(b"\x00")

        pairs = _discover_chunks(str(tmp_path))
        assert len(pairs) == 2
        stems = {Path(p.video_path).stem for p in pairs}
        assert stems == {"video_p1", "video_p2"}

    def test_explicit_video_path(self, tmp_path: Path) -> None:
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        out = tmp_path / "subs"
        out.mkdir()
        (out / "zh.vtt").write_text("WEBVTT\n", encoding="utf-8")

        pairs = _discover_chunks(str(out), explicit_video=str(video))
        assert len(pairs) == 1
        assert pairs[0].video_path == str(video)
        assert "zh.vtt" in pairs[0].subtitles


# ── API response sanitization ───────────────────────────


class TestChunkSanitization:
    def test_chunk_to_out_strips_paths(self) -> None:
        out = _chunk_to_out(
            {
                "id": "abc",
                "chunk_index": 0,
                "video_path": "/secret/data/chunk_000.webm",
                "output_dir": "/secret/data/out_000",
                "duration": 120.5,
                "subtitles": {"zh.vtt": "/secret/data/out_000/zh.vtt", "en.srt": "/secret/data/out_000/en.srt"},
            }
        )
        assert "video_path" not in out
        assert "output_dir" not in out
        assert out["video_ext"] == "webm"
        assert set(out["subtitles"]) == {"zh.vtt", "en.srt"}


# ── Stream endpoint integration ─────────────────────────


@pytest.fixture
def playback_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("LIGHT_DATA_DIR", str(data_dir))

    with TestClient(app) as client:
        cfg = get_config()
        db_path = cfg.db_path

        video = insert_video(db_path, title="test", source="import", status="done")
        video_file = data_dir / "sample.mp4"
        video_file.write_bytes(b"\x00" * 2048)

        output_dir = data_dir / "out_000"
        output_dir.mkdir()
        sub = output_dir / "zh.vtt"
        sub.write_text("WEBVTT\n", encoding="utf-8")

        chunk = insert_chunk(
            db_path,
            video_id=video["id"],
            chunk_index=0,
            video_path=str(video_file),
            output_dir=str(output_dir),
            subtitles={"zh.vtt": str(sub)},
        )
        yield client, chunk, video_file, sub


class TestStreamEndpoint:
    def test_full_file_response(self, playback_client) -> None:
        client, chunk, video_file, _ = playback_client
        resp = client.get(f"/api/chunks/{chunk['id']}/stream")
        assert resp.status_code == 200
        assert resp.content == video_file.read_bytes()
        assert resp.headers.get("accept-ranges") == "bytes"

    def test_partial_content(self, playback_client) -> None:
        client, chunk, _, _ = playback_client
        resp = client.get(f"/api/chunks/{chunk['id']}/stream", headers={"Range": "bytes=10-19"})
        assert resp.status_code == 206
        assert len(resp.content) == 10
        assert resp.headers.get("content-range") == "bytes 10-19/2048"

    def test_suffix_range(self, playback_client) -> None:
        client, chunk, video_file, _ = playback_client
        resp = client.get(f"/api/chunks/{chunk['id']}/stream", headers={"Range": "bytes=-8"})
        assert resp.status_code == 206
        assert resp.content == video_file.read_bytes()[-8:]

    def test_unsatisfiable_range_returns_416(self, playback_client) -> None:
        client, chunk, _, _ = playback_client
        resp = client.get(f"/api/chunks/{chunk['id']}/stream", headers={"Range": "bytes=9999-"})
        assert resp.status_code == 416
        assert resp.headers.get("content-range") == "bytes */2048"

    def test_subtitle_served(self, playback_client) -> None:
        client, chunk, _, sub = playback_client
        resp = client.get(f"/api/chunks/{chunk['id']}/subtitles/zh.vtt")
        assert resp.status_code == 200
        assert resp.text == sub.read_text(encoding="utf-8")

    def test_subtitle_traversal_blocked(self, playback_client) -> None:
        client, chunk, _, _ = playback_client
        resp = client.get(f"/api/chunks/{chunk['id']}/subtitles/..%2F..%2Fetc/passwd.vtt")
        assert resp.status_code == 404
