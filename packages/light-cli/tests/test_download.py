"""Tests for download.py — slug derivation and mock download."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from light_cli.download import _slugify, derive_slug_from_path


class TestSlugify:
    def test_basic_english(self) -> None:
        assert _slugify("Hello World") == "Hello_World"

    def test_cjk(self) -> None:
        assert _slugify("人工智能的未来") == "人工智能的未来"

    def test_special_characters_stripped(self) -> None:
        assert _slugify("What's Next? (2024)") == "Whats_Next_2024"

    def test_multiple_spaces_collapsed(self) -> None:
        assert _slugify("  many   spaces  ") == "many_spaces"

    def test_truncate_long_title(self) -> None:
        long_title = "A" * 100
        result = _slugify(long_title)
        assert len(result) == 80
        assert result == "A" * 80

    def test_colon_replaced(self) -> None:
        assert _slugify("Foo: Bar") == "Foo_Bar"

    def test_leading_trailing_whitespace(self) -> None:
        assert _slugify("  hello  ") == "hello"


class TestDeriveSlugFromPath:
    def test_simple_filename(self) -> None:
        assert derive_slug_from_path(Path("/videos/interview.mp4")) == "interview"

    def test_filename_with_spaces(self) -> None:
        slug = derive_slug_from_path(Path("Joscha Bach podcast.webm"))
        assert slug == "Joscha_Bach_podcast"

    def test_cjk_filename(self) -> None:
        slug = derive_slug_from_path(Path("人工智能对话.mp4"))
        assert slug == "人工智能对话"

    def test_filename_with_special_chars(self) -> None:
        slug = derive_slug_from_path(Path("Best of 2024! (Full).mkv"))
        assert slug == "Best_of_2024_Full"


class TestDownloadProgress:
    def setup_method(self) -> None:
        for d in (Path("/tmp/fake_output"), Path("/tmp/fake_output_err")):
            if d.exists():
                shutil.rmtree(d)

    def test_progress_hook_monotonic_fractions(self) -> None:
        progress_calls: list[float] = []

        from light_cli.download import download_video

        mock_ydl = MagicMock()
        mock_info = {"title": "Test Video"}
        captured_opts: dict = {}

        def fake_make_ydl(opts):
            captured_opts.update(opts)
            return mock_ydl

        with (
            patch("light_cli.download._dump_json", return_value=mock_info),
            patch("light_cli.download._make_ydl", side_effect=fake_make_ydl),
            patch("light_cli.download._save_url_slug"),
        ):
            tmp = Path("/tmp/fake_output")
            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / "Test_Video").mkdir(parents=True, exist_ok=True)

            def mock_download(urls):
                progress_cb = captured_opts["progress_hooks"][0]
                progress_cb({"status": "downloading", "downloaded_bytes": 25, "total_bytes": 100})
                progress_cb({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
                progress_cb({"status": "downloading", "downloaded_bytes": 75, "total_bytes": 100})
                progress_cb({"status": "downloading", "downloaded_bytes": 100, "total_bytes": 100})
                progress_cb({"status": "finished"})
                (tmp / "Test_Video" / "video.mp4").touch()

            mock_ydl.download.side_effect = mock_download

            messages: list[str] = []

            def on_progress(frac: float, message: str) -> None:
                progress_calls.append(frac)
                messages.append(message)

            download_video(
                "https://example.com/video",
                tmp,
                progress=on_progress,
            )

        assert progress_calls == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
        assert messages[0] == "获取视频信息…"
        assert all(m == "" for m in messages[1:-1])  # byte progress: empty message
        assert messages[-1] == "下载完成"

    def test_progress_hook_with_estimate(self) -> None:
        progress_calls: list[float] = []

        from light_cli.download import download_video

        mock_ydl = MagicMock()
        mock_info = {"title": "Test Video"}
        captured_opts: dict = {}

        def fake_make_ydl(opts):
            captured_opts.update(opts)
            return mock_ydl

        with (
            patch("light_cli.download._dump_json", return_value=mock_info),
            patch("light_cli.download._make_ydl", side_effect=fake_make_ydl),
            patch("light_cli.download._save_url_slug"),
        ):
            tmp = Path("/tmp/fake_output")
            (tmp / "Test_Video").mkdir(parents=True, exist_ok=True)

            def mock_download(urls):
                progress_cb = captured_opts["progress_hooks"][0]
                progress_cb({"status": "downloading", "downloaded_bytes": 50, "total_bytes_estimate": 200})
                (tmp / "Test_Video" / "video.mp4").touch()

            mock_ydl.download.side_effect = mock_download

            download_video(
                "https://example.com/video",
                tmp,
                progress=lambda f, m: progress_calls.append(f),
            )

        assert progress_calls == [0.0, 0.25]

    def test_progress_hook_no_total_skips(self) -> None:
        progress_calls: list[tuple[float, str]] = []

        from light_cli.download import download_video

        mock_ydl = MagicMock()
        mock_info = {"title": "Test Video"}
        captured_opts: dict = {}

        def fake_make_ydl(opts):
            captured_opts.update(opts)
            return mock_ydl

        with (
            patch("light_cli.download._dump_json", return_value=mock_info),
            patch("light_cli.download._make_ydl", side_effect=fake_make_ydl),
            patch("light_cli.download._save_url_slug"),
        ):
            tmp = Path("/tmp/fake_output")
            (tmp / "Test_Video").mkdir(parents=True, exist_ok=True)

            def mock_download(urls):
                progress_cb = captured_opts["progress_hooks"][0]
                progress_cb({"status": "downloading", "downloaded_bytes": 50})
                (tmp / "Test_Video" / "video.mp4").touch()

            mock_ydl.download.side_effect = mock_download

            download_video(
                "https://example.com/video",
                tmp,
                progress=lambda f, m: progress_calls.append((f, m)),
            )

        # Probe emits 0.0; download with unknown total does not emit fractions.
        assert progress_calls == [(0.0, "获取视频信息…")]

    def test_download_passes_cookies_to_ydl_and_probe(self) -> None:
        from light_cli.download import download_video

        mock_ydl = MagicMock()
        mock_info = {"title": "Test Video"}
        captured_opts: dict = {}
        dump_kwargs: dict = {}

        def fake_make_ydl(opts):
            captured_opts.update(opts)
            return mock_ydl

        def fake_dump(url, **kwargs):
            dump_kwargs.update(kwargs)
            return mock_info

        with (
            patch("light_cli.download._dump_json", side_effect=fake_dump),
            patch("light_cli.download._make_ydl", side_effect=fake_make_ydl),
            patch("light_cli.download._save_url_slug"),
        ):
            tmp = Path("/tmp/fake_output_cookies")
            if tmp.exists():
                shutil.rmtree(tmp)
            (tmp / "Test_Video").mkdir(parents=True, exist_ok=True)

            def mock_download(urls):
                (tmp / "Test_Video" / "video.mp4").touch()

            mock_ydl.download.side_effect = mock_download

            download_video(
                "https://example.com/video",
                tmp,
                cookies_from_browser="chrome",
                cookies_file="/tmp/cookies.txt",
            )

        assert dump_kwargs["cookies_from_browser"] == "chrome"
        assert dump_kwargs["cookies_file"] == "/tmp/cookies.txt"
        assert captured_opts["cookiesfrombrowser"][0] == "chrome"
        assert captured_opts["cookiefile"] == "/tmp/cookies.txt"

    def test_download_error_wraps(self) -> None:
        from light_cli.download import _DownloadError, download_video

        mock_ydl = MagicMock()
        mock_info = {"title": "Test Video"}

        with (
            patch("light_cli.download._dump_json", return_value=mock_info),
            patch("light_cli.download._make_ydl", return_value=mock_ydl),
            patch("light_cli.download._save_url_slug"),
        ):
            mock_ydl.download.side_effect = _DownloadError("Connection reset")

            try:
                download_video("https://example.com/video", Path("/tmp/fake_output_err"))
            except RuntimeError as e:
                assert "yt-dlp download failed" in str(e)
            else:
                raise AssertionError("Expected RuntimeError")


class TestCookieOpts:
    def test_parse_browser_chrome(self) -> None:
        from light_cli.download import _parse_cookies_from_browser

        tup = _parse_cookies_from_browser("chrome")
        assert tup[0] == "chrome"

    def test_cookie_ydl_opts_empty(self) -> None:
        from light_cli.download import _cookie_ydl_opts

        assert _cookie_ydl_opts(None, None) == {}
        assert _cookie_ydl_opts("", "") == {}
