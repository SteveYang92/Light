"""Tests for download.py — slug derivation and mock download."""

from __future__ import annotations

from pathlib import Path

from light_subtitle.download import _slugify, derive_slug_from_path


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
