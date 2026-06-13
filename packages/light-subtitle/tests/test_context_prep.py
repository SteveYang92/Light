"""Tests for translation context preparation."""

from __future__ import annotations

from light_models import Segment, Word
from light_subtitle.pipeline.context_prep import (
    _parse_context_response,
    build_transcript_text,
    merge_glossary,
)


def _segment(i: int, text: str, speaker: str = "S1") -> Segment:
    w = Word(text=text, start=float(i), end=float(i) + 1.0, confidence=0.9)
    return Segment(
        unit_id=f"u{i:04d}",
        start=w.start,
        end=w.end,
        speaker=speaker,
        source_text=text,
        words=[w],
    )


class TestMergeGlossary:
    def test_user_overrides_auto(self):
        auto = {"scaling": "规模扩展", "RL": "强化学习"}
        user = {"scaling": "缩放"}
        merged = merge_glossary(auto, user)
        assert merged["scaling"] == "缩放"
        assert merged["RL"] == "强化学习"


class TestBuildTranscriptText:
    def test_all_segments_included(self):
        segments = [_segment(i, f"line {i}") for i in range(5)]
        text = build_transcript_text(segments)
        assert text.count("\n") == 4
        for i in range(5):
            assert f"line {i}" in text

    def test_speaker_prefix(self):
        segments = [_segment(0, "hello", speaker="SPEAKER_00")]
        text = build_transcript_text(segments)
        assert text == "[SPEAKER_00] hello"

    def test_long_input_keeps_all_lines(self):
        segments = [_segment(i, f"text {i}") for i in range(200)]
        text = build_transcript_text(segments)
        assert text.count("\n") + 1 == 200

    def test_unknown_speaker_fallback(self):
        segments = [_segment(0, "hello", speaker="")]
        text = build_transcript_text(segments)
        assert text.startswith("[UNKNOWN]")


class TestParseContextResponse:
    def test_parses_summary_and_glossary(self):
        response = """{
            "summary": {
                "title": "AI Talk",
                "domain": "AI",
                "overview": "Discussion about scaling.",
                "key_topics": ["scaling", "research"],
                "speakers": {"SPEAKER_00": "host"}
            },
            "glossary": {"scaling laws": "缩放定律"}
        }"""
        result = _parse_context_response(response)
        assert result.summary is not None
        assert result.summary["overview"] == "Discussion about scaling."
        assert result.glossary["scaling laws"] == "缩放定律"
