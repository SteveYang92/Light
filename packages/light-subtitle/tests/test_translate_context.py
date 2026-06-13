"""Tests for glossary + content summary injection into translation."""

from __future__ import annotations

from light_models import Segment, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.translate import (
    _build_payload,
    _render_translate_prompt,
    _translation_context_fields,
)


def _segment(unit_id: str, text: str) -> Segment:
    w = Word(text=text, start=0.0, end=2.0, confidence=0.9)
    return Segment(
        unit_id=unit_id,
        start=0.0,
        end=2.0,
        speaker="S1",
        source_text=text,
        words=[w],
    )


class TestTranslationContextInjection:
    def test_payload_includes_glossary_and_summary(self):
        config = SubtitleConfig(
            input_path="test.wav",
            target_lang="zh",
            glossary={"RL": "强化学习"},
            content_summary={"overview": "AI research talk", "key_topics": ["scaling"]},
        )
        segments = [_segment("u0001", "RL training is hard")]
        payload = _build_payload(segments, segments, 0, config)
        assert payload["glossary"] == {"RL": "强化学习"}
        assert payload["content_summary"]["overview"] == "AI research talk"

    def test_context_fields_omit_summary_when_none(self):
        config = SubtitleConfig(input_path="test.wav", target_lang="zh")
        fields = _translation_context_fields(config)
        assert "content_summary" not in fields
        assert fields["glossary"] == {}

    def test_prompt_includes_glossary_and_summary(self):
        config = SubtitleConfig(
            input_path="test.wav",
            target_lang="zh",
            glossary={"scaling": "缩放"},
            content_summary={"overview": "Talk about AI scaling"},
        )
        prompt = _render_translate_prompt(config)
        assert "MANDATORY" in prompt
        assert "scaling" in prompt
        assert "Content Summary" in prompt
        assert "AI scaling" in prompt
