"""Tests for annotate context filtering and system/user prompt split."""

from __future__ import annotations

import json
from pathlib import Path

from light_subtitle.annotate import (
    _filter_annotate_context,
    _filter_glossary,
    _filter_summary,
    _filter_terms,
    _load_domain_context,
    _render_annotate_system_prompt,
    _render_annotate_user_prompt,
)


class TestFilterSummary:
    def test_drops_speakers_keeps_title_domain_overview_topics(self):
        raw = {
            "title": "AI Talk",
            "domain": "Technology",
            "overview": "Discussion about scaling.",
            "key_topics": ["scaling", "RL"],
            "speakers": {"S1": "host"},
        }
        result = _filter_summary(raw)
        assert result == {
            "title": "AI Talk",
            "domain": "Technology",
            "overview": "Discussion about scaling.",
            "key_topics": ["scaling", "RL"],
        }
        assert "speakers" not in result

    def test_returns_none_for_empty(self):
        assert _filter_summary(None) is None
        assert _filter_summary({}) is None


class TestFilterTerms:
    def test_extracts_term_and_context(self):
        domain_context = {
            "domain": "History",
            "topics": ["WWII"],
            "terminology": [
                {"term": "kamikaze", "context": "divine wind", "confidence": "high"},
                {"term": "Onoda", "context": "", "confidence": "medium"},
            ],
        }
        result = _filter_terms(domain_context)
        assert result == [
            {"term": "kamikaze", "context": "divine wind"},
            {"term": "Onoda"},
        ]

    def test_returns_none_when_missing(self):
        assert _filter_terms(None) is None
        assert _filter_terms({"terminology": []}) is None


class TestFilterGlossary:
    def test_skips_keep_as_is(self):
        raw = {"RL": "强化学习", "ASR": "ASR", "GPU": "GPU"}
        result = _filter_glossary(raw)
        assert result == {"RL": "强化学习"}

    def test_returns_none_for_empty(self):
        assert _filter_glossary(None) is None
        assert _filter_glossary({"ASR": "ASR"}) is None


class TestFilterAnnotateContext:
    def test_combined_filter(self):
        summary, terms, glossary = _filter_annotate_context(
            {"title": "T", "domain": "D", "overview": "O", "key_topics": ["a"], "speakers": {}},
            {"terminology": [{"term": "x", "context": "y"}]},
            {"a": "甲", "b": "b"},
        )
        assert summary["title"] == "T"
        assert terms == [{"term": "x", "context": "y"}]
        assert glossary == {"a": "甲"}


class TestLoadDomainContext:
    def test_loads_from_disk(self, tmp_path: Path):
        correct_dir = tmp_path / "transcript_correct"
        correct_dir.mkdir()
        payload = {"domain": "History", "terminology": [{"term": "foo", "context": "bar"}]}
        (correct_dir / "domain_context.json").write_text(json.dumps(payload), encoding="utf-8")

        loaded = _load_domain_context(correct_dir / "domain_context.json")
        assert loaded == payload

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert _load_domain_context(tmp_path / "transcript_correct" / "domain_context.json") is None


class TestAnnotatePromptRendering:
    def test_system_prompt_contains_rules_and_context(self, tmp_path: Path):
        correct_dir = tmp_path / "transcript_correct"
        correct_dir.mkdir()
        (correct_dir / "domain_context.json").write_text(
            json.dumps({"terminology": [{"term": "RL", "context": "reinforcement learning"}]}),
            encoding="utf-8",
        )
        prompt = _render_annotate_system_prompt(
            glossary={"RL": "强化学习", "ASR": "ASR"},
            content_summary={
                "title": "AI Talk",
                "domain": "Technology",
                "overview": "Talk about AI scaling",
                "key_topics": ["scaling"],
            },
            domain_context_path=correct_dir / "domain_context.json",
        )
        assert "Information Density Assessment" in prompt
        assert "AI Talk" in prompt
        assert "Talk about AI scaling" in prompt
        assert "RL (reinforcement learning)" in prompt
        assert "RL→强化学习" in prompt
        assert "ASR" not in prompt

    def test_system_prompt_excludes_batch_input(self, tmp_path: Path):
        prompt = _render_annotate_system_prompt(
            domain_context_path=tmp_path / "transcript_correct" / "domain_context.json"
        )
        assert "## Input Format" in prompt
        assert "## Input\n" not in prompt
        assert "Already Annotated Terms" not in prompt

    def test_user_prompt_contains_batch_and_already_annotated(self):
        batch = '[{"unit_id": "u001", "source": "hello", "translation": "你好"}]'
        prompt = _render_annotate_user_prompt(batch, ["rl训练"])
        assert batch in prompt
        assert "rl训练" in prompt
        assert "Already Annotated Terms" in prompt

    def test_user_prompt_omits_already_annotated_when_empty(self):
        batch = '[{"unit_id": "u001", "source": "hello", "translation": "你好"}]'
        prompt = _render_annotate_user_prompt(batch, None)
        assert batch in prompt
        assert "Already Annotated Terms" not in prompt
