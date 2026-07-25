"""Baseline checks for translate.j2 prompt rules.

Assertions track the rewritten template's rule headings (Rule A/B/C +
self-correction checklist), not the pre-rewrite wording.
"""

from __future__ import annotations

from importlib.resources import files

from light_subtitle.prompts import render_prompt


def test_translate_prompt_layers_isolation_and_coherence():
    prompt = render_prompt("translate.j2", target_lang="zh", glossary={"AI": "人工智能"})
    assert "Information Isolation" in prompt
    assert "Cross-Segment Coherence" in prompt
    assert "Coherence Test" in prompt
    assert "Split Groups" in prompt
    assert "end with a full stop" in prompt
    assert "batch_index" in prompt
    assert "Context Units" in prompt


def test_translate_prompt_natural_chinese_and_appositive_rules():
    prompt = render_prompt("translate.j2", target_lang="zh", glossary={"AI": "人工智能"})
    assert "natural oral Chinese" in prompt
    assert "Appositive Splits" in prompt
    assert "Appositive Verification" in prompt
    assert "Read-Aloud Spoken Style" in prompt
    assert "Anchor the Head Noun" in prompt
    assert "standalone clause" in prompt
    assert "is_continuation" in prompt


def test_translate_prompt_file_exists():
    assert files("light_subtitle.prompts").joinpath("translate.j2").is_file()
