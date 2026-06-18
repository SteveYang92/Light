"""Baseline checks for translate.j2 prompt rules."""

from __future__ import annotations

from pathlib import Path

from light_subtitle.llm.prompts import render_prompt


def test_translate_prompt_layers_isolation_and_coherence():
    prompt = render_prompt("translate.j2", target_lang="zh", glossary={"AI": "人工智能"})
    assert "Per-Unit Isolation" in prompt
    assert "Cross-segment coherence" in prompt
    assert "Coherence check" in prompt
    assert "Split construction check" in prompt
    assert "both parts together" in prompt
    assert "batch_index" in prompt
    assert "prior context" in prompt


def test_translate_prompt_natural_chinese_and_appositive_rules():
    prompt = render_prompt("translate.j2", target_lang="zh", glossary={"AI": "人工智能"})
    assert "Natural Chinese" in prompt
    assert "Appositive splits" in prompt
    assert "Appositive attachment check" in prompt
    assert "Read-aloud naturalness" in prompt
    assert "Anchor the head noun in part 0" in prompt
    assert "standalone clause" in prompt
    assert "Clause fragments" in prompt


def test_translate_prompt_file_exists():
    assert (Path(__file__).resolve().parents[3] / "prompts" / "translate.j2").exists()
