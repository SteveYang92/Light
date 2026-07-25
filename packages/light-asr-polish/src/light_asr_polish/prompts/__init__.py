"""Bundled prompt templates for transcript correction and punctuation restoration."""

from __future__ import annotations

from importlib.resources import files

from light_llm.prompts import render


def render_prompt(name: str, **kwargs) -> str:
    """Render a bundled ``.j2`` template with light_llm's jinja semantics."""
    template = files(__package__).joinpath(name).read_text(encoding="utf-8")
    return render(template, **kwargs)
