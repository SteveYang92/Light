"""Shared typer option aliases and LLM helpers for the standalone commands."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from light_llm.client import OpenAIClient

# ── Common LLM options (mirrors the pipeline flags) ───────────────────────────

LlmBaseUrl = Annotated[str, typer.Option("--llm-base-url", help="LLM API base URL")]
LlmModel = Annotated[str, typer.Option("--llm-model", help="LLM model name")]
LlmApiKey = Annotated[str, typer.Option("--llm-api-key", help="LLM API key (env: DEEPSEEK_API_KEY)")]


def resolve_api_key(api_key: str) -> str:
    """CLI value wins; fall back to ``DEEPSEEK_API_KEY`` like the pipeline does."""
    return api_key or os.environ.get("DEEPSEEK_API_KEY", "")


def build_client(base_url: str, model: str, api_key: str) -> OpenAIClient:
    """Build an ``OpenAIClient`` from the common LLM options."""
    return OpenAIClient(base_url=base_url, api_key=resolve_api_key(api_key), model=model)
