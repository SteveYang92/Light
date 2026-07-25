"""Adapter building :class:`OpenAIClient` from the pipeline config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from light_llm.client import OpenAIClient

if TYPE_CHECKING:
    from .config import SubtitleConfig


def client_from_config(config: SubtitleConfig) -> OpenAIClient:
    """Build an ``OpenAIClient`` from the pipeline config's LLM fields."""
    return OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )
