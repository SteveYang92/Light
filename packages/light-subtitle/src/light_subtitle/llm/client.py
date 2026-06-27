"""Thin wrapper around the ``openai`` library for chat-completion calls.

All pipeline steps instantiate ``OpenAIClient`` with the same constructor
signature and call ``chat()`` with the same return type, so switching from
raw httpx to the official SDK required no caller changes.
"""

from __future__ import annotations

from openai import OpenAI, Timeout

from ..usage.tracker import format_token_usage, merge_token_usage, parse_api_usage

# ── Client defaults ─────────────────────────────────────

_DEFAULT_TIMEOUT = 300  # seconds — long enough for translation batches
_DEFAULT_MAX_RETRIES = 3


class OpenAIClient:
    """OpenAI-compatible chat completion client (DeepSeek, etc.)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=Timeout(_DEFAULT_TIMEOUT),
            max_retries=_DEFAULT_MAX_RETRIES,
        )
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.3) -> tuple[str, dict]:
        """Return ``(content, usage_dict)``.

        ``usage_dict`` includes standard token counts plus API-provided cache
        buckets, reasoning tokens, and direct cost fields when available.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content or ""
        return content, parse_api_usage(response.usage)


# Re-export helpers for backward compatibility.
__all__ = ["OpenAIClient", "format_token_usage", "merge_token_usage"]
