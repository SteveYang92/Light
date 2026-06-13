"""Thin wrapper around the ``openai`` library for chat-completion calls.

All pipeline steps instantiate ``OpenAIClient`` with the same constructor
signature and call ``chat()`` with the same return type, so switching from
raw httpx to the official SDK required no caller changes.
"""

from __future__ import annotations

from openai import OpenAI, Timeout

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

        ``usage_dict`` contains: ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        if usage:
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        else:
            usage_dict = {}
        return content, usage_dict


# ── Token accounting helpers (unchanged API) ─────────────


def format_token_usage(usage: dict | None) -> str:
    """Format token usage for log output."""
    if not usage:
        return "tokens: ?"
    return (
        f"tokens: {usage.get('total_tokens', '?')} "
        f"(prompt: {usage.get('prompt_tokens', '?')}, "
        f"completion: {usage.get('completion_tokens', '?')})"
    )


def merge_token_usage(total: dict[str, int], usage: dict | None) -> None:
    """Accumulate token counts from a single LLM call into *total*."""
    if not usage:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + usage.get(key, 0)
