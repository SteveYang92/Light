"""LLM boundary planning — one call per shard, word-level with cum-time hints."""

from __future__ import annotations

import json

from light_core import logger
from light_llm.client import OpenAIClient
from light_llm.json_extract import extract_json_object
from light_llm.usage.tracker import merge_token_usage

from ..config import PlanConfig
from ..prompts import render_prompt
from .lexicon import FUNC_TAIL

_MAX_ATTEMPTS = 3


def plan_shard(
    shard_words: list, cum_times: list[float], markers: set[int], config: PlanConfig, client: OpenAIClient
) -> tuple[list[int] | None, dict]:
    """Ask the LLM to place breaks in a shard; returns break indices and usage."""
    system = render_prompt(
        "plan_system.j2",
        min_duration=config.min_duration,
        max_duration=config.max_duration,
    )
    user_text = _render_word_list(shard_words, cum_times, markers)

    total_usage: dict = {}
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]

    for attempt in range(_MAX_ATTEMPTS):
        response, usage = client.chat(messages, temperature=config.llm_temperature, thinking=config.llm_thinking)
        merge_token_usage(total_usage, usage)
        breaks = _parse_breaks(response, len(shard_words))
        if breaks is not None:
            return breaks, total_usage
        logger.warning(f"  Plan attempt {attempt + 1}: unparseable output — retrying")
        if attempt < _MAX_ATTEMPTS - 1:
            messages.append({"role": "user", "content": 'Output must be {"breaks": [int, ...]}. Try again.'})

    return None, total_usage


def _render_word_list(words: list, cum_times: list[float], markers: set[int]) -> str:
    lines = []
    for i, w in enumerate(words):
        text = w.text.strip()
        suffix = ""
        if i in markers:
            suffix = f"  | cum:{cum_times[i]:.1f}"
        lines.append(f"{i:>4d}  {text}{suffix}")
    return "\n".join(lines)


def _parse_breaks(response: str, n_words: int) -> list[int] | None:
    fragment = extract_json_object(response)
    if fragment is None:
        return None
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    raw = data.get("breaks") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(b, int) and not isinstance(b, bool) for b in raw):
        return None
    breaks = [int(b) for b in raw]
    if breaks != sorted(set(breaks)):
        return None
    if breaks[-1] != n_words - 1:
        return None
    for b in breaks:
        if b < 0 or b >= n_words:
            return None
    return breaks


def fix_illegal_tails(breaks: list[int], words: list) -> list[int]:
    """Slide breaks rightward until every left tail is legal."""
    for idx in range(len(breaks)):
        b = breaks[idx]
        while b >= 0 and b < len(words) - 1:
            w = words[b]
            text = w.text.strip()
            if not text:
                break
            if any(text.endswith(p) for p in (".", ",", "!", "?", ";", ":", "—", "…")):
                break
            core = text.strip(".,!?;:—\"'()[]“”‘’").lower()
            if core not in FUNC_TAIL:
                break
            b += 1
        breaks[idx] = min(b, len(words) - 1)
    return sorted(set(breaks))


def compute_duration(cum_times: list[float], prev_end: int, curr_end: int) -> float:
    """Unit duration from cum times from word prev_end+1 to curr_end."""
    start_cum = cum_times[prev_end] if prev_end >= 0 else 0.0
    return cum_times[curr_end] - start_cum
