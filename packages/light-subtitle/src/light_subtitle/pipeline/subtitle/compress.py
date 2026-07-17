"""Compress over-CPS cue texts via LLM rewrite (meaning preserved).

Reading-speed violations are a *text* problem: after pace has borrowed
all the time it can, any cue still over the CPS ceiling is rewritten by
the LLM into a shorter form.  Code verifies the character cap before
applying; cues that fail verification keep their original wording and
are left for QC to flag.
"""

from __future__ import annotations

import json
import re

from light_models import SubtitleCue

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient
from ...llm.prompts import render_prompt
from ...usage.tracker import merge_token_usage

_MAX_ATTEMPTS = 2  # initial try + one retry with violations fed back


def compress_over_cps(cues: list[SubtitleCue], indices: list[int], config: SubtitleConfig) -> dict | None:
    """Rewrite cues at *indices* so they fit their CPS ceiling.

    Returns token usage (None when the LLM is unavailable).  Only texts
    that verify against the cap are applied.
    """
    if not indices or not config.llm_api_key:
        return None

    client = OpenAIClient(base_url=config.llm_base_url, api_key=config.llm_api_key, model=config.llm_model)
    items = []
    for n, i in enumerate(indices):
        cue = cues[i]
        limit = config.cps_limit if cue.lang == "zh" else config.cps_limit_en
        max_chars = max(1, int((cue.end - cue.start) * limit))
        source = " ".join(w.text.strip() for w in cue.words if w.text.strip())
        items.append({"id": n, "max_chars": max_chars, "text": cue.text.replace("\n", " "), "source": source})

    total_usage: dict = {}
    feedback = ""
    accepted: dict[int, str] = {}
    for attempt in range(_MAX_ATTEMPTS):
        payload: dict = {"target_lang": config.target_lang, "items": items}
        if feedback:
            payload["previous_error"] = feedback
        messages = [
            {"role": "system", "content": render_prompt("compress_cues_system.j2")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response, usage = client.chat(messages, temperature=config.llm_temperature)
        merge_token_usage(total_usage, usage)
        results = _parse_results(response)
        if results is None:
            feedback = 'Output was not valid JSON of the form {"results": [{"id": 0, "text": "..."}]}.'
            continue
        rejected = []
        for n, item in enumerate(items):
            text = results.get(n)
            if text is None:
                rejected.append(f"missing id {n}")
            elif _count_chars(text) > item["max_chars"]:
                rejected.append(f"id {n} has {_count_chars(text)} chars (cap {item['max_chars']})")
            else:
                accepted[n] = text
        if not rejected:
            break
        feedback = "Rejected: " + "; ".join(rejected)
        logger.warning(f"  Compress attempt {attempt + 1}: {feedback}")

    for n, text in accepted.items():
        cue = cues[indices[n]]
        logger.info(f"  Compress: {cue.unit_id} {len(items[n]['text'])} → {_count_chars(text)} chars")
        cue.text = text

    return total_usage or None


def _count_chars(text: str) -> int:
    """Character count matching pace's CPS metric (newlines excluded)."""
    return len(text.replace("\n", ""))


def _parse_results(response: str) -> dict[int, str] | None:
    match = re.search(r"\{[\s\S]*\}", response)
    try:
        data = json.loads(match.group(0) if match else response)
    except json.JSONDecodeError:
        return None
    raw = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    results: dict[int, str] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            results[item["id"]] = text.strip()
    return results
