"""Shared JSON fragment extraction for LLM responses.

LLM replies frequently wrap JSON in prose or markdown fences.  These
helpers centralize the two extraction regexes used across the pipeline
(first ``[...]`` array span / first ``{...}`` object span, both greedy
DOTALL matches) plus the canonical array-response fallback chain shared
by punct restore and transcript correction.
"""

from __future__ import annotations

import json
import re


def extract_json_array(text: str) -> str | None:
    """Return the first ``[...]`` span (greedy, DOTALL) in *text*, else None."""
    match = re.search(r"\[[\s\S]*\]", text)
    return match.group(0) if match else None


def extract_json_object(text: str) -> str | None:
    """Return the first ``{...}`` span (greedy, DOTALL) in *text*, else None."""
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def parse_json_array_response(response: str) -> list[dict]:
    """Parse an LLM response expected to contain a JSON array of objects.

    Tries the extracted ``[...]`` span first, then the whole response;
    returns ``[]`` when neither parses.
    """
    json_fragment = extract_json_array(response)
    if json_fragment is not None:
        try:
            return json.loads(json_fragment)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return []
