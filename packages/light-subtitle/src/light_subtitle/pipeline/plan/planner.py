"""LLM boundary planning — one global pass plus focused word-level splits.

Both passes follow the same contract: the LLM decides, code validates
only hard invariants (JSON shape, coverage, order, duration cap,
speaker purity, no dangling function-word tails).  Validation failures
are fed back for one retry; persistent failure returns ``None`` so the
caller can fall back to the deterministic insurance plan.
"""

from __future__ import annotations

import json
import re

from light_models import Segment, Word

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient
from ...llm.prompts import render_prompt
from ...usage.tracker import merge_token_usage
from .boundary import dangling_tail as _dangling_tail

_MAX_ATTEMPTS = 2  # initial try + one retry with validation feedback
_SOFT_MAX_RATIO = 1.15  # tolerance on the duration cap when accepting splits
_MIN_PART_WORDS = 3  # split parts smaller than this read as stubs (QC TinyCue)
_MIN_PART_DURATION = 1.0  # seconds; shorter parts are unreadable flash cues

_DURATION_PREFIX = "DURATION:"


# ── Global pass: segment-level grouping ───────────────────


def plan_groups(segments: list[Segment], config: SubtitleConfig) -> tuple[list[list[int]] | None, dict | None]:
    """Plan cue boundaries as groups of consecutive segment indices.

    Returns ``(groups, usage)``; ``groups`` is None when no API key is
    configured or the LLM output failed hard validation twice.  Duration
    warnings are soft — they are fed back but never cause plan rejection
    (overlong groups are handled by word-level splits downstream).
    """
    if not config.llm_api_key or not segments:
        return None, None
    client = OpenAIClient(base_url=config.llm_base_url, api_key=config.llm_api_key, model=config.llm_model)
    system = render_prompt("plan_system.j2", max_duration=config.max_duration, min_duration=config.min_duration)
    payload = {
        "segments": [
            {
                "id": i,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "dur": round(s.end - s.start, 1),
                "speaker": s.speaker or "",
                "text": s.source_text.strip(),
            }
            for i, s in enumerate(segments)
        ]
    }
    total_usage: dict = {}
    feedback = ""
    for attempt in range(_MAX_ATTEMPTS):
        user = dict(payload)
        if feedback:
            user["previous_error"] = feedback
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        response, usage = client.chat(messages, temperature=config.llm_temperature)
        merge_token_usage(total_usage, usage)
        groups = _parse_groups(response)
        if groups is None:
            feedback = 'Output was not valid JSON of the form {"cues": [[0, 1], [2], ...]}.'
            logger.warning(f"  Plan attempt {attempt + 1}: unparseable output")
            continue
        problems = _group_problems(groups, segments, config.max_duration)
        hard = [p for p in problems if not p.startswith(_DURATION_PREFIX)]
        if not hard:
            dur_count = len(problems) - len(hard)
            if dur_count:
                logger.warning(
                    f"  Plan: {dur_count} overlong cue group(s) accepted (word-level splits will handle them)"
                )
            return groups, total_usage or None
        feedback = "Invalid plan: " + "; ".join(problems)
        logger.warning(f"  Plan attempt {attempt + 1} invalid: {feedback}")
    return None, total_usage or None


def _parse_groups(response: str) -> list[list[int]] | None:
    """Extract ``{"cues": [[int, ...], ...]}`` from an LLM response."""
    match = re.search(r"\{[\s\S]*\}", response)
    try:
        data = json.loads(match.group(0) if match else response)
    except json.JSONDecodeError:
        return None
    cues = data.get("cues") if isinstance(data, dict) else None
    if not isinstance(cues, list):
        return None
    groups: list[list[int]] = []
    for cue in cues:
        if not isinstance(cue, list) or not cue or not all(isinstance(i, int) and not isinstance(i, bool) for i in cue):
            return None
        groups.append(list(cue))
    return groups or None


def _group_problems(groups: list[list[int]], segments: list[Segment], max_duration: float | None = None) -> list[str]:
    """Hard checks: ordered full coverage, no speaker mixing.  Soft duration
    warnings (prefixed with ``DURATION:``) are included for LLM feedback but
    never cause plan rejection."""
    flat = [i for g in groups for i in g]
    if flat != list(range(len(segments))):
        return [f"cues must cover every segment id 0..{len(segments) - 1} exactly once, in ascending order"]
    problems: list[str] = []
    for g in groups:
        speakers = {segments[i].speaker for i in g if segments[i].speaker}
        if len(speakers) > 1:
            problems.append(f"cue {g} mixes speakers {sorted(speakers)}; split it at the speaker change")
    if max_duration is not None:
        for g in groups:
            if len(g) < 2:
                continue
            dur = segments[g[-1]].end - segments[g[0]].start
            if dur > max_duration:
                problems.append(
                    f"{_DURATION_PREFIX}cue {g} ({dur:.1f}s) exceeds the {max_duration}s cap; "
                    "split at a fragment boundary"
                )
    return problems


# ── Focused pass: word-level split of one overlong cue ────


def split_span(
    words: list[Word], config: SubtitleConfig, client: OpenAIClient | None = None
) -> tuple[list[tuple[int, int]] | None, dict | None]:
    """Split one overlong cue's word span into [start, end) word ranges.

    Returns ``(ranges, usage)``; ``ranges`` is None on failure (caller
    falls back to silence-gap splitting).  Pass *client* to share a
    single connection pool across concurrent split calls.
    """
    if not config.llm_api_key or len(words) < 2:
        return None, None
    if client is None:
        client = OpenAIClient(base_url=config.llm_base_url, api_key=config.llm_api_key, model=config.llm_model)
    system = render_prompt(
        "plan_split_system.j2",
        max_duration=config.max_duration,
        soft_max_duration=round(config.max_duration * _SOFT_MAX_RATIO, 2),
    )
    payload = {
        "max_duration": config.max_duration,
        "words": [
            {"i": i, "start": round(w.start, 2), "end": round(w.end, 2), "text": w.text.strip()}
            for i, w in enumerate(words)
        ],
    }
    total_usage: dict = {}
    feedback = ""
    for attempt in range(_MAX_ATTEMPTS):
        user = dict(payload)
        if feedback:
            user["previous_error"] = feedback
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        response, usage = client.chat(messages, temperature=config.llm_temperature)
        merge_token_usage(total_usage, usage)
        breaks = _parse_breaks(response)
        if breaks is None:
            feedback = 'Output was not valid JSON of the form {"breaks": [{"after": 23}, ...]}.'
            logger.warning(f"  Split attempt {attempt + 1}: unparseable output")
            continue
        problems = _break_problems(breaks, words, config.max_duration)
        if not problems:
            return _breaks_to_ranges(breaks, len(words)), total_usage or None
        feedback = "Invalid split: " + "; ".join(problems)
        logger.warning(f"  Split attempt {attempt + 1} invalid: {feedback}")
    return None, total_usage or None


def _parse_breaks(response: str) -> list[int] | None:
    """Extract break word indices from ``{"breaks": [{"after": int}, ...]}``."""
    match = re.search(r"\{[\s\S]*\}", response)
    try:
        data = json.loads(match.group(0) if match else response)
    except json.JSONDecodeError:
        return None
    raw = data.get("breaks") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    breaks: list[int] = []
    for item in raw:
        after = item.get("after") if isinstance(item, dict) else (item if isinstance(item, int) else None)
        if not isinstance(after, int) or isinstance(after, bool):
            return None
        breaks.append(after)
    return breaks


def _break_problems(breaks: list[int], words: list[Word], max_duration: float) -> list[str]:
    """Hard checks: strictly increasing in-range breaks; each part fits
    duration, is big enough to stand on screen, and does not end on a
    dangling function word (the contract stated in the split prompt)."""
    n = len(words)
    if not breaks:
        return ["no break points returned for an overlong cue"]
    if any(b < 0 or b > n - 2 for b in breaks):
        return [f"break indices must be within [0, {n - 2}]"]
    if breaks != sorted(set(breaks)):
        return ["break indices must be strictly increasing and unique"]
    problems = []
    ranges = _breaks_to_ranges(breaks, n)
    if n >= 2 * _MIN_PART_WORDS:
        for s, e in ranges:
            if e - s < _MIN_PART_WORDS:
                problems.append(f"part words[{s}:{e}] has only {e - s} words (minimum {_MIN_PART_WORDS})")
    for s, e in ranges:
        dur = words[e - 1].end - words[s].start
        cap = max_duration * _SOFT_MAX_RATIO
        if dur > cap:
            problems.append(f"part words[{s}:{e}] is {dur:.2f}s, over the {cap:.2f}s cap")
        elif dur < _MIN_PART_DURATION and len(ranges) > 1:
            problems.append(f"part words[{s}:{e}] is {dur:.2f}s, under the {_MIN_PART_DURATION}s minimum")
    for b in breaks:
        bad = _dangling_tail(words[b])
        if bad is not None:
            problems.append(
                f'break {{"after": {b}}} cuts right after "{words[b].text.strip()}", stranding it from what it '
                f"attaches to; move the break before {bad!r} or past the phrase it introduces"
            )
    return problems


def _breaks_to_ranges(breaks: list[int], n_words: int) -> list[tuple[int, int]]:
    points = [b + 1 for b in breaks]
    bounds = [0, *points, n_words]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
