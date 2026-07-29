"""Judge + Refine: LLM reviews plan output and applies deterministic fixes.

The judge reports each issue with a concrete action (merge_right/left,
shift_to_next/prev), and the refine step executes them in right-to-left
order to avoid cascading conflicts.
"""

from __future__ import annotations

import json

from light_core import logger
from light_llm.client import OpenAIClient
from light_llm.json_extract import extract_json_object
from light_llm.usage.tracker import merge_token_usage
from light_models import Segment

from ..config import PlanConfig
from ..prompts import render_prompt
from .lexicon import FUNC_TAIL

_BATCH_SIZE = 60
_MAX_ATTEMPTS = 2
_VALID_ACTIONS = frozenset({"merge_right", "merge_left", "shift_to_next", "shift_to_prev"})
_VALID_TYPES = frozenset({"dangling_word", "flash_unit", "over_fragmentation", "semantic_boundary"})


def refine_plan(
    units: list[Segment], source_segments: list[Segment], config: PlanConfig, client: OpenAIClient | None
) -> tuple[list[Segment], list[dict], dict]:
    """Review planned units and apply automated fixes."""
    if client is None or len(units) < 2:
        return units, [], {}

    batches = [units[i : i + _BATCH_SIZE] for i in range(0, len(units), _BATCH_SIZE)]
    all_issues: list[dict] = []
    total_usage: dict = {}

    for batch in batches:
        issues, usage = _judge_batch(batch, source_segments, config, client)
        if usage:
            merge_token_usage(total_usage, usage)
        if issues:
            logger.info(f"  Refine: batch of {len(batch)} units → {len(issues)} issue(s)")
            for iss in issues:
                a = iss.get("action", "?")
                c = iss.get("count", "")
                logger.info(f"    {iss['unit_id']} {iss['problem_type']} → {a}" + (f" x{c}" if c else ""))
            all_issues.extend(issues)

    if not all_issues:
        return units, [], total_usage

    refined = _apply_refinements(units, all_issues, config)
    n_actions = len(units) - len(refined)
    if n_actions:
        logger.info(f"  Refine: {len(all_issues)} issue(s) → {n_actions} unit(s) reduced")
    return refined, all_issues, total_usage


def _judge_batch(
    batch: list[Segment], source_segments: list[Segment], config: PlanConfig, client: OpenAIClient
) -> tuple[list[dict] | None, dict | None]:
    system = render_prompt(
        "plan_refine_system.j2",
        min_duration=config.min_duration,
        soft_cap=round(config.max_duration * 1.15, 1),
    )
    user = _build_user_data(batch, source_segments)
    usage: dict = {}
    for attempt in range(_MAX_ATTEMPTS):
        response, u = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        merge_token_usage(usage, u)
        issues = _parse_issues(response, batch)
        if issues is not None:
            return issues, usage
        logger.warning(f"  Refine attempt {attempt + 1}: unparseable — {response[:120]!r}")
    return None, usage


def _build_user_data(batch: list[Segment], source_segments: list[Segment]) -> str:
    lo = min(u.start for u in batch)
    hi = max(u.end for u in batch)
    nearby = [s for s in source_segments if s.end >= lo - 5.0 and s.start <= hi + 5.0]
    if not nearby:
        nearby = source_segments[:10]

    source_lines = "\n".join(
        f"{s.unit_id} | {s.start:.2f}-{s.end:.2f} | {s.source_text}" for s in nearby
    )
    unit_lines = "\n".join(
        f"{u.unit_id} | {u.start:.2f}-{u.end:.2f} | {u.source_text}" for u in batch
    )

    user_data = f"""<source_segments>
{source_lines}
</source_segments>

<planned_units>
{unit_lines}
</planned_units>"""
    return user_data


def _parse_issues(response: str, batch: list[Segment]) -> list[dict] | None:
    fragment = extract_json_object(response)
    if fragment is None:
        return None
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    raw = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None

    batch_ids = {u.unit_id for u in batch}
    issues: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        uid = item.get("unit_id")
        if not isinstance(uid, str) or uid not in batch_ids:
            continue
        action = item.get("action", "")
        if action not in _VALID_ACTIONS:
            continue
        pt = item.get("problem_type", "")
        if pt not in _VALID_TYPES:
            continue
        count = item.get("count")
        if action in ("shift_to_next", "shift_to_prev"):
            if not isinstance(count, int) or count < 1:
                continue
        note = str(item.get("note", ""))
        entry: dict = {"unit_id": uid, "problem_type": pt, "action": action, "note": note}
        if count is not None:
            entry["count"] = int(count)
        issues.append(entry)
    return issues or None


def _apply_refinements(
    units: list[Segment], issues: list[dict], config: PlanConfig
) -> list[Segment]:
    """Execute concrete actions proposed by the judge, right-to-left to avoid conflicts."""
    from dataclasses import replace

    soft_cap = config.max_duration * 1.15
    by_id = {u.unit_id: i for i, u in enumerate(units)}
    consumed: set[str] = set()
    mutable = [replace(u) for u in units]  # shallow copy for in-place mutation

    # Sort issues right-to-left
    ordered = sorted(issues, key=lambda iss: by_id.get(iss["unit_id"], 9999), reverse=True)
    for iss in ordered:
        uid = iss["unit_id"]
        action = iss["action"]
        idx = by_id.get(uid)
        if idx is None or uid in consumed:
            continue

        if action == "merge_right":
            if idx + 1 >= len(mutable) or mutable[idx + 1].unit_id in consumed:
                continue
            nxt = mutable[idx + 1]
            combined_dur = nxt.end - mutable[idx].start
            if combined_dur > soft_cap:
                continue
            merged_words = mutable[idx].words + nxt.words
            mutable[idx] = replace(
                mutable[idx],
                end=nxt.end,
                source_text=" ".join(w.text.strip() for w in merged_words),
                words=merged_words,
            )
            consumed.add(nxt.unit_id)
            by_id.pop(nxt.unit_id, None)

        elif action == "merge_left":
            if idx - 1 < 0 or mutable[idx - 1].unit_id in consumed:
                continue
            prev = mutable[idx - 1]
            combined_dur = mutable[idx].end - prev.start
            if combined_dur > soft_cap:
                continue
            merged_words = prev.words + mutable[idx].words
            mutable[idx - 1] = replace(
                prev,
                end=mutable[idx].end,
                source_text=" ".join(w.text.strip() for w in merged_words),
                words=merged_words,
            )
            consumed.add(uid)

        elif action == "shift_to_next":
            count = iss.get("count", 1)
            if idx + 1 >= len(mutable) or len(mutable[idx].words) <= count:
                continue
            tail = mutable[idx].words[-count:]
            new_last = mutable[idx].words[-count - 1] if len(mutable[idx].words) > count else None
            if new_last and _is_illegal_tail(new_last):
                continue
            mutable[idx] = replace(
                mutable[idx],
                words=mutable[idx].words[:-count],
                source_text=" ".join(w.text.strip() for w in mutable[idx].words[:-count]),
                end=new_last.end if new_last else mutable[idx].end,
            )
            mutable[idx + 1] = replace(
                mutable[idx + 1],
                words=tail + mutable[idx + 1].words,
                source_text=" ".join(w.text.strip() for w in (tail + mutable[idx + 1].words)),
                start=tail[0].start,
            )

        elif action == "shift_to_prev":
            count = iss.get("count", 1)
            if idx - 1 < 0 or len(mutable[idx].words) <= count:
                continue
            tail = mutable[idx].words[-count:]
            new_last = mutable[idx].words[-count - 1] if len(mutable[idx].words) > count else None
            if new_last and _is_illegal_tail(new_last):
                continue
            mutable[idx] = replace(
                mutable[idx],
                words=mutable[idx].words[:-count],
                source_text=" ".join(w.text.strip() for w in mutable[idx].words[:-count]),
                start=mutable[idx].words[count].start if len(mutable[idx].words) > count else mutable[idx].start,
            )
            mutable[idx - 1] = replace(
                mutable[idx - 1],
                words=mutable[idx - 1].words + tail,
                source_text=" ".join(w.text.strip() for w in (mutable[idx - 1].words + tail)),
                end=tail[-1].end,
            )

    # Drop consumed units and renumber
    result = [u for u in mutable if u.unit_id not in consumed]
    for j, ru in enumerate(result):
        result[j] = replace(ru, unit_id=f"p{j:04d}")
    return result


def _is_illegal_tail(word) -> bool:
    """Check if a word is a bare function word (not a legal unit tail)."""
    text = word.text.strip()
    if not text:
        return False
    for ch in reversed(text):
        if ch in ".,!?;:—":
            return False
        break
    return text.strip(".,!?;:—\"'()[]“”‘’").lower() in FUNC_TAIL
