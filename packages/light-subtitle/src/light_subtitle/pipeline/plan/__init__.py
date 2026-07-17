"""Cue planning — one LLM pass decides every subtitle cue boundary.

Replaces the old compose/split/merge-review/display-merge chain: the
planner sees the whole timed word stream together with all display
budgets (duration, reading speed, two-line capacity) and emits cue
boundaries directly.  Code only performs hard validation — coverage,
order, speaker breaks, duration cap — and never scores boundaries or
vetoes them with word lists.  If the LLM fails validation twice (or is
unavailable), a small deterministic fallback guarantees *a* valid plan.

Boundary decisions reference the global word array by index, so word
timing reaches every cue by construction and the EN/ZH tracks share one
unit graph.

Artifacts (``plan/`` output directory):
- ``plan.json``: planned units with global word-index spans (resume).
- ``segment_words.json``: per-unit word timing (written by the caller;
  used by pace word re-attachment and hydrate).

Usage::

    from .plan import run as plan_run
    units, usage = plan_run(segments, config, plan_dir)
"""

from __future__ import annotations

import json
from pathlib import Path

from light_models import Segment, Word

from ... import logger
from ...config import SubtitleConfig
from ...usage.tracker import merge_token_usage
from . import fallback, planner

_PLAN_VERSION = 1


# ── Public API ────────────────────────────────────────────


def run(segments: list[Segment], config: SubtitleConfig, plan_dir: str | Path) -> tuple[list[Segment], dict | None]:
    """Plan cue units from pause-based segments; persist ``plan/plan.json``.

    Returns ``(units, usage)``.  Unit ids are ``pNNNN``; a unit that had
    to be split at the word level becomes ``pNNNN_0``, ``pNNNN_1``, …
    (the split-group protocol consumed by the translate payload builder).
    """
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    if not segments:
        _save_plan(plan_dir, [])
        return [], None

    words = [w for seg in segments for w in seg.words]
    total_usage: dict = {}

    groups, usage = planner.plan_groups(segments, config)
    if usage:
        merge_token_usage(total_usage, usage)
    if groups is None:
        logger.warning("  Plan: LLM planning failed/unavailable — using deterministic fallback")
        groups = fallback.merge_fragments(segments)
    logger.info(f"  Plan: {len(segments)} segments → {len(groups)} cue groups")

    units, meta = _materialize(segments, words, groups, config, total_usage)
    _save_plan(plan_dir, meta)
    return units, total_usage or None


def load_plan_units(plan_dir: str | Path) -> list[Segment] | None:
    """Rebuild planned units from ``plan/plan.json``; None when absent."""
    path = Path(plan_dir) / "plan.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        Segment(
            unit_id=item["unit_id"],
            start=item.get("start", 0.0),
            end=item.get("end", 0.0),
            speaker=item.get("speaker", ""),
            source_text=item.get("text", ""),
            words=[],
        )
        for item in data.get("units", [])
    ]


# ── Unit construction ─────────────────────────────────────


def _materialize(
    segments: list[Segment],
    words: list[Word],
    groups: list[list[int]],
    config: SubtitleConfig,
    total_usage: dict,
) -> tuple[list[Segment], list[dict]]:
    """Turn segment-index groups into timed Segments (splitting overlong ones)."""
    offsets: list[tuple[int, int]] = []
    offset = 0
    for seg in segments:
        offsets.append((offset, offset + len(seg.words)))
        offset += len(seg.words)

    units: list[Segment] = []
    meta: list[dict] = []
    for n, group in enumerate(groups):
        first, last = group[0], group[-1]
        w_start, w_end = offsets[first][0], offsets[last][1]
        span_words = words[w_start:w_end]
        span_text = " ".join(t for t in (segments[i].source_text.strip() for i in group) if t)
        start, end = segments[first].start, segments[last].end
        speaker = segments[first].speaker or ""

        ranges = [(0, len(span_words))]
        if end - start > config.max_duration and len(span_words) > 1:
            split_ranges, usage = planner.split_span(span_words, config)
            if usage:
                merge_token_usage(total_usage, usage)
            if split_ranges is None:
                split_ranges = fallback.split_at_gaps(span_words, config.max_duration)
            ranges = split_ranges
            logger.info(f"  Plan: overlong cue {end - start:.1f}s → {len(ranges)} word-level parts")

        base = f"p{n:04d}"
        for k, (rs, re_) in enumerate(ranges):
            part_words = span_words[rs:re_]
            if not part_words:
                continue
            if len(ranges) == 1:
                unit_id, text, u_start, u_end = base, span_text, start, end
            else:
                unit_id = f"{base}_{k}"
                text = _text_from_words(part_words)
                u_start = part_words[0].start
                u_end = end if k == len(ranges) - 1 else part_words[-1].end
            units.append(
                Segment(unit_id=unit_id, start=u_start, end=u_end, speaker=speaker, source_text=text, words=part_words)
            )
            meta.append(
                {
                    "unit_id": unit_id,
                    "start": round(u_start, 3),
                    "end": round(u_end, 3),
                    "speaker": speaker,
                    "text": text,
                    "word_start": w_start + rs,
                    "word_end": w_start + re_,
                }
            )
    return units, meta


def _text_from_words(words: list[Word]) -> str:
    """Rebuild display text from word tokens (punctuation rides on words)."""
    return " ".join(clean for w in words if (clean := w.text.strip()))


def _save_plan(plan_dir: Path, meta: list[dict]) -> None:
    payload = {"version": _PLAN_VERSION, "units": meta}
    (plan_dir / "plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
