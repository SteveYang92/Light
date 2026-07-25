"""Cue planning — one LLM pass decides every subtitle cue boundary.

Replaces the old compose/split/merge-review/display-merge chain: the
planner sees the whole timed word stream together with all display
budgets (duration, reading speed, two-line capacity) and emits cue
boundaries directly.  Code performs hard validation only — coverage,
order, speaker breaks, duration cap, plus the dangling-tail contract
the split prompt itself declares (no cut right after a function word).
It never scores boundary quality beyond that contract.  If the LLM
fails validation twice (or is unavailable), a small deterministic
fallback guarantees *a* valid plan.

Boundary decisions reference the global word array by index, so word
timing reaches every cue by construction and the EN/ZH tracks share one
unit graph.

Artifacts (``plan/`` output directory):
- ``plan.json``: planned units with global word-index spans (resume).
- ``segment_words.json``: per-unit word timing (written by the caller;
  used by pace word re-attachment and hydrate).

Usage::

    from light_subtitle import plan as plan_pipeline
    units, usage = plan_pipeline.run(segments, config, plan_dir, llm=client)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from light_core import logger
from light_core.progress import ProgressCallback
from light_llm.client import OpenAIClient
from light_llm.parallel import run_parallel_with_warmup
from light_llm.usage.tracker import merge_token_usage
from light_models import Segment, Word
from light_text import SENTENCE_ENDS

from .. import artifacts
from ..config import PlanConfig
from . import fallback, planner

_PLAN_VERSION = 1
_MAX_WORKERS = 4


# ── Public API ────────────────────────────────────────────


def run(
    segments: list[Segment],
    config: PlanConfig,
    plan_dir: str | Path,
    *,
    llm: OpenAIClient | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[Segment], dict | None]:
    """Plan cue units from pause-based segments; persist ``plan/plan.json``.

    Returns ``(units, usage)``.  Unit ids are ``pNNNN``; a unit that had
    to be split at the word level becomes ``pNNNN_0``, ``pNNNN_1``, …
    (the split-group protocol consumed by the translate payload builder).
    *progress*, when given, receives ``(fraction, message)`` per split span.
    ``llm=None`` forces the deterministic fallback (no LLM calls).
    """
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    if not segments:
        _save_plan(plan_dir, [])
        return [], None

    words = [w for seg in segments for w in seg.words]
    total_usage: dict = {}

    groups, usage = planner.plan_groups(segments, config, llm)
    if usage:
        merge_token_usage(total_usage, usage)
    if groups is None:
        logger.warning("  Plan: LLM planning failed/unavailable — using deterministic fallback")
        groups = fallback.merge_fragments(segments)
    logger.info(f"  Plan: {len(segments)} segments → {len(groups)} cue groups")

    units, meta = _materialize(segments, words, groups, config, total_usage, llm=llm, progress=progress)
    _save_plan(plan_dir, meta)
    return units, total_usage or None


def load_plan_units(plan_dir: str | Path) -> list[Segment] | None:
    """Rebuild planned units from ``plan/plan.json``; None when absent."""
    path = Path(plan_dir) / artifacts.PLAN_JSON
    if not path.exists():
        return None
    return artifacts.read_plan_units(path)


# ── Unit materialization ──────────────────────────────────


@dataclass
class _GroupInfo:
    n: int
    segment_indices: list[int]
    w_start: int
    w_end: int
    span_words: list[Word]
    span_text: str
    start: float
    end: float
    speaker: str
    base: str
    overlong: bool


def _materialize(
    segments: list[Segment],
    words: list[Word],
    groups: list[list[int]],
    config: PlanConfig,
    total_usage: dict,
    *,
    llm: OpenAIClient | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[Segment], list[dict]]:
    """Turn segment-index groups into timed Segments (splitting overlong ones).

    Overlong splits run concurrently via ``run_parallel_with_warmup``,
    sharing one ``OpenAIClient`` connection pool (``_MAX_WORKERS``=4).
    """
    offsets: list[tuple[int, int]] = []
    offset = 0
    for seg in segments:
        offsets.append((offset, offset + len(seg.words)))
        offset += len(seg.words)

    # ── Phase 1: collect group metadata ──
    infos: list[_GroupInfo] = []
    for n, group in enumerate(groups):
        first, last = group[0], group[-1]
        w_start, w_end = offsets[first][0], offsets[last][1]
        span_words = words[w_start:w_end]
        span_text = " ".join(t for t in (segments[i].source_text.strip() for i in group) if t)
        start, end = segments[first].start, segments[last].end
        speaker = segments[first].speaker or ""
        overlong = (end - start) > config.max_duration and len(span_words) > 1
        infos.append(
            _GroupInfo(
                n=n,
                segment_indices=group,
                w_start=w_start,
                w_end=w_end,
                span_words=span_words,
                span_text=span_text,
                start=start,
                end=end,
                speaker=speaker,
                base=f"p{n:04d}",
                overlong=overlong,
            )
        )

    # ── Phase 2: concurrent split for overlong groups ──
    split_results: dict[int, list[tuple[int, int]]] = {}
    overlong_infos = [info for info in infos if info.overlong]
    if overlong_infos and llm is not None:
        shared_client = llm
        split_tasks: list[tuple[Callable[[], tuple[list[tuple[int, int]] | None, dict | None]], int]] = []
        for info in overlong_infos:
            # Capture by-value via default args to avoid closure late-binding
            def _make_split_task(
                w: list[Word] = info.span_words, cfg: PlanConfig = config
            ) -> tuple[list[tuple[int, int]] | None, dict | None]:
                return planner.split_span(w, cfg, client=shared_client)

            split_tasks.append((_make_split_task, info.n))

        n_spans = len(overlong_infos)
        done_spans = 0

        def _on_span_done() -> None:
            nonlocal done_spans
            done_spans += 1
            if progress is not None:
                progress(done_spans / n_spans, f"规划字幕边界中... {done_spans}/{n_spans}")

        raw_results = run_parallel_with_warmup(split_tasks, max_workers=_MAX_WORKERS, on_complete=_on_span_done)
        for idx, (ranges, usage) in raw_results.items():
            if usage:
                merge_token_usage(total_usage, usage)
            if ranges is None:
                ranges = fallback.split_at_gaps(infos[idx].span_words, config.max_duration)
            split_results[idx] = ranges
    else:
        for info in overlong_infos:
            split_results[info.n] = fallback.split_at_gaps(info.span_words, config.max_duration)

    # ── Phase 3: materialize units, preserving global order ──
    units: list[Segment] = []
    meta: list[dict] = []
    for info in infos:
        if info.overlong:
            ranges = split_results[info.n]
            logger.info(f"  Plan: overlong cue {info.end - info.start:.1f}s → {len(ranges)} word-level parts")
        else:
            ranges = [(0, len(info.span_words))]

        for k, (rs, re_) in enumerate(ranges):
            part_words = info.span_words[rs:re_]
            if not part_words:
                continue
            if len(ranges) == 1:
                unit_id, text, u_start, u_end = info.base, info.span_text, info.start, info.end
            else:
                unit_id = f"{info.base}_{k}"
                text = _text_from_words(part_words)
                u_start = part_words[0].start
                u_end = info.end if k == len(ranges) - 1 else part_words[-1].end
            units.append(
                Segment(
                    unit_id=unit_id,
                    start=u_start,
                    end=u_end,
                    speaker=info.speaker,
                    source_text=text,
                    words=part_words,
                )
            )
            meta.append(
                {
                    "unit_id": unit_id,
                    "start": round(u_start, 3),
                    "end": round(u_end, 3),
                    "speaker": info.speaker,
                    "text": text,
                    "word_start": info.w_start + rs,
                    "word_end": info.w_start + re_,
                }
            )
    _fold_flash_units(units, meta, config.min_duration)
    _return_leading_punctuation(units, meta)
    return units, meta


def _fold_flash_units(units: list[Segment], meta: list[dict], min_duration: float) -> None:
    """Fold sub-``min_duration`` units into a neighbour.

    The LLM occasionally emits a flash cue (e.g. a 0.7s "What do you
    think?") despite the prompt guardrail; QC's MinDuration rule flags these
    downstream.  Deterministically merge each flash unit into whichever
    neighbour yields the shorter combined cue, extending words/text/meta.
    """
    i = 0
    while i < len(units):
        unit = units[i]
        if unit.end - unit.start >= min_duration or len(units) == 1:
            i += 1
            continue
        # Choose the neighbour giving the shorter merged cue; prefer previous on ties.
        if i > 0:
            prev_words = units[i - 1].words + unit.words
            prev_dur = prev_words[-1].end - prev_words[0].start
        else:
            prev_dur = float("inf")
        if i + 1 < len(units):
            next_words = unit.words + units[i + 1].words
            next_dur = next_words[-1].end - next_words[0].start
        else:
            next_dur = float("inf")
        if prev_dur <= next_dur:
            target, j, merged_words = units[i - 1], i - 1, units[i - 1].words + unit.words
            meta[j]["word_end"] = meta[i]["word_end"]
            meta[j]["end"] = meta[i]["end"]
        else:
            target, j, merged_words = units[i + 1], i + 1, unit.words + units[i + 1].words
            meta[j]["word_start"] = meta[i]["word_start"]
            meta[j]["start"] = meta[i]["start"]
        target.words = merged_words
        target.start = merged_words[0].start
        target.end = merged_words[-1].end
        target.source_text = _text_from_words(merged_words)
        meta[j]["text"] = target.source_text
        del units[i], meta[i]
        # Re-check from the merge point: the merged cue may now border another flash.
        i = max(0, j - 1)


def _return_leading_punctuation(units: list[Segment], meta: list[dict]) -> None:
    """Return a unit's leading sentence-ending punctuation to the previous unit's text.

    ASR may glue sentence-final punctuation onto the next word (``" ? It's"``
    is a single token), so a word-level boundary can leave ``"? "`` at the
    start of a unit's display text.  The punctuation carries no timing of its
    own, so only the display text is rebalanced — word timelines, unit ids and
    word-coverage are untouched.  Guards: the first unit is left alone; a
    punctuation-only unit is never emptied; punctuation is not stacked onto a
    previous unit that already ends with sentence punctuation.
    """
    for i in range(1, len(units)):
        text = units[i].source_text
        stripped = text.lstrip()
        if not stripped or stripped[0] not in SENTENCE_ENDS:
            continue
        # Full leading punctuation run ("...", "?!") attaches to the previous unit.
        run_len = 0
        while run_len < len(stripped) and stripped[run_len] in SENTENCE_ENDS:
            run_len += 1
        punct = stripped[:run_len]
        rest = stripped[run_len:].lstrip()
        if not rest:  # punctuation-only unit — never empty it
            continue
        prev_text = units[i - 1].source_text.rstrip()
        if prev_text and prev_text[-1] in SENTENCE_ENDS:
            continue
        units[i - 1].source_text = prev_text + punct
        units[i].source_text = rest
        meta[i - 1]["text"] = units[i - 1].source_text
        meta[i]["text"] = units[i].source_text


def _text_from_words(words: list[Word]) -> str:
    """Rebuild display text from word tokens (punctuation rides on words)."""
    return " ".join(clean for w in words if (clean := w.text.strip()))


def _save_plan(plan_dir: Path, meta: list[dict]) -> None:
    artifacts.write_plan_meta(plan_dir / artifacts.PLAN_JSON, meta, version=_PLAN_VERSION)
