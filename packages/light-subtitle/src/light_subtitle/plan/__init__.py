"""Cue planning — single LLM pass per shard with cum-time duration hints.

Pipeline: normalize → pre-split → LLM per shard → tail fix → flash merge
→ gap-split → assemble → judge+refine.
"""

from __future__ import annotations

from pathlib import Path

from light_core import logger
from light_core.progress import ProgressCallback
from light_llm.client import OpenAIClient
from light_llm.parallel import run_parallel_with_warmup
from light_llm.usage.tracker import merge_token_usage
from light_models import Segment, Word

from .. import artifacts
from ..config import PlanConfig
from . import gap, normalize, plan_refine, planner

_PLAN_VERSION = 4
_MAX_WORKERS = 4
_PRE_SPLIT_GAP = 3.0
_MAX_SHARD_WORDS = 300


def run(
    segments: list[Segment],
    config: PlanConfig,
    plan_dir: str | Path,
    *,
    llm: OpenAIClient | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[Segment], dict | None]:
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    if not segments:
        _save_plan(plan_dir, [])
        return [], None
    if llm is None:
        logger.warning("  Plan: no LLM client — cannot plan units")
        _save_plan(plan_dir, [])
        return [], None

    words = [w for seg in segments for w in seg.words]
    nwords = normalize.normalize(words)

    seg_of_word: list[int] = []
    for i, seg in enumerate(segments):
        seg_of_word.extend([i] * len(seg.words))

    shards = _pre_split(nwords, seg_of_word, segments)
    logger.info(f"  Plan: {len(words)} words → {len(shards)} shard(s)")

    total_usage: dict = {}
    shared_client = llm

    tasks = []
    for idx, (ws, we) in enumerate(shards):
        sw = words[ws:we]
        nw = nwords[ws:we]
        seg_ids = seg_of_word[ws:we]
        markers = _find_markers(nw, seg_ids)
        cum = _compute_cum(sw)

        def _task(
            wl: list = sw, ct: list = cum, mk: set = markers, cfg: PlanConfig = config
        ) -> tuple[list[tuple[int, int, str]], dict]:
            return _plan_one(wl, ct, mk, cfg, shared_client)

        tasks.append((_task, idx))

    n_shards = len(shards)
    done = 0

    def _on_done() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done / n_shards, f"规划字幕边界中... {done}/{n_shards}")

    raw_results = run_parallel_with_warmup(tasks, max_workers=_MAX_WORKERS, on_complete=_on_done)

    all_ranges: list[tuple[int, int, str]] = []
    for idx in range(len(shards)):
        ranges, usage = raw_results[idx]
        if usage:
            merge_token_usage(total_usage, usage)
        ws_offset = shards[idx][0]
        for r_start, r_end, speaker in ranges:
            all_ranges.append((r_start + ws_offset, r_end + ws_offset, speaker))

    units, meta = _assemble(all_ranges, words)

    refined, issues, j_usage = plan_refine.refine_plan(units, segments, config, llm)
    if j_usage:
        merge_token_usage(total_usage, j_usage)
    if issues:
        units = refined
        meta = _rebuild_meta(units)

    _save_plan(plan_dir, meta)
    logger.info(f"  Plan: {len(units)} unit(s)")
    return units, total_usage or None


def load_plan_units(plan_dir: str | Path) -> list[Segment] | None:
    path = Path(plan_dir) / artifacts.PLAN_JSON
    if not path.exists():
        return None
    return artifacts.read_plan_units(path)


# ── Shard processing ──────────────────────────────────────────


def _plan_one(
    shard_words: list[Word],
    cum_times: list[float],
    markers: set[int],
    config: PlanConfig,
    client: OpenAIClient,
) -> tuple[list[tuple[int, int, str]], dict]:
    usage: dict = {}

    breaks, u = planner.plan_shard(shard_words, cum_times, markers, config, client)
    if u:
        merge_token_usage(usage, u)
    if breaks is None:
        breaks = [len(shard_words) - 1]
        logger.warning("  Plan: LLM failed — using whole shard as one unit")

    breaks = planner.fix_illegal_tails(breaks, shard_words)

    units: list[tuple[int, int, str]] = []
    prev_end = -1
    for b in breaks:
        w_start = prev_end + 1
        w_end = b + 1
        dur = planner.compute_duration(cum_times, prev_end, w_end - 1)
        if dur > config.max_duration and w_end - w_start > 1:
            sub = gap.gap_split(shard_words[w_start:w_end], config.max_duration)
            for ss, se in sub:
                units.append((w_start + ss, w_start + se, shard_words[w_start + ss].speaker or ""))
        else:
            units.append((w_start, w_end, shard_words[w_start].speaker or ""))
        prev_end = b

    # fix tails on gap-split results too
    units = _fix_unit_tails(units, shard_words)
    units = _merge_flash_units(units, shard_words, config)
    return units, usage


# ── Post-processing ─────────────────────────────────────────


def _fix_unit_tails(units: list[tuple[int, int, str]], words: list[Word]) -> list[tuple[int, int, str]]:
    """Slide right boundaries of units that end on a bare function word."""
    from .lexicon import FUNC_TAIL
    for i in range(len(units) - 1):
        s, e, sp = units[i]
        while e < len(words):
            last = words[e - 1]
            text = last.text.strip()
            if not text:
                break
            if any(text.endswith(p) for p in (".", ",", "!", "?", ";", ":", "—", "…")):
                break
            core = text.strip(".,!?;:—\"'()[]“”‘’").lower()
            if core not in FUNC_TAIL:
                break
            e += 1
        if e != units[i][1]:
            shift = e - units[i][1]
            units[i] = (s, e, sp)
            if i + 1 < len(units):
                units[i + 1] = (units[i + 1][0] + shift, units[i + 1][1], units[i + 1][2])
    return units


def _merge_flash_units(
    units: list[tuple[int, int, str]], words: list[Word], config: PlanConfig
) -> list[tuple[int, int, str]]:
    """Merge units < min_duration into a neighbour, preserving word coverage."""
    if len(units) <= 1:
        return units
    result: list[tuple[int, int, str]] = []
    i = 0
    while i < len(units):
        s, e, sp = units[i]
        dur = words[e - 1].end - words[s].start
        if dur >= config.min_duration or (e - s) >= 4:
            result.append(units[i])
            i += 1
            continue

        # Flash: merge with the neighbour giving the shorter combined duration
        if i > 0 and i + 1 < len(units):
            prev_dur = words[e - 1].end - words[result[-1][0]].start
            next_dur = words[units[i + 1][1] - 1].end - words[s].start
            if prev_dur <= next_dur:
                result[-1] = (result[-1][0], e, result[-1][2])
            else:
                units[i + 1] = (s, units[i + 1][1], units[i + 1][2])
        elif i > 0:
            result[-1] = (result[-1][0], e, result[-1][2])
        elif i + 1 < len(units):
            result.append((s, units[i + 1][1], units[i + 1][2]))
            i += 1
        else:
            result.append(units[i])
        i += 1
    return result


# ── Pre-split ──────────────────────────────────────────────────


def _pre_split(nwords, seg_of_word, segments) -> list[tuple[int, int]]:
    shards = []
    start = 0
    for i in range(1, len(nwords)):
        gap = nwords[i].start - nwords[i - 1].end
        sp_cur = segments[seg_of_word[i - 1]].speaker or ""
        sp_nxt = segments[seg_of_word[i]].speaker or ""
        speaker_change = sp_cur and sp_nxt and sp_cur != sp_nxt
        too_large = (i - start) >= _MAX_SHARD_WORDS and gap >= 0.50
        if gap > _PRE_SPLIT_GAP or speaker_change or too_large:
            shards.append((start, i))
            start = i
    shards.append((start, len(nwords)))
    return shards


# ── Annotation ────────────────────────────────────────────────


def _find_markers(nwords, seg_ids: list[int]) -> set[int]:
    """Return indices of words that carry a `|` marker: sentence endings or gaps ≥ 0.3s."""
    markers: set[int] = set()
    for i in range(len(nwords) - 1):
        w = nwords[i]
        if w.is_sentence_final or w.gap_after >= 0.30:
            markers.add(i)
    return markers
    markers: set[int] = set()
    for i in range(len(nwords) - 1):
        w = nwords[i]
        if w.is_sentence_final or w.gap_after >= 0.50:
            markers.add(i)
    return markers


def _compute_cum(words: list[Word]) -> list[float]:
    cum: list[float] = []
    start = words[0].start
    for w in words:
        cum.append(round(w.end - start, 2))
    return cum


# ── Assembly ──────────────────────────────────────────────────


def _assemble(
    ranges: list[tuple[int, int, str]], words: list[Word]
) -> tuple[list[Segment], list[dict]]:
    units: list[Segment] = []
    meta: list[dict] = []
    for i, (ws, we, speaker) in enumerate(ranges):
        u_words = words[ws:we]
        text = " ".join(w.text.strip() for w in u_words)
        units.append(
            Segment(
                unit_id=f"p{i:04d}",
                start=u_words[0].start,
                end=u_words[-1].end,
                speaker=speaker,
                source_text=text,
                words=u_words,
            )
        )
        meta.append(
            {
                "unit_id": f"p{i:04d}",
                "start": round(u_words[0].start, 3),
                "end": round(u_words[-1].end, 3),
                "speaker": speaker,
                "text": text,
                "word_start": ws,
                "word_end": we,
            }
        )
    return units, meta


def _rebuild_meta(units: list[Segment]) -> list[dict]:
    """Rebuild plan meta from unit list (after refine)."""
    meta: list[dict] = []
    word_idx = 0
    for u in units:
        nw = len(u.words)
        meta.append(
            {
                "unit_id": u.unit_id,
                "start": round(u.start, 3),
                "end": round(u.end, 3),
                "speaker": u.speaker,
                "text": u.source_text,
                "word_start": word_idx,
                "word_end": word_idx + nw,
            }
        )
        word_idx += nw
    return meta


def _save_plan(plan_dir: Path, meta: list[dict]) -> None:
    artifacts.write_plan_meta(plan_dir / artifacts.PLAN_JSON, meta, version=_PLAN_VERSION)
