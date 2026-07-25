"""Punctuation restoration — LLM-based, with context window and parallelism.

After wav2vec2 alignment, whisper word-level output often lacks
punctuation (especially for conversational audio).  This module adds
punctuation by sending pause-based segments to an LLM.

Flow::

    words → gap-based grouping → merge short → batch (context + target) → LLM
        → word-level diff back to words → return

Debug artifacts (written under ``work_dir/punct_restore/`` when *work_dir*
is given; skipped entirely when *work_dir* is ``None``)::

    pre_punct.json     — gap-grouped words before punctuation restoration
    punct_restore.json — words after punctuation restoration
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from light_core import logger
from light_core.progress import ProgressCallback
from light_llm.client import OpenAIClient
from light_llm.json_extract import parse_json_array_response
from light_llm.parallel import run_parallel_with_warmup
from light_llm.retry import chat_with_retry
from light_llm.usage.tracker import format_token_usage, merge_token_usage, save_step_usage
from light_models import Word, word_to_dict
from light_text.punctuation import SENTENCE_ENDS

from .prompts import render_prompt
from .word_segments import WordSegment, group_words_by_gap, join_word_text, merge_short_segments

# ── Constants ──────────────────────────────────────────────────────

_CHUNK_SIZE = 10
_CONTEXT_WINDOW = 2
_MAX_WORKERS = 4
_PUNCT_CHARS = set(",.?!:;，。？！、；：")
_PUNCT_SUFFICIENT_THRESHOLD = 0.3

# Backward-compatible aliases for tests.
_Segment = WordSegment
_join_text = join_word_text
_merge_short_segments = merge_short_segments


# ── Public API ─────────────────────────────────────────────────────


def restore_punct(
    words: list[Word],
    llm: OpenAIClient,
    work_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[Word], dict | None]:
    """Add punctuation to *words* via LLM."""
    if not words:
        return words, None

    punct_dir = Path(work_dir) / "punct_restore" if work_dir is not None else None
    if punct_dir is not None:
        punct_dir.mkdir(parents=True, exist_ok=True)

    segments = group_words_by_gap(words)
    logger.info(f"  Punct restore: {len(words)} words → {len(segments)} segments")

    segments = merge_short_segments(segments)
    if punct_dir is not None:
        _save_segments(segments, str(punct_dir / "pre_punct.json"))

    if not segments:
        return words, None

    if _has_sufficient_punctuation(segments):
        logger.info("  Punct restore skipped (already punctuated)")
        if punct_dir is not None:
            _save_segments_restored(segments, str(punct_dir / "punct_restore.json"))
        return words, None

    client = llm
    system_prompt = render_prompt("restore_punct.j2")

    chunks: list[list[WordSegment]] = []
    for i in range(0, len(segments), _CHUNK_SIZE):
        chunks.append(segments[i : i + _CHUNK_SIZE])

    all_results: dict[int, str] = {}
    total_usage: dict[str, int] = {}

    tasks = [
        (
            lambda c=chunk, idx=idx: _restore_batch(client, system_prompt, c, segments),
            idx,
        )
        for idx, chunk in enumerate(chunks)
    ]
    batch_results = run_parallel_with_warmup(tasks, max_workers=_MAX_WORKERS)
    for idx, (response_str, usage) in batch_results.items():
        all_results[idx] = response_str
        merge_token_usage(total_usage, usage)

    for chunk_idx in range(len(chunks)):
        restored_str = all_results.get(chunk_idx, "")
        if not restored_str:
            continue
        restored_segments = _parse_llm_response(restored_str)
        for rs in restored_segments:
            seg = segments[rs["index"]]
            _apply_punct_to_words(seg.words, seg.text, rs["text"])

    if punct_dir is not None:
        _save_segments_restored(segments, str(punct_dir / "punct_restore.json"))
    logger.info(f"  Punct restored: {len(segments)} segments, {format_token_usage(total_usage)}")
    if punct_dir is not None:
        save_step_usage(punct_dir / "usage.json", total_usage)
    if progress:
        progress(1.0, None)

    return words, total_usage


# ── Punctuation sufficiency check ──────────────────────────────────


def _has_sufficient_punctuation(segments: list[WordSegment], threshold: float = _PUNCT_SUFFICIENT_THRESHOLD) -> bool:
    """Check if enough segments already end with sentence-ending punctuation."""
    if not segments:
        return False
    punctuated = sum(1 for s in segments if s.text.rstrip() and s.text.rstrip()[-1] in SENTENCE_ENDS)
    return punctuated / len(segments) >= threshold


# ── LLM batch ──────────────────────────────────────────────────────


def _restore_batch(
    client: OpenAIClient,
    system_prompt: str,
    chunk: list[WordSegment],
    all_segments: list[WordSegment],
) -> tuple[str, dict]:
    """Send a chunk of segments (with context) to the LLM for punctuation."""
    chunk_start = chunk[0].index
    chunk_end = chunk[-1].index

    ctx_start = max(0, chunk_start - _CONTEXT_WINDOW)
    ctx_end = min(len(all_segments), chunk_end + 1 + _CONTEXT_WINDOW)

    if chunk_start >= len(all_segments) or chunk_end >= len(all_segments):
        fallback = json.dumps([{"index": s.index, "text": s.text} for s in chunk])
        return fallback, {}

    payload_units: list[dict] = []
    for i in range(ctx_start, chunk_start):
        s = all_segments[i]
        payload_units.append({"index": s.index, "text": s.text, "context": True})
    for s in chunk:
        payload_units.append({"index": s.index, "text": s.text, "context": False})
    for i in range(chunk_end + 1, ctx_end):
        s = all_segments[i]
        payload_units.append({"index": s.index, "text": s.text, "context": True})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload_units, ensure_ascii=False)},
    ]

    max_retries = 3

    def _on_retry(attempt: int, _exc: BaseException) -> float:
        delay = 2**attempt
        logger.warning(f"    Punct restore retry {attempt + 1}/{max_retries}, waiting {delay}s")
        return delay

    try:
        return chat_with_retry(
            lambda: client.chat(messages, temperature=0.0),
            max_retries=max_retries,
            on_retry=_on_retry,
        )
    except Exception as e:
        logger.warning(
            f"    Punct restore failed after {max_retries} retries ({type(e).__name__}: {e}), using original text"
        )
        fallback = json.dumps([{"index": s.index, "text": s.text} for s in chunk])
        return fallback, {}


# Shared implementation (kept as an alias; tests import this name).
_parse_llm_response = parse_json_array_response


# ── Word-level punctuation diff ────────────────────────────────────


def _apply_punct_to_words(words: list[Word], old_text: str, new_text: str) -> None:
    """Apply punctuation from LLM output to original words via character diff."""
    if not words or old_text == new_text:
        return

    char_to_word: dict[int, int] = {}
    pos = 0
    for wi, w in enumerate(words):
        stripped = w.text.strip()
        while pos < len(old_text) and old_text[pos].isspace():
            pos += 1
        for _ in stripped:
            if pos < len(old_text):
                char_to_word[pos] = wi
                pos += 1

    sm = difflib.SequenceMatcher(
        lambda c: c in " \t",
        old_text.lower(),
        new_text.lower(),
        autojunk=False,
    )

    punct_by_word: dict[int, str] = {}

    for tag, i1, _i2, j1, j2 in sm.get_opcodes():
        if tag not in ("insert", "replace"):
            continue
        for j in range(j1, j2):
            ch = new_text[j]
            if ch not in _PUNCT_CHARS:
                continue
            target = i1 - 1
            while target >= 0 and target not in char_to_word:
                target -= 1
            if target in char_to_word:
                wi = char_to_word[target]
                existing = punct_by_word.get(wi, "")
                if ch not in existing:
                    punct_by_word[wi] = existing + ch

    for wi, punct in punct_by_word.items():
        if wi >= len(words):
            continue
        w = words[wi]
        trail = w.text.rstrip()
        for ch in punct:
            if not trail.endswith(ch):
                trail += ch
        trail_space = len(w.text) - len(w.text.rstrip())
        w.text = trail + " " * trail_space


# ── Save debug artifacts ───────────────────────────────────────────


def _save_segments(segments: list[WordSegment], output_path: str) -> None:
    """Save pre-punct segments as JSON."""
    data = [
        {
            "index": s.index,
            "start": s.words[0].start if s.words else 0.0,
            "end": s.words[-1].end if s.words else 0.0,
            "word_count": len(s.words),
            "text": s.text,
            "words": [word_to_dict(w) for w in s.words],
        }
        for s in segments
    ]
    Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_segments_restored(segments: list[WordSegment], output_path: str) -> None:
    """Save post-punct segments as JSON."""
    data = [
        {
            "index": s.index,
            "start": s.words[0].start if s.words else 0.0,
            "end": s.words[-1].end if s.words else 0.0,
            "word_count": len(s.words),
            "text": join_word_text(s.words),
            "words": [word_to_dict(w) for w in s.words],
        }
        for s in segments
    ]
    Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
