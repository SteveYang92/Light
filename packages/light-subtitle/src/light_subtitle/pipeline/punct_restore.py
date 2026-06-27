"""Punctuation restoration — LLM-based, with context window and parallelism.

After wav2vec2 alignment, whisper word-level output often lacks
punctuation (especially for conversational audio).  This module adds
punctuation by sending pause-based segments to an LLM.

Flow::

    words → gap-based grouping → merge short → batch (context + target) → LLM
        → word-level diff back to words → return

Debug artifacts (written to ``output_dir/punct_restore/``)::

    pre_punct.json     — gap-grouped words before punctuation restoration
    punct_restore.json — words after punctuation restoration
"""

from __future__ import annotations

import difflib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from light_models import Word
from light_models.punctuation import SENTENCE_ENDS

from .. import logger
from ..config import SubtitleConfig
from ..llm.client import OpenAIClient
from ..llm.prompts import render_prompt
from ..usage.tracker import format_token_usage, merge_token_usage, save_step_usage
from ._word_segments import WordSegment, group_words_by_gap, join_word_text, merge_short_segments

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


def restore_punctuation(
    words: list[Word],
    config: SubtitleConfig,
    output_dir: str | Path,
) -> tuple[list[Word], dict | None]:
    """Add punctuation to *words* via LLM."""
    if not words or not config.llm_api_key:
        return words, None

    output_dir = Path(output_dir)
    punct_dir = output_dir / "punct_restore"
    punct_dir.mkdir(parents=True, exist_ok=True)

    segments = group_words_by_gap(words)
    logger.info(f"  Punct restore: {len(words)} words → {len(segments)} segments")

    segments = merge_short_segments(segments)
    _save_segments(segments, str(punct_dir / "pre_punct.json"))

    if not segments:
        return words, None

    if _has_sufficient_punctuation(segments):
        logger.info("  Punct restore skipped (already punctuated)")
        _save_segments_restored(segments, str(punct_dir / "punct_restore.json"))
        return words, None

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )
    system_prompt = render_prompt("restore_punct.j2")

    chunks: list[list[WordSegment]] = []
    for i in range(0, len(segments), _CHUNK_SIZE):
        chunks.append(segments[i : i + _CHUNK_SIZE])

    all_results: dict[int, str] = {}
    total_usage: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(chunks))) as executor:
        futures = {
            executor.submit(_restore_batch, client, system_prompt, chunk, segments): i for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            response_str, usage = future.result()
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

    _save_segments_restored(segments, str(punct_dir / "punct_restore.json"))
    logger.info(f"  Punct restored: {len(segments)} segments, {format_token_usage(total_usage)}")
    save_step_usage(punct_dir / "usage.json", total_usage)

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
    for attempt in range(max_retries):
        try:
            response, usage = client.chat(messages, temperature=0.0)
            return response, usage
        except Exception:
            if attempt < max_retries - 1:
                delay = 2**attempt
                logger.warning(f"    Punct restore retry {attempt + 1}/{max_retries}, waiting {delay}s")
                time.sleep(delay)

    logger.warning(f"    Punct restore failed after {max_retries} retries, using original text")
    fallback = json.dumps([{"index": s.index, "text": s.text} for s in chunk])
    return fallback, {}


def _parse_llm_response(response: str) -> list[dict]:
    """Extract list of {index, text} from LLM JSON response."""
    json_match = re.search(r"\[[\s\S]*\]", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return []


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
            "words": [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    "speaker": w.speaker,
                }
                for w in s.words
            ],
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
            "words": [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    "speaker": w.speaker,
                }
                for w in s.words
            ],
        }
        for s in segments
    ]
    Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
