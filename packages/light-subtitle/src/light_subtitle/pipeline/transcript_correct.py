"""Transcript correction — LLM-based ASR error fixes with word-level safety.

Corrects homophones, spelling, and obvious grammar word forms without
changing word count or timestamps.

Flow::

    words → gap-based grouping → merge short → batch (context + target) → LLM
        → 1:1 word array back to words → return

Debug artifacts (``output_dir/transcript_correct/``)::

    pre_correct.json  — gap-grouped words before correction
    post_correct.json — words after correction (with changed flags)
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from light_models import Word

from .. import logger
from ..config import SubtitleConfig
from ..llm.client import OpenAIClient, format_token_usage, merge_token_usage
from ..llm.prompts import render_prompt
from ._word_segments import WordSegment, group_words_by_gap, join_word_text, merge_short_segments

_CHUNK_SIZE = 10
_CONTEXT_WINDOW = 2
_MAX_WORKERS = 4
_DEFAULT_CONFIDENCE = 0.85
_GRAMMAR_MIN_WORDS = 4


def correct_transcript(
    words: list[Word],
    config: SubtitleConfig,
    output_dir: str | Path,
) -> list[Word]:
    """Fix ASR errors in *words* via LLM while preserving word count and timestamps."""
    if not words or not config.llm_api_key or not config.correct_enabled:
        return words

    output_dir = Path(output_dir)
    correct_dir = output_dir / "transcript_correct"
    correct_dir.mkdir(parents=True, exist_ok=True)

    segments = group_words_by_gap(words)
    logger.info(f"  Transcript correct: {len(words)} words → {len(segments)} segments")

    segments = merge_short_segments(segments)
    _save_segments(segments, str(correct_dir / "pre_correct.json"), include_changed=False)

    if not segments:
        return words

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )
    system_prompt = render_prompt("transcript_correct.j2")

    chunks: list[list[WordSegment]] = []
    for i in range(0, len(segments), _CHUNK_SIZE):
        chunks.append(segments[i : i + _CHUNK_SIZE])

    all_results: dict[int, str] = {}
    total_usage: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(chunks))) as executor:
        futures = {
            executor.submit(_correct_batch, client, system_prompt, chunk, segments): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            response_str, usage = future.result()
            all_results[idx] = response_str
            merge_token_usage(total_usage, usage)

    changed_words = 0
    for chunk_idx in range(len(chunks)):
        response_str = all_results.get(chunk_idx, "")
        if not response_str:
            continue
        corrected_segments = _parse_llm_response(response_str)
        for item in corrected_segments:
            seg = segments[item["index"]]
            pre_texts = [w.text for w in seg.words]
            if _apply_word_corrections(seg.words, item.get("words", [])):
                post_texts = [w.text for w in seg.words]
                changed_words += sum(1 for old, new in zip(pre_texts, post_texts, strict=False) if old != new) + abs(
                    len(post_texts) - len(pre_texts)
                )

    _save_segments(segments, str(correct_dir / "post_correct.json"), include_changed=True)
    logger.info(
        f"  Transcript corrected: {changed_words} word(s) changed across {len(segments)} segments, "
        f"{format_token_usage(total_usage)}"
    )

    # Rebuild the flat word list from segments — grammar fixes may have
    # changed word counts per segment, and those changes happen on
    # WordSegment.words (a slice), not on the top-level *words* list.
    rebuilt: list[Word] = []
    for seg in segments:
        rebuilt.extend(seg.words)
    return rebuilt


def preserve_leading_space(old: str, new: str) -> str:
    """Keep whisper's leading-space convention when applying corrected text."""
    if old.startswith(" ") and new and not new.startswith(" "):
        return " " + new
    return new


def _redistribute_timing(words: list[Word], new_texts: list[str]) -> list[Word]:
    """Redistribute timestamps when word count changes via linear interpolation.

    Preserves the segment boundary (first word start, last word end) and maps
    the original timing contour onto the new word sequence.

    Original N words have N+1 boundary points B[0..N]:
        B[0] = words[0].start
        B[i] = words[i-1].end   for 0 < i < N
        B[N] = words[-1].end

    For M new tokens we produce M new boundary points B'[0..M]
    via linear interpolation of *position* within the segment.
    """
    if not words or not new_texts:
        return words

    n = len(words)
    m = len(new_texts)

    # Original boundaries.
    seg_start = words[0].start
    seg_end = words[-1].end
    orig_bounds = [seg_start] + [w.end for w in words]

    # New boundaries via fractional mapping.
    new_bounds = [seg_start]
    for k in range(1, m):
        pos = k / m  # fractional position in new seq [0, 1]
        orig_idx = pos * n  # same position in original [0, n]
        i = int(orig_idx)  # left bound index
        if i >= n:
            i = n - 1
        f = orig_idx - i  # fraction between i and i+1
        t = orig_bounds[i] * (1.0 - f) + orig_bounds[i + 1] * f
        new_bounds.append(t)
    new_bounds.append(seg_end)

    # Compute average confidence for the segment.
    avg_conf = sum(w.confidence for w in words) / n if n else _DEFAULT_CONFIDENCE

    result: list[Word] = []
    for j in range(m):
        orig_idx = int(j * n / m) if m > 1 else 0
        if orig_idx >= n:
            orig_idx = n - 1
        w = Word(
            text=new_texts[j],
            start=new_bounds[j],
            end=new_bounds[j + 1],
            confidence=avg_conf,
            speaker=words[orig_idx].speaker,
        )
        result.append(w)

    return result


def apply_word_corrections(words: list[Word], corrected_tokens: list[str]) -> bool:
    """Apply 1:1 corrected tokens to *words*. Returns False if word count mismatches."""
    if len(words) != len(corrected_tokens):
        return False
    for word, new_text in zip(words, corrected_tokens, strict=True):
        word.text = preserve_leading_space(word.text, str(new_text))
    return True


def _apply_word_corrections(words: list[Word], corrected_tokens: list) -> bool:
    """Apply corrected tokens, allowing ±1 word for grammar fixes.

    - If word count matches → 1:1 word-level correction (timestamps unchanged).
    - If word count differs by exactly 1 and original ≥ ``_GRAMMAR_MIN_WORDS``
      → grammar fix: rebuild word list with redistributed timing.
    - Otherwise → skip the segment, log a warning.
    """
    if not isinstance(corrected_tokens, list):
        return False
    tokens = [str(t) for t in corrected_tokens]

    n, m = len(words), len(tokens)
    if n == m:
        return apply_word_corrections(words, tokens)

    delta = m - n
    if delta == 1 and n >= _GRAMMAR_MIN_WORDS:
        word_texts = []
        for i, t in enumerate(tokens):
            if i == 0 and delta > 0:
                word_texts.append(t)
            else:
                old_idx = max(0, i - max(delta, 0))
                old_idx = min(old_idx, n - 1)
                word_texts.append(preserve_leading_space(words[old_idx].text, t))
        new_words = _redistribute_timing(words, word_texts)
        words[:] = new_words
        return True

    logger.warning(f"    Transcript correct skipped segment: word count {n} → {m} (delta={delta})")
    return False


def _correct_batch(
    client: OpenAIClient,
    system_prompt: str,
    chunk: list[WordSegment],
    all_segments: list[WordSegment],
) -> tuple[str, dict]:
    """Send a chunk of segments (with context) to the LLM for correction."""
    chunk_start = chunk[0].index
    chunk_end = chunk[-1].index

    ctx_start = max(0, chunk_start - _CONTEXT_WINDOW)
    ctx_end = min(len(all_segments), chunk_end + 1 + _CONTEXT_WINDOW)

    if chunk_start >= len(all_segments) or chunk_end >= len(all_segments):
        fallback = json.dumps([{"index": s.index, "words": [w.text for w in s.words]} for s in chunk])
        return fallback, {}

    payload_units: list[dict] = []
    for i in range(ctx_start, chunk_start):
        s = all_segments[i]
        payload_units.append({"index": s.index, "words": [w.text for w in s.words], "context": True})
    for s in chunk:
        payload_units.append({"index": s.index, "words": [w.text for w in s.words], "context": False})
    for i in range(chunk_end + 1, ctx_end):
        s = all_segments[i]
        payload_units.append({"index": s.index, "words": [w.text for w in s.words], "context": True})

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
                logger.warning(f"    Transcript correct retry {attempt + 1}/{max_retries}, waiting {delay}s")
                time.sleep(delay)

    logger.warning(f"    Transcript correct failed after {max_retries} retries, using original text")
    fallback = json.dumps([{"index": s.index, "words": [w.text for w in s.words]} for s in chunk])
    return fallback, {}


def _parse_llm_response(response: str) -> list[dict]:
    """Extract list of {index, words} from LLM JSON response."""
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


def _save_segments(segments: list[WordSegment], output_path: str, *, include_changed: bool) -> None:
    """Save segments as JSON, optionally marking per-word changes."""
    pre_texts: dict[int, list[str]] = {}
    if include_changed:
        pre_path = Path(output_path).parent / "pre_correct.json"
        if pre_path.exists():
            pre_data = json.loads(pre_path.read_text(encoding="utf-8"))
            for entry in pre_data:
                pre_texts[entry["index"]] = [w["text"] for w in entry.get("words", [])]

    data = []
    for s in segments:
        word_entries = []
        pre = pre_texts.get(s.index, [])
        for wi, w in enumerate(s.words):
            entry: dict = {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
                "speaker": w.speaker,
            }
            if include_changed:
                entry["changed"] = wi < len(pre) and w.text != pre[wi]
            word_entries.append(entry)
        data.append(
            {
                "index": s.index,
                "start": s.words[0].start if s.words else 0.0,
                "end": s.words[-1].end if s.words else 0.0,
                "word_count": len(s.words),
                "text": join_word_text(s.words),
                "words": word_entries,
            }
        )
    Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
