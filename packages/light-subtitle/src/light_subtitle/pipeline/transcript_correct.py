"""Transcript correction — LLM-based ASR error fixes with domain-aware correction.

Extracts domain context from the full transcript before correction, then
corrects homophones, proper nouns, duplicate words, and grammar word forms
via batched LLM calls with expanded context windows.

Flow::

    words → domain context extraction (LLM) → gap-based grouping → merge short
          → batch (context + target) → LLM → diff-based word alignment → return

Debug artifacts (``output_dir/transcript_correct/``)::

    domain_context.json — extracted domain, topics, and terminology
    pre_correct.json     — gap-grouped words before correction
    post_correct.json    — words after correction (with changed flags)
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from light_models import Word

from .. import logger
from ..config import SubtitleConfig
from ..llm.client import OpenAIClient
from ..llm.prompts import render_prompt
from ..usage.tracker import format_token_usage, merge_token_usage, save_step_usage
from ._word_segments import WordSegment, group_words_by_gap, join_word_text, merge_short_segments

_CHUNK_SIZE = 50
_CONTEXT_WINDOW = 5
_MAX_WORKERS = 4
_MAX_DELTA = 2
_DEFAULT_CONFIDENCE = 0.85
_GRAMMAR_MIN_WORDS = 4

_CONTEXT_LOG = logging.getLogger(__name__)


def correct_transcript(
    words: list[Word],
    config: SubtitleConfig,
    output_dir: str | Path,
) -> tuple[list[Word], dict | None]:
    """Fix ASR errors in *words* via LLM while preserving timestamps."""
    if not words or not config.llm_api_key or not config.correct_enabled:
        return words, None

    output_dir = Path(output_dir)
    correct_dir = output_dir / "transcript_correct"
    correct_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    # Step 0: extract domain context from the full transcript
    domain_context, domain_usage = _extract_domain_context(client, words, correct_dir)

    # Step 1: group words into pause-based segments
    segments = group_words_by_gap(words)
    logger.info(f"  Transcript correct: {len(words)} words → {len(segments)} segments")

    segments = merge_short_segments(segments)
    _save_segments(segments, str(correct_dir / "pre_correct.json"), include_changed=False)

    if not segments:
        return words, None

    # Step 2: build system prompt with domain context injected
    domain_str = _format_domain_context(domain_context)
    system_prompt = render_prompt("transcript_correct.j2", domain_context=domain_str)

    # Step 3: batch and correct
    chunks: list[list[WordSegment]] = []
    for i in range(0, len(segments), _CHUNK_SIZE):
        chunks.append(segments[i : i + _CHUNK_SIZE])

    all_results: dict[int, str] = {}
    batch_usage: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(chunks))) as executor:
        futures = {
            executor.submit(_correct_batch, client, system_prompt, chunk, segments): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            response_str, usage = future.result()
            all_results[idx] = response_str
            merge_token_usage(batch_usage, usage)

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
        f"{format_token_usage(batch_usage)}"
    )

    breakdown: dict[str, dict] = {"correct": batch_usage}
    if domain_usage:
        breakdown["correct.domain_context"] = domain_usage
    usage_payload: dict = {"breakdown": breakdown}
    merge_token_usage(usage_payload, domain_usage)
    merge_token_usage(usage_payload, batch_usage)
    save_step_usage(correct_dir / "usage.json", usage_payload)

    rebuilt: list[Word] = []
    for seg in segments:
        rebuilt.extend(seg.words)
    return rebuilt, usage_payload


def preserve_leading_space(old: str, new: str) -> str:
    """Keep whisper's leading-space convention when applying corrected text."""
    if old.startswith(" ") and new and not new.startswith(" "):
        return " " + new
    return new


def _redistribute_timing(words: list[Word], new_texts: list[str]) -> list[Word]:
    """Redistribute timestamps when word count changes via linear interpolation.

    Preserves the segment boundary (first word start, last word end) and maps
    the original timing contour onto the new word sequence.
    """
    if not words or not new_texts:
        return words

    n = len(words)
    m = len(new_texts)

    seg_start = words[0].start
    seg_end = words[-1].end
    orig_bounds = [seg_start] + [w.end for w in words]

    new_bounds = [seg_start]
    for k in range(1, m):
        pos = k / m
        orig_idx = pos * n
        i = int(orig_idx)
        if i >= n:
            i = n - 1
        f = orig_idx - i
        t = orig_bounds[i] * (1.0 - f) + orig_bounds[i + 1] * f
        new_bounds.append(t)
    new_bounds.append(seg_end)

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
    """Apply corrected tokens, allowing ±MAX_DELTA word count changes.

    - If word count matches → 1:1 word-level correction (timestamps unchanged).
    - If delta within ±MAX_DELTA and original ≥ _GRAMMAR_MIN_WORDS
      → align via difflib and rebuild with redistributed timing.
    - Otherwise → skip the segment, log a warning.
    """
    if not isinstance(corrected_tokens, list):
        return False
    tokens = [str(t) for t in corrected_tokens]

    n, m = len(words), len(tokens)
    if n == m:
        return apply_word_corrections(words, tokens)

    delta = m - n
    if abs(delta) <= _MAX_DELTA and n >= _GRAMMAR_MIN_WORDS:
        _align_and_apply(words, tokens)
        return True

    logger.warning(f"    Transcript correct skipped segment: word count {n} → {m} (delta={delta})")
    return False


def _align_and_apply(words: list[Word], tokens: list[str]) -> None:
    """Align old words with new tokens via word-level difflib, rebuild with redistributed timing.

    Uses difflib on stripped word forms to compute the edit script, then
    applies the sequence of operations (equal/replace/delete/insert) to build
    the corrected word list. Timing is redistributed proportionally.
    """
    old_texts = [w.text for w in words]

    old_stripped = [t.strip().lower() for t in old_texts]
    new_stripped = [t.strip().lower() for t in tokens]

    matcher = difflib.SequenceMatcher(None, old_stripped, new_stripped)

    result_texts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result_texts.extend(old_texts[i1:i2])
        elif tag == "replace":
            for j in range(j1, j2):
                if j - j1 < i2 - i1:
                    old_idx = i1 + (j - j1)
                    result_texts.append(preserve_leading_space(old_texts[old_idx], tokens[j]))
                else:
                    result_texts.append(_infer_leading_space(old_texts, tokens[j], len(result_texts)))
        elif tag == "delete":
            pass
        elif tag == "insert":
            for j in range(j1, j2):
                result_texts.append(_infer_leading_space(old_texts, tokens[j], len(result_texts)))

    new_words = _redistribute_timing(words, result_texts)
    words[:] = new_words


def _infer_leading_space(old_texts: list[str], token: str, result_len: int) -> str:
    """Determine whether *token* should have a leading space based on context."""
    if token.startswith(" "):
        return token
    if result_len > 0:
        return " " + token
    return token


def _extract_domain_context(
    client: OpenAIClient,
    words: list[Word],
    correct_dir: Path,
) -> tuple[dict, dict | None]:
    """Extract domain, topics, and terminology from the full transcript via LLM.

    Returns cached result if ``domain_context.json`` already exists.
    """
    cache_path = correct_dir / "domain_context.json"
    usage_path = correct_dir / "usage.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info(f"  Transcript correct: using cached domain context ({len(data.get('terminology', []))} terms)")
        cached_usage: dict | None = None
        if usage_path.exists():
            saved = json.loads(usage_path.read_text(encoding="utf-8"))
            breakdown = saved.get("breakdown") or {}
            cached_usage = breakdown.get("correct.domain_context")
        return data, cached_usage

    full_text = join_word_text(words)
    prompt = render_prompt("correct_context.j2", full_text=full_text)

    try:
        response, usage = client.chat([{"role": "user", "content": prompt}], temperature=0.1)
    except Exception as e:
        logger.warning(f"  Domain context extraction failed: {e}")
        return {"domain": "", "topics": [], "terminology": []}, None

    context = _parse_domain_context(response)
    cache_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"  Transcript correct: extracted domain context "
        f"({len(context.get('terminology', []))} terms, {format_token_usage(usage)})"
    )
    return context, usage


def _parse_domain_context(response: str) -> dict:
    """Parse LLM response into domain context dict."""
    json_match = re.search(r"\{[\s\S]*\}", response)
    raw: dict = {}
    if json_match:
        try:
            raw = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    if not raw and response.strip():
        try:
            raw = json.loads(response.strip())
        except json.JSONDecodeError:
            pass

    if not isinstance(raw, dict):
        return {"domain": "", "topics": [], "terminology": []}

    terminology = raw.get("terminology", [])
    if not isinstance(terminology, list):
        terminology = []

    return {
        "domain": str(raw.get("domain", "")),
        "topics": [str(t) for t in raw.get("topics", []) if isinstance(raw.get("topics"), list)],
        "terminology": terminology,
    }


def _format_domain_context(context: dict) -> str:
    """Format domain context dict as a string for injection into the correction prompt."""
    parts: list[str] = []

    domain = context.get("domain", "")
    if domain:
        parts.append(f"Domain: {domain}")

    topics = context.get("topics", [])
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")

    terminology = context.get("terminology", [])
    if terminology:
        lines = ["Known terminology (prefer these spellings when phonetically similar):"]
        for entry in terminology:
            if isinstance(entry, dict) and entry.get("term"):
                term = entry["term"]
                ctx = entry.get("context", "")
                if ctx:
                    lines.append(f"  - {term}  ({ctx})")
                else:
                    lines.append(f"  - {term}")
        parts.append("\n".join(lines))

    return "\n".join(parts)


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
