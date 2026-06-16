"""LLM translation — chunked, parallel, with retry and JSON parsing."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from light_models import Segment, SubtitleCue
from light_models.punctuation import CJK_CLAUSE_PUNCT, SENTENCE_ENDS

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient
from ...llm.prompts import render_prompt

CHUNK_SIZE = 100
MAX_WORKERS = 4
_SPLIT_PART_RE = re.compile(r"^(mu\d+_u\d+)_(\d+)$")
_EN_SENTENCE_END = frozenset(".!?…")


def _render_translate_prompt(config: SubtitleConfig) -> str:
    """Build system prompt with glossary and content summary."""
    return render_prompt(
        "translate.j2",
        target_lang=config.target_lang,
        glossary=config.glossary,
        content_summary=config.content_summary,
    )


def _translation_context_fields(config: SubtitleConfig) -> dict:
    """Shared glossary + summary fields for translation user payloads."""
    fields: dict = {
        "target_lang": config.target_lang,
        "glossary": config.glossary,
    }
    if config.content_summary is not None:
        fields["content_summary"] = config.content_summary
    return fields


def run(
    segments: list[Segment],
    config: SubtitleConfig,
    tx_dir: Path | None = None,
) -> tuple[list[SubtitleCue], dict | None]:
    """Return (translated_cues, usage_dict).

    When *tx_dir* is set, saves ``partial.json`` after each batch for resume.
    """
    if not config.llm_api_key:
        return [], None

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    system_prompt = _render_translate_prompt(config)
    existing: dict[str, SubtitleCue] = {}
    if tx_dir is not None:
        existing = {c.unit_id: c for c in load_partial_cues(tx_dir, config)}

    pending = [s for s in segments if s.unit_id not in existing]
    if not pending and existing:
        return _order_cues(segments, existing), None

    batch_chunks = _chunk_pending_segments(pending, CHUNK_SIZE)
    if len(batch_chunks) == 1:
        chunk = batch_chunks[0]
        abs_idx = segments.index(chunk[0]) if chunk else 0
        cues, usage = _translate_batch(client, system_prompt, chunk, segments, abs_idx, config)
        merged = dict(existing)
        for c in cues:
            merged[c.unit_id] = c
        if tx_dir is not None:
            _save_partial_cues(tx_dir, _order_cues(segments, merged))
        return _order_cues(segments, merged), usage

    merged = dict(existing)
    total_usage: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch_chunks))) as executor:
        futures = {
            executor.submit(
                _translate_batch,
                client,
                system_prompt,
                chunk,
                segments,
                segments.index(chunk[0]),
                config,
            ): segments.index(chunk[0])
            for chunk in batch_chunks
            if chunk
        }
        results: dict[int, tuple[list[SubtitleCue], dict]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

        for idx in sorted(results):
            cues, usage = results[idx]
            for c in cues:
                merged[c.unit_id] = c
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] = total_usage.get(k, 0) + usage.get(k, 0)
            if tx_dir is not None:
                _save_partial_cues(tx_dir, _order_cues(segments, merged))

    ordered = _order_cues(segments, merged)
    for i, c in enumerate(ordered):
        c.cue_id = f"{config.target_lang}_{i:04d}"

    return ordered, total_usage or None


def _order_cues(segments: list[Segment], by_id: dict[str, SubtitleCue]) -> list[SubtitleCue]:
    return [by_id[s.unit_id] for s in segments if s.unit_id in by_id]


def _save_partial_cues(tx_dir: Path, cues: list[SubtitleCue]) -> None:
    tx_dir.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "cue_id": c.cue_id,
            "unit_id": c.unit_id,
            "start": c.start,
            "end": c.end,
            "text": c.text,
            "lang": c.lang,
        }
        for c in cues
    ]
    (tx_dir / "partial.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_partial_cache(tx_dir: Path) -> bool:
    """Remove ``partial.json`` when compose/split is re-run.

    Unit ids may be reused with different source text or timing; stale
    partial entries would otherwise skip LLM translation.
    """
    path = tx_dir / "partial.json"
    if not path.exists():
        return False
    path.unlink()
    logger.info("  Cleared stale partial.json (re-compose)")
    return True


def load_partial_cues(tx_dir: Path, config: SubtitleConfig) -> list[SubtitleCue]:
    path = tx_dir / "partial.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        SubtitleCue(
            cue_id=c["cue_id"],
            unit_id=c["unit_id"],
            start=c["start"],
            end=c["end"],
            text=c["text"],
            lang=c.get("lang", config.target_lang),
        )
        for c in raw
    ]


def _translate_batch(
    client: OpenAIClient,
    system_prompt: str,
    segments: list[Segment],
    all_segments: list[Segment],
    batch_idx: int,
    config: SubtitleConfig,
) -> tuple[list[SubtitleCue], dict]:
    payload = _build_payload(segments, all_segments, batch_idx, config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    import time

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            response, usage = client.chat(messages, temperature=config.llm_temperature)
            return _parse_response(response, segments, config, all_segments), usage
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"    Retry {attempt + 1}/{max_retries}: JSON parse error in batch {batch_idx}")
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = 2**attempt
                logger.warning(
                    f"    Retry {attempt + 1}/{max_retries}: {type(e).__name__} in batch {batch_idx}, waiting {delay}s"
                )
                time.sleep(delay)
    raise last_error  # type: ignore[misc]


def _parse_split_part(unit_id: str) -> tuple[str, int] | None:
    """Return (split_group_id, part_index) for units like ``mu0059_u0060_0``."""
    match = _SPLIT_PART_RE.match(unit_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _split_group_part_counts(segments: list[Segment]) -> dict[str, int]:
    """Map split group id to number of parts (max index + 1)."""
    max_index: dict[str, int] = {}
    for segment in segments:
        parsed = _parse_split_part(segment.unit_id)
        if parsed is None:
            continue
        group_id, part_index = parsed
        max_index[group_id] = max(max_index.get(group_id, 0), part_index + 1)
    return max_index


def _is_last_split_part(unit_id: str, part_counts: dict[str, int]) -> bool | None:
    """Return whether *unit_id* is the last part of its split group, or None if not split."""
    parsed = _parse_split_part(unit_id)
    if parsed is None:
        return None
    group_id, part_index = parsed
    count = part_counts.get(group_id, part_index + 1)
    return part_index >= count - 1


def _source_ends_sentence(source_text: str) -> bool:
    """True when English source ends with sentence-final punctuation."""
    stripped = source_text.rstrip()
    return bool(stripped) and stripped[-1] in _EN_SENTENCE_END


def _split_payload_fields(unit_id: str, part_counts: dict[str, int]) -> dict:
    """Optional split-group metadata for translation payloads."""
    parsed = _parse_split_part(unit_id)
    if parsed is None:
        return {}
    group_id, part_index = parsed
    part_count = part_counts.get(group_id, part_index + 1)
    return {
        "split_group": group_id,
        "part_index": part_index,
        "part_count": part_count,
        "is_continuation": part_index > 0,
    }


def _unit_payload_entry(segment: Segment, part_counts: dict[str, int], *, translate: bool = True) -> dict:
    """Build one translation payload item for a segment."""
    entry: dict = {
        "unit_id": segment.unit_id,
        "source_text": segment.source_text,
        "speaker": segment.speaker,
    }
    if translate:
        entry["duration"] = round(segment.end - segment.start, 1)
        entry["max_chars_hint"] = int((segment.end - segment.start) * 8)
    else:
        entry["translate"] = False
    entry.update(_split_payload_fields(segment.unit_id, part_counts))
    return entry


def _chunk_pending_segments(pending: list[Segment], chunk_size: int) -> list[list[Segment]]:
    """Chunk pending segments without splitting a ``split_group`` across batches when feasible."""
    if not pending:
        return []
    if len(pending) <= chunk_size:
        return [pending]

    chunks: list[list[Segment]] = []
    start = 0
    while start < len(pending):
        end = min(start + chunk_size, len(pending))
        if end < len(pending):
            end = _adjust_chunk_end(pending, start, end, chunk_size)
        chunks.append(pending[start:end])
        start = end
    return chunks


def _adjust_chunk_end(pending: list[Segment], start: int, end: int, chunk_size: int) -> int:
    """Extend or shrink *end* so split siblings stay in one batch when reasonable."""
    if end >= len(pending):
        return end

    left = pending[end - 1]
    right = pending[end]
    left_part = _parse_split_part(left.unit_id)
    right_part = _parse_split_part(right.unit_id)
    if not left_part or not right_part or left_part[0] != right_part[0]:
        return end

    group_id = left_part[0]
    group_end = end
    while group_end < len(pending):
        part = _parse_split_part(pending[group_end].unit_id)
        if part and part[0] == group_id:
            group_end += 1
        else:
            break

    if group_end - start <= chunk_size + 16:
        return group_end

    group_start = end - 1
    while group_start > start:
        part = _parse_split_part(pending[group_start - 1].unit_id)
        if part and part[0] == group_id:
            group_start -= 1
        else:
            break
    return max(start + 1, group_start)


def _build_payload(
    segments: list[Segment], all_segments: list[Segment], batch_idx: int, config: SubtitleConfig
) -> dict:
    """Build translation payload with context units and split-group metadata."""
    part_counts = _split_group_part_counts(all_segments)
    ctx_start = max(0, batch_idx - 2)
    ctx_end = min(len(all_segments), batch_idx + len(segments) + 2)
    context_items = [
        _unit_payload_entry(all_segments[i], part_counts, translate=False) for i in range(ctx_start, batch_idx)
    ]

    unit_items = [_unit_payload_entry(segment, part_counts, translate=True) for segment in segments]

    context_items.extend(
        _unit_payload_entry(all_segments[i], part_counts, translate=False)
        for i in range(batch_idx + len(segments), ctx_end)
    )

    return {
        **_translation_context_fields(config),
        "units": context_items + unit_items,
    }


def _parse_response(
    response: str,
    source_segments: list[Segment],
    config: SubtitleConfig,
    all_segments: list[Segment] | None = None,
) -> list[SubtitleCue]:
    """Parse LLM response into SubtitleCue list.

    Expected format — one translation per segment:
      [{"unit_id": "u001", "text": "..."}]

    Each source segment produces exactly one SubtitleCue.
    Timestamps come directly from the source segment (one-to-one mapping).
    Word-level timestamps are passed through for downstream alignment.
    """
    json_match = re.search(r"\[([\s\S]*)\]", response)
    if json_match:
        data = json.loads(json_match.group(0))
    else:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return []

    segment_map: dict[str, Segment] = {s.unit_id: s for s in source_segments}
    part_counts = _split_group_part_counts(all_segments if all_segments is not None else source_segments)

    cues: list[SubtitleCue] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        uid = item.get("unit_id", "")
        seg = segment_map.get(uid)
        if seg is None:
            continue

        # Take the first non-empty text: prefer `text`, fallback to joined chunks
        # (for backward compatibility with models that still emit chunks).
        text = item.get("text", "") or ""
        if not text:
            chunks = item.get("chunks") or []
            if chunks:
                text = "".join(chunks)
        text = text.replace("\\n", "\n")
        text = _normalize_punctuation(
            text,
            config.target_lang,
            is_last_split_part=_is_last_split_part(uid, part_counts),
            source_ends_sentence=_source_ends_sentence(seg.source_text),
        )

        cues.append(
            SubtitleCue(
                cue_id=f"{config.target_lang}_{len(cues):04d}",
                unit_id=uid,
                start=seg.start,
                end=seg.end,
                text=text,
                lang=config.target_lang,
                words=list(seg.words),
            )
        )

    return cues


def translate_missing(
    segments: list[Segment],
    missing_ids: set[str],
    config: SubtitleConfig,
) -> tuple[list[SubtitleCue], dict]:
    """Retranslate specific missing segments with context.

    For each missing segment, includes 2 neighbours before/after as
    context (marked ``translate: false``), exactly like normal translation.
    Returns only the cues for missing unit_ids and their token usage.
    """
    if not config.llm_api_key or not missing_ids:
        return [], {}

    logger.info(f"    Retranslating {len(missing_ids)} missing: {', '.join(sorted(missing_ids)[:8])}")

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )
    system_prompt = _render_translate_prompt(config)

    all_cues: list[SubtitleCue] = []
    total_usage: dict[str, int] = {}

    # Group by individual missing segment with context
    for i, s in enumerate(segments):
        if s.unit_id not in missing_ids:
            continue

        # Build context: 2 before + this + 2 after
        ctx_start = max(0, i - 2)
        ctx_end = min(len(segments), i + 3)
        batch = segments[ctx_start:ctx_end]
        part_counts = _split_group_part_counts(segments)

        payload_items = []
        for bs in batch:
            entry = _unit_payload_entry(bs, part_counts, translate=bs.unit_id == s.unit_id)
            payload_items.append(entry)

        payload = {
            **_translation_context_fields(config),
            "units": payload_items,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            response, usage = client.chat(messages, temperature=config.llm_temperature)
            cues = _parse_response(response, batch, config, segments)
            # Keep only the missing unit's cue
            cues = [c for c in cues if c.unit_id == s.unit_id]
            all_cues.extend(cues)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] = total_usage.get(k, 0) + usage.get(k, 0)
        except Exception as e:
            logger.warning(f"      Retry failed for {s.unit_id}: {e}")

    # Reassign cue IDs
    for i, c in enumerate(all_cues):
        c.cue_id = f"{config.target_lang}_retry_{i:04d}"

    return all_cues, total_usage


def _normalize_punctuation(
    text: str,
    lang: str,
    *,
    is_last_split_part: bool | None = None,
    source_ends_sentence: bool = False,
) -> str:
    """Ensure Chinese text ends with proper punctuation.

    For non-final split parts whose English source continues mid-sentence,
    do not force a full stop — that breaks cross-segment readability.
    """
    if lang != "zh" or not text:
        return text

    stripped = text.rstrip()
    if not stripped:
        return text

    last = stripped[-1]
    mid_split = is_last_split_part is False and not source_ends_sentence

    if last in CJK_CLAUSE_PUNCT:
        if mid_split:
            return stripped
        return stripped[:-1] + "。"

    if last not in SENTENCE_ENDS:
        if mid_split:
            return stripped
        return stripped + "。"

    return text
