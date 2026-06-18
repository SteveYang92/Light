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
from .merge_apply import apply_display_merges, covered_unit_ids
from .merge_review import MergeHint, log_merge_hints, review_merge_hints

CHUNK_SIZE = 100
MAX_WORKERS = 4
_PARTIAL_VERSION = 1
_SPLIT_PART_RE = re.compile(r"^(mu\d+_u\d+(?:_\d+)*)_(\d+)$")
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
    existing_cues: list[SubtitleCue] = []
    hint_records: list[dict[str, str]] = []
    if tx_dir is not None:
        existing_cues, hint_records = load_partial(tx_dir, config)

    covered = covered_unit_ids(existing_cues)
    pending = [s for s in segments if s.unit_id not in covered]
    stored_hints = _hints_from_records(segments, hint_records)

    if not pending and existing_cues:
        ordered = _finalize_translated_cues(
            existing_cues,
            stored_hints,
            config,
        )
        return ordered, None

    existing = {c.unit_id: c for c in existing_cues}
    batch_chunks = _chunk_pending_segments(pending, CHUNK_SIZE)
    if len(batch_chunks) == 1:
        chunk = batch_chunks[0]
        abs_idx = segments.index(chunk[0]) if chunk else 0
        cues, usage, hints = _translate_batch(client, system_prompt, chunk, segments, abs_idx, config)
        for c in cues:
            existing[c.unit_id] = c
        ordered_1_1 = _order_cues(segments, existing)
        hint_records = _dedupe_hint_records(hint_records + _hint_records_from_hints(hints))
        if tx_dir is not None:
            _save_partial(tx_dir, ordered_1_1, hint_records)
        ordered = _finalize_translated_cues(ordered_1_1, _hints_from_records(segments, hint_records), config)
        return ordered, usage

    total_usage: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch_chunks))) as executor:
        futures = {
            executor.submit(
                logger.run_with_file_logger(
                    _translate_batch,
                    client,
                    system_prompt,
                    chunk,
                    segments,
                    segments.index(chunk[0]),
                    config,
                ),
            ): segments.index(chunk[0])
            for chunk in batch_chunks
            if chunk
        }
        results: dict[int, tuple[list[SubtitleCue], dict, list[MergeHint]]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

        for idx in sorted(results):
            cues, usage, hints = results[idx]
            hint_records = _dedupe_hint_records(hint_records + _hint_records_from_hints(hints))
            for c in cues:
                existing[c.unit_id] = c
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] = total_usage.get(k, 0) + usage.get(k, 0)
            ordered_1_1 = _order_cues(segments, existing)
            if tx_dir is not None:
                _save_partial(tx_dir, ordered_1_1, hint_records)

    ordered_1_1 = _order_cues(segments, existing)
    ordered = _finalize_translated_cues(
        ordered_1_1,
        _hints_from_records(segments, hint_records),
        config,
    )
    return ordered, total_usage or None


def _order_cues(segments: list[Segment], by_id: dict[str, SubtitleCue]) -> list[SubtitleCue]:
    return [by_id[s.unit_id] for s in segments if s.unit_id in by_id]


def _finalize_translated_cues(
    cues_1_1: list[SubtitleCue],
    hints: list[MergeHint],
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    """Apply display merges and assign sequential cue_ids."""
    ordered = cues_1_1
    if config.merge_hints_apply and hints:
        ordered = apply_display_merges(cues_1_1, hints, config)
    for i, c in enumerate(ordered):
        c.cue_id = f"{config.target_lang}_{i:04d}"
    return ordered


def _hint_records_from_hints(hints: list[MergeHint]) -> list[dict[str, str]]:
    return [{"curr_unit_id": curr.unit_id, "next_unit_id": nxt.unit_id} for curr, nxt, _, _ in hints]


def _dedupe_hint_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for record in records:
        key = (record["curr_unit_id"], record["next_unit_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _hints_from_records(segments: list[Segment], records: list[dict[str, str]]) -> list[MergeHint]:
    seg_by_id = {s.unit_id: s for s in segments}
    hints: list[MergeHint] = []
    for record in records:
        curr = seg_by_id.get(record["curr_unit_id"])
        nxt = seg_by_id.get(record["next_unit_id"])
        if curr is None or nxt is None:
            continue
        hints.append((curr, nxt, "", ""))
    return hints


def _cue_dict_from_partial(cue: SubtitleCue) -> dict:
    return {
        "cue_id": cue.cue_id,
        "unit_id": cue.unit_id,
        "start": cue.start,
        "end": cue.end,
        "text": cue.text,
        "lang": cue.lang,
    }


def _save_partial(tx_dir: Path, cues: list[SubtitleCue], hint_records: list[dict[str, str]]) -> None:
    """Persist 1:1 translation checkpoint with merge hints."""
    tx_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "version": _PARTIAL_VERSION,
        "cues": [_cue_dict_from_partial(c) for c in cues],
        "merge_hints": hint_records,
    }
    (tx_dir / "partial.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_partial_cues(tx_dir: Path, cues: list[SubtitleCue]) -> None:
    """Backward-compatible alias — saves 1:1 cues with empty merge hints."""
    _save_partial(tx_dir, cues, [])


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


def load_partial(tx_dir: Path, config: SubtitleConfig) -> tuple[list[SubtitleCue], list[dict[str, str]]]:
    """Load 1:1 partial cues and persisted merge-hint records."""
    path = tx_dir / "partial.json"
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        if any(c.get("merged_from") for c in raw if isinstance(c, dict)):
            logger.warning("  Legacy partial.json contains merged cues; delete partial.json for a clean resume.")
        return [_cue_from_partial_dict(c, config) for c in raw], []

    if isinstance(raw, dict):
        cue_items = raw.get("cues", [])
        hint_records = raw.get("merge_hints", [])
        return [_cue_from_partial_dict(c, config) for c in cue_items], list(hint_records)

    return [], []


def _cue_from_partial_dict(data: dict, config: SubtitleConfig) -> SubtitleCue:
    return SubtitleCue(
        cue_id=data["cue_id"],
        unit_id=data["unit_id"],
        start=data["start"],
        end=data["end"],
        text=data["text"],
        lang=data.get("lang", config.target_lang),
        merged_from=data.get("merged_from", []),
    )


def load_partial_cues(tx_dir: Path, config: SubtitleConfig) -> list[SubtitleCue]:
    cues, _hints = load_partial(tx_dir, config)
    return cues


def _translate_batch(
    client: OpenAIClient,
    system_prompt: str,
    segments: list[Segment],
    all_segments: list[Segment],
    batch_idx: int,
    config: SubtitleConfig,
) -> tuple[list[SubtitleCue], dict, list[MergeHint]]:
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
            cues, parsed_texts = _parse_response(response, segments, config, all_segments)
            merge_hints = review_merge_hints(client, segments, parsed_texts, config)
            log_merge_hints(merge_hints)
            return cues, usage, merge_hints
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"    Retry {attempt + 1}/{max_retries}: {type(e).__name__} in batch {batch_idx}: {e}")
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
    """Return ``(split_group_id, part_index)`` for units like ``mu0059_u0060_0`` or ``mu0187_u0190_0_1``."""
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


def _unit_payload_entry(
    segment: Segment,
    part_counts: dict[str, int],
    *,
    translate: bool = True,
    batch_index: int | None = None,
) -> dict:
    """Build one translation payload item for a segment."""
    entry: dict = {
        "unit_id": segment.unit_id,
        "source_text": segment.source_text,
        "speaker": segment.speaker,
    }
    if translate:
        entry["duration"] = round(segment.end - segment.start, 1)
        entry["max_chars_hint"] = int((segment.end - segment.start) * 8)
        if batch_index is not None:
            entry["batch_index"] = batch_index
    else:
        entry["translate"] = False
    entry.update(_split_payload_fields(segment.unit_id, part_counts))
    return entry


def _split_group_extent(pending: list[Segment], index: int) -> tuple[int, int]:
    """Return ``[start, end)`` indices of the split_group containing ``pending[index]``."""
    parsed = _parse_split_part(pending[index].unit_id)
    if parsed is None:
        return index, index + 1
    group_id = parsed[0]
    start = index
    while start > 0:
        prev = _parse_split_part(pending[start - 1].unit_id)
        if prev and prev[0] == group_id:
            start -= 1
        else:
            break
    end = index + 1
    while end < len(pending):
        nxt = _parse_split_part(pending[end].unit_id)
        if nxt and nxt[0] == group_id:
            end += 1
        else:
            break
    return start, end


def _chunk_pending_segments(pending: list[Segment], chunk_size: int) -> list[list[Segment]]:
    """Chunk pending segments; never split a ``split_group`` across batches.

    A split_group may occupy a batch larger than *chunk_size* when the whole
    group does not fit in the remaining space of the current batch.
    """
    if not pending:
        return []

    chunks: list[list[Segment]] = []
    i = 0
    while i < len(pending):
        chunk: list[Segment] = []
        while i < len(pending):
            g_start, g_end = _split_group_extent(pending, i)
            g_len = g_end - g_start
            if g_len > 1:
                group_slice = pending[g_start:g_end]
                if chunk and len(chunk) + g_len > chunk_size:
                    break
                if not chunk and g_len > chunk_size:
                    chunks.append(group_slice)
                    i = g_end
                    chunk = []
                    break
                chunk.extend(group_slice)
                i = g_end
                continue
            if len(chunk) >= chunk_size:
                break
            chunk.append(pending[i])
            i += 1
        if chunk:
            chunks.append(chunk)
    return chunks


def _adjust_chunk_end(pending: list[Segment], start: int, end: int, chunk_size: int) -> int:
    """Extend *end* so a split_group at the boundary stays in one batch (may exceed *chunk_size*)."""
    if end >= len(pending):
        return end

    g_start, g_end = _split_group_extent(pending, end - 1)
    if g_end <= end:
        g_start, g_end = _split_group_extent(pending, end)
    if g_end - g_start <= 1:
        return end
    if g_start < start:
        return end
    return g_end


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

    unit_items = [
        _unit_payload_entry(segment, part_counts, translate=True, batch_index=idx)
        for idx, segment in enumerate(segments)
    ]

    context_items.extend(
        _unit_payload_entry(all_segments[i], part_counts, translate=False)
        for i in range(batch_idx + len(segments), ctx_end)
    )

    return {
        **_translation_context_fields(config),
        "units": context_items + unit_items,
    }


def _resolve_batch_index(
    item: dict,
    source_segments: list[Segment],
    segment_map: dict[str, Segment],
) -> int | None:
    """Map one LLM item to a batch index (preferred) or unit_id fallback."""
    if "batch_index" in item:
        try:
            return int(item["batch_index"])
        except (TypeError, ValueError):
            return None
    uid = item.get("unit_id", "")
    if uid and uid in segment_map:
        for idx, seg in enumerate(source_segments):
            if seg.unit_id == uid:
                return idx
    return None


def _parse_response(
    response: str,
    source_segments: list[Segment],
    config: SubtitleConfig,
    all_segments: list[Segment] | None = None,
) -> tuple[list[SubtitleCue], dict[int, str]]:
    """Parse LLM response into SubtitleCue list and per-index translated text.

    Expected format — one translation per batch index:
      [{"batch_index": 0, "text": "..."}]

    Raises ``ValueError`` when batch indices are incomplete or duplicated.
    """
    json_match = re.search(r"\[([\s\S]*)\]", response)
    if json_match:
        data = json.loads(json_match.group(0))
    else:
        data = json.loads(response)

    if not isinstance(data, list):
        raise ValueError("Translation response is not a JSON array")

    segment_map: dict[str, Segment] = {s.unit_id: s for s in source_segments}
    part_counts = _split_group_part_counts(all_segments if all_segments is not None else source_segments)
    expected = set(range(len(source_segments)))

    by_index: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = _resolve_batch_index(item, source_segments, segment_map)
        if idx is None or idx not in expected:
            continue
        if idx in by_index:
            raise ValueError(f"Duplicate batch_index in translation response: {idx}")
        by_index[idx] = item

    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        raise ValueError(f"Batch incomplete: missing index {missing}, unexpected {extra}")

    cues: list[SubtitleCue] = []
    parsed_texts: dict[int, str] = {}
    for idx in range(len(source_segments)):
        seg = source_segments[idx]
        item = by_index[idx]
        uid = seg.unit_id

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
        parsed_texts[idx] = text

    for idx in range(len(source_segments)):
        seg = source_segments[idx]
        item = by_index[idx]
        uid = seg.unit_id
        resp_uid = item.get("unit_id")
        if resp_uid and resp_uid != uid:
            logger.warning(f"  batch_index {idx} unit_id mismatch: expected {uid}, got {resp_uid} — using index")

        text = parsed_texts[idx]

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

    return cues, parsed_texts


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
            is_target = bs.unit_id == s.unit_id
            entry = _unit_payload_entry(
                bs,
                part_counts,
                translate=is_target,
                batch_index=0 if is_target else None,
            )
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
            cues, parsed_texts = _parse_response(response, [s], config, segments)
            log_merge_hints(review_merge_hints(client, [s], parsed_texts, config))
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
