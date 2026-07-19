"""LLM translation — chunked, parallel, with retry and JSON parsing."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from light_models import Segment, SubtitleCue
from light_models.punctuation import CJK_CLAUSE_PUNCT, SENTENCE_ENDS

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient, client_from_config
from ...llm.json_extract import extract_json_array
from ...llm.parallel import run_parallel_with_warmup
from ...llm.prompts import render_prompt
from ...llm.retry import chat_with_retry
from ...usage.tracker import merge_token_usage, pick_usage_fields
from .align_check import check_batch_alignment, format_align_failures, render_align_check_system_prompt
from .checkpoint import _save_partial, load_partial
from .chunking import _chunk_pending_segments
from .protocol import _is_last_split_part, _parse_split_part, _source_ends_sentence, _split_group_part_counts

CHUNK_SIZE = 100
MAX_WORKERS = 4


def covered_unit_ids(cues: list[SubtitleCue]) -> set[str]:
    """unit_ids covered by a cue list, including units absorbed via ``merged_from``."""
    ids = {c.unit_id for c in cues}
    for c in cues:
        ids.update(c.merged_from)
    return ids


def _render_translate_prompt(
    config: SubtitleConfig,
    *,
    glossary: dict | None = None,
    content_summary: dict | None = None,
) -> str:
    """Build system prompt with glossary and content summary.

    *glossary* / *content_summary* override the config fields when given
    (the pipeline passes the merged values from PipelineState).
    """
    return render_prompt(
        "translate.j2",
        target_lang=config.target_lang,
        glossary=config.glossary if glossary is None else glossary,
        content_summary=config.content_summary if content_summary is None else content_summary,
    )


def _translation_context_fields(config: SubtitleConfig) -> dict:
    """Per-batch user payload fields (glossary/summary live in system prompt)."""
    return {"target_lang": config.target_lang}


def run(
    segments: list[Segment],
    config: SubtitleConfig,
    tx_dir: Path | None = None,
    *,
    glossary: dict | None = None,
    content_summary: dict | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[SubtitleCue], dict | None]:
    """Return (translated_cues, usage_dict).

    When *tx_dir* is set, saves ``partial.json`` after each batch for resume.
    *progress*, when given, receives ``(fraction, message)`` after each batch.
    """
    if not config.llm_api_key:
        return [], None

    client = client_from_config(config)

    system_prompt = _render_translate_prompt(config, glossary=glossary, content_summary=content_summary)
    align_system_prompt = render_align_check_system_prompt(config)
    existing_cues: list[SubtitleCue] = []
    if tx_dir is not None:
        existing_cues = load_partial(tx_dir, config, segments)

    covered = covered_unit_ids(existing_cues)
    pending = [s for s in segments if s.unit_id not in covered]

    if not pending and existing_cues:
        return _finalize_translated_cues(existing_cues, config), None

    existing = {c.unit_id: c for c in existing_cues}
    batch_chunks = _chunk_pending_segments(pending, CHUNK_SIZE)
    if len(batch_chunks) == 1:
        chunk = batch_chunks[0]
        abs_idx = segments.index(chunk[0]) if chunk else 0
        cues, usage, batch_breakdown = _translate_batch(
            client,
            system_prompt,
            chunk,
            segments,
            abs_idx,
            config,
            align_system_prompt=align_system_prompt,
        )
        if progress is not None:
            progress(1.0, "翻译中... 1/1")
        for c in cues:
            existing[c.unit_id] = c
        ordered_1_1 = _order_cues(segments, existing)
        if tx_dir is not None:
            _save_partial(tx_dir, ordered_1_1, segments)
        ordered = _finalize_translated_cues(ordered_1_1, config)
        if batch_breakdown:
            usage = usage or {}
            usage["breakdown"] = batch_breakdown
        return ordered, usage

    total_usage: dict[str, int] = {}
    usage_breakdown: dict[str, dict] = {}

    def _run_chunk(chunk: list[Segment]) -> tuple[list[SubtitleCue], dict, dict[str, dict]]:
        abs_idx = segments.index(chunk[0])
        return _translate_batch(
            client,
            system_prompt,
            chunk,
            segments,
            abs_idx,
            config,
            align_system_prompt=align_system_prompt,
        )

    tasks = [
        (lambda c=chunk: logger.run_with_file_logger(_run_chunk, c)(), segments.index(chunk[0]))
        for chunk in batch_chunks
        if chunk
    ]

    n_batches = len(tasks)
    completed = 0

    def _on_batch_done() -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed / n_batches, f"翻译中... {completed}/{n_batches}")

    results = run_parallel_with_warmup(tasks, max_workers=MAX_WORKERS, on_complete=_on_batch_done)

    for idx in sorted(results):
        cues, usage, batch_breakdown = results[idx]
        for c in cues:
            existing[c.unit_id] = c
        merge_token_usage(total_usage, usage)
        for step_id, step_usage in batch_breakdown.items():
            merge_token_usage(usage_breakdown.setdefault(step_id, {}), step_usage)
        ordered_1_1 = _order_cues(segments, existing)
        if tx_dir is not None:
            _save_partial(tx_dir, ordered_1_1, segments)

    ordered_1_1 = _order_cues(segments, existing)
    ordered = _finalize_translated_cues(ordered_1_1, config)
    if usage_breakdown:
        total_usage["breakdown"] = usage_breakdown
    return ordered, total_usage or None


def _order_cues(segments: list[Segment], by_id: dict[str, SubtitleCue]) -> list[SubtitleCue]:
    return [by_id[s.unit_id] for s in segments if s.unit_id in by_id]


def _finalize_translated_cues(cues_1_1: list[SubtitleCue], config: SubtitleConfig) -> list[SubtitleCue]:
    """Assign sequential cue_ids (cue boundaries are final once planned)."""
    for i, c in enumerate(cues_1_1):
        c.cue_id = f"{config.target_lang}_{i:04d}"
    return cues_1_1


class _AlignRejected(Exception):
    """Alignment check failed on an otherwise successful batch translation.

    Carries the round's outputs so the caller can fall back to the last
    result after retries are exhausted.
    """

    def __init__(
        self,
        detail: str,
        cues: list[SubtitleCue],
        usage: dict,
        translate_usage: dict,
        align_usage: dict,
    ):
        super().__init__(detail)
        self.detail = detail
        self.cues = cues
        self.usage = usage
        self.translate_usage = translate_usage
        self.align_usage = align_usage


def _translate_batch(
    client: OpenAIClient,
    system_prompt: str,
    segments: list[Segment],
    all_segments: list[Segment],
    batch_idx: int,
    config: SubtitleConfig,
    *,
    align_system_prompt: str | None = None,
) -> tuple[list[SubtitleCue], dict, dict[str, dict]]:
    payload = _build_payload(segments, all_segments, batch_idx, config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    max_retries = 3

    def _attempt() -> tuple[list[SubtitleCue], dict, dict[str, dict]]:
        response, usage = client.chat(messages, temperature=config.llm_temperature)
        translate_usage = pick_usage_fields(usage)
        cues, parsed_texts = _parse_response(response, segments, config, all_segments)

        aligned, failures, align_usage = check_batch_alignment(
            client,
            segments,
            parsed_texts,
            all_segments,
            batch_idx,
            config,
            system_prompt=align_system_prompt,
        )
        merge_token_usage(usage, align_usage)
        if not aligned:
            raise _AlignRejected(format_align_failures(failures), cues, usage, translate_usage, align_usage)

        breakdown = {
            "translate.translate": translate_usage,
            "translate.align_check": pick_usage_fields(align_usage),
        }
        return cues, usage, breakdown

    def _on_retry(attempt: int, exc: BaseException) -> float | None:
        if isinstance(exc, _AlignRejected):
            logger.warning(
                f"    Align check failed batch@{batch_idx} (attempt {attempt + 1}/{max_retries}): {exc.detail}"
            )
            return None
        if isinstance(exc, (json.JSONDecodeError, ValueError)):
            logger.warning(f"    Retry {attempt + 1}/{max_retries}: {type(exc).__name__} in batch {batch_idx}: {exc}")
            return None
        delay = 2**attempt
        logger.warning(
            f"    Retry {attempt + 1}/{max_retries}: {type(exc).__name__} in batch {batch_idx}, waiting {delay}s"
        )
        return delay

    try:
        return chat_with_retry(_attempt, max_retries=max_retries, on_retry=_on_retry)
    except _AlignRejected as e:
        logger.warning(f"    Align check failed batch@{batch_idx} (attempt {max_retries}/{max_retries}): {e.detail}")
        logger.warning(f"    Align check failed batch@{batch_idx} after {max_retries} attempts; using last result")
        breakdown = {
            "translate.translate": e.translate_usage,
            "translate.align_check": pick_usage_fields(e.align_usage),
        }
        return e.cues, e.usage, breakdown


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
    json_fragment = extract_json_array(response)
    if json_fragment is not None:
        data = json.loads(json_fragment)
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
        text = normalize_punctuation(
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
    *,
    glossary: dict | None = None,
    content_summary: dict | None = None,
) -> tuple[list[SubtitleCue], dict]:
    """Retranslate specific missing segments with context.

    For each missing segment, includes 2 neighbours before/after as
    context (marked ``translate: false``), exactly like normal translation.
    Returns only the cues for missing unit_ids and their token usage.
    """
    if not config.llm_api_key or not missing_ids:
        return [], {}

    logger.info(f"    Retranslating {len(missing_ids)} missing: {', '.join(sorted(missing_ids)[:8])}")

    client = client_from_config(config)
    system_prompt = _render_translate_prompt(config, glossary=glossary, content_summary=content_summary)

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
            "target_lang": config.target_lang,
            "units": payload_items,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            response, usage = client.chat(messages, temperature=config.llm_temperature)
            cues, _parsed_texts = _parse_response(response, [s], config, segments)
            all_cues.extend(cues)
            merge_token_usage(total_usage, usage)
        except Exception as e:
            logger.warning(f"      Retry failed for {s.unit_id}: {e}")

    # Reassign cue IDs
    for i, c in enumerate(all_cues):
        c.cue_id = f"{config.target_lang}_retry_{i:04d}"

    return all_cues, total_usage


def normalize_punctuation(
    text: str,
    lang: str,
    *,
    is_last_split_part: bool | None = None,
    source_ends_sentence: bool = False,
    punctuate_blank: bool = False,
) -> str:
    """Ensure Chinese text ends with proper punctuation.

    For non-final split parts whose English source continues mid-sentence,
    do not force a full stop — that breaks cross-segment readability.

    ``punctuate_blank`` reproduces the legacy refine behavior of turning
    whitespace-only input into a lone ``。``; the default (translate path)
    returns such input unchanged.
    """
    if lang != "zh" or not text:
        return text

    stripped = text.rstrip()
    if not stripped:
        return stripped + "。" if punctuate_blank else text

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
