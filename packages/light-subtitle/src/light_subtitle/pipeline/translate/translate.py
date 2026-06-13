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

    if len(pending) <= CHUNK_SIZE:
        cues, usage = _translate_batch(client, system_prompt, pending, segments, 0, config)
        merged = dict(existing)
        for c in cues:
            merged[c.unit_id] = c
        if tx_dir is not None:
            _save_partial_cues(tx_dir, _order_cues(segments, merged))
        return _order_cues(segments, merged), usage

    chunks: list[tuple[int, list[Segment]]] = []
    for batch_idx in range(0, len(pending), CHUNK_SIZE):
        chunks.append((batch_idx, pending[batch_idx : batch_idx + CHUNK_SIZE]))

    merged = dict(existing)
    total_usage: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as executor:
        futures = {
            executor.submit(_translate_batch, client, system_prompt, chunk, segments, idx, config): idx
            for idx, chunk in chunks
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
            return _parse_response(response, segments, config), usage
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


def _build_payload(
    segments: list[Segment], all_segments: list[Segment], batch_idx: int, config: SubtitleConfig
) -> dict:
    # Context: up to 2 segments before and after for style reference.
    ctx_start = max(0, batch_idx - 2)
    ctx_end = min(len(all_segments), batch_idx + len(segments) + 2)
    context_items = []
    for i in range(ctx_start, batch_idx):
        s = all_segments[i]
        context_items.append(
            {
                "unit_id": s.unit_id,
                "source_text": s.source_text,
                "speaker": s.speaker,
                "translate": False,
            }
        )

    unit_items = []
    for s in segments:
        unit_items.append(
            {
                "unit_id": s.unit_id,
                "source_text": s.source_text,
                "duration": round(s.end - s.start, 1),
                "max_chars_hint": int((s.end - s.start) * 8),
                "speaker": s.speaker,
            }
        )
    for i in range(batch_idx + len(segments), ctx_end):
        s = all_segments[i]
        context_items.append(
            {
                "unit_id": s.unit_id,
                "source_text": s.source_text,
                "speaker": s.speaker,
                "translate": False,
            }
        )

    return {
        **_translation_context_fields(config),
        "units": context_items + unit_items,
    }


def _parse_response(
    response: str,
    source_segments: list[Segment],
    config: SubtitleConfig,
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
        text = _normalize_punctuation(text, config.target_lang)

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

        # Build payload with explicit translate flags
        payload_items = []
        for bs in batch:
            entry = {
                "unit_id": bs.unit_id,
                "source_text": bs.source_text,
                "speaker": bs.speaker,
            }
            if bs.unit_id == s.unit_id:
                entry["duration"] = round(bs.end - bs.start, 1)
                entry["max_chars_hint"] = int((bs.end - bs.start) * 8)
            else:
                entry["translate"] = False
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
            cues = _parse_response(response, batch, config)
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


def _normalize_punctuation(text: str, lang: str) -> str:
    """Ensure Chinese text ends with proper punctuation.

    This is the only punctuation normalization needed now that each
    segment produces exactly one cue (no more chunk splitting).
    """
    if lang == "zh" and text:
        last = text.rstrip()[-1] if text.rstrip() else ""
        if last in CJK_CLAUSE_PUNCT:
            text = text.rstrip()[:-1] + "。"
        elif last not in SENTENCE_ENDS:
            text = text.rstrip() + "。"
    return text
