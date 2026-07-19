"""LLM batch alignment check — sample first/middle/last units after translation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from light_models import Segment

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient
from ...llm.json_extract import extract_json_array
from ...llm.prompts import render_prompt
from ...llm.retry import chat_with_retry

_ALIGN_CHECK_RETRIES = 3
_CONFIDENCE_THRESHOLD = 0.90
_CONTEXT_WINDOW = 2
_LOG_SNIP_LIMIT = 48


@dataclass(frozen=True)
class AlignFailure:
    """One high-confidence alignment failure on a sampled unit."""

    unit_id: str
    reason: str
    confidence: float
    source: str
    translation: str


def _snip(text: str, limit: int = _LOG_SNIP_LIMIT) -> str:
    flat = text.replace("\n", " ").strip()
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def format_align_failures(failures: list[AlignFailure]) -> str:
    """Format failures for logs — include the source/translation actually checked."""
    parts: list[str] = []
    for f in failures:
        parts.append(
            f"{f.unit_id} (conf={f.confidence:.2f}): {f.reason} "
            f"| src='{_snip(f.source)}' trans='{_snip(f.translation)}'"
        )
    return "; ".join(parts)


def _alignment_sample_indices(n: int) -> list[int]:
    """Return deduplicated first / middle / last indices for a batch of size *n*."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    return [0, n // 2, n - 1]


def render_align_check_system_prompt(config: SubtitleConfig) -> str:
    """Build align-check system prompt (cache-friendly when reused across batches)."""
    return render_prompt(
        "translate_align_check.j2",
        target_lang=config.target_lang or "unknown",
    )


def _render_align_check_prompt(config: SubtitleConfig) -> str:
    return render_align_check_system_prompt(config)


def _in_current_batch(global_idx: int, batch_idx: int, batch_len: int) -> bool:
    return batch_idx <= global_idx < batch_idx + batch_len


def _neighbor_entry(
    global_idx: int,
    all_segments: list[Segment],
    batch_idx: int,
    batch_len: int,
    parsed_texts: dict[int, str],
) -> dict:
    entry: dict = {"source": all_segments[global_idx].source_text}
    if _in_current_batch(global_idx, batch_idx, batch_len):
        entry["translation"] = parsed_texts.get(global_idx - batch_idx, "")
    return entry


def _build_check_entry(
    global_idx: int,
    all_segments: list[Segment],
    batch_idx: int,
    batch_len: int,
    parsed_texts: dict[int, str],
) -> dict:
    """Build one alignment check: target pair + before/after context."""
    ctx_start = max(0, global_idx - _CONTEXT_WINDOW)
    ctx_end = min(len(all_segments), global_idx + _CONTEXT_WINDOW + 1)

    before = [
        _neighbor_entry(j, all_segments, batch_idx, batch_len, parsed_texts) for j in range(ctx_start, global_idx)
    ]
    after = [
        _neighbor_entry(j, all_segments, batch_idx, batch_len, parsed_texts) for j in range(global_idx + 1, ctx_end)
    ]

    return {
        "source": all_segments[global_idx].source_text,
        "translation": parsed_texts.get(global_idx - batch_idx, ""),
        "before": before,
        "after": after,
    }


def _build_align_payload(
    segments: list[Segment],
    parsed_texts: dict[int, str],
    sample_indices: list[int],
    all_segments: list[Segment],
    batch_idx: int,
    config: SubtitleConfig,
) -> dict:
    batch_len = len(segments)
    checks = [
        _build_check_entry(
            batch_idx + idx,
            all_segments,
            batch_idx,
            batch_len,
            parsed_texts,
        )
        for idx in sample_indices
    ]
    return {
        "target_lang": config.target_lang,
        "checks": checks,
    }


def _is_actionable_misaligned(item: dict) -> bool:
    if item.get("aligned", True):
        return False
    try:
        conf = float(item.get("confidence", 0))
    except (TypeError, ValueError):
        return False
    return conf >= _CONFIDENCE_THRESHOLD


def _parse_align_response(
    response: str,
    expected_len: int,
    sample_indices: list[int],
    segments: list[Segment],
    parsed_texts: dict[int, str],
) -> tuple[bool, list[AlignFailure]]:
    """Parse alignment LLM response; conservative pass on parse/length errors."""
    response = response.strip()
    json_fragment = extract_json_array(response)
    if json_fragment is not None:
        data = json.loads(json_fragment)
    else:
        data = json.loads(response)

    if not isinstance(data, list) or len(data) != expected_len:
        logger.warning(
            f"    Align check: expected {expected_len} results, got "
            f"{len(data) if isinstance(data, list) else 'non-list'}; treating as aligned"
        )
        return True, []

    failures: list[AlignFailure] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if not _is_actionable_misaligned(item):
            continue
        seg_idx = sample_indices[i]
        seg = segments[seg_idx]
        reason = str(item.get("reason", "") or "")
        confidence = float(item["confidence"])
        failures.append(
            AlignFailure(
                unit_id=seg.unit_id,
                reason=reason,
                confidence=confidence,
                source=seg.source_text,
                translation=parsed_texts.get(seg_idx, ""),
            )
        )

    return len(failures) == 0, failures


def check_batch_alignment(
    client: OpenAIClient,
    segments: list[Segment],
    parsed_texts: dict[int, str],
    all_segments: list[Segment],
    batch_idx: int,
    config: SubtitleConfig,
    *,
    system_prompt: str | None = None,
) -> tuple[bool, list[AlignFailure], dict]:
    """Check first/middle/last sampled units for source/translation alignment.

    Each check includes the target source/translation pair plus ``before`` /
    ``after`` neighbor context from *all_segments*. In-batch neighbors include
    translations; cross-batch neighbors are source-only.

    Returns ``(aligned, failures, usage)`` where *failures* are
    :class:`AlignFailure` records for high-confidence misalignments only.
    """
    if not segments:
        return True, [], {}

    sample_indices = _alignment_sample_indices(len(segments))
    payload = _build_align_payload(
        segments,
        parsed_texts,
        sample_indices,
        all_segments,
        batch_idx,
        config,
    )
    system_prompt = system_prompt or _render_align_check_prompt(config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    def _attempt() -> tuple[bool, list[AlignFailure], dict]:
        response, usage = client.chat(messages, temperature=0.0)
        aligned, failures = _parse_align_response(
            response,
            len(sample_indices),
            sample_indices,
            segments,
            parsed_texts,
        )
        return aligned, failures, usage

    def _on_retry(attempt: int, exc: BaseException) -> None:
        logger.warning(f"    Align check retry {attempt + 1}/{_ALIGN_CHECK_RETRIES}: {exc}")

    try:
        return chat_with_retry(
            _attempt,
            max_retries=_ALIGN_CHECK_RETRIES,
            retry_exceptions=(json.JSONDecodeError, ValueError),
            on_retry=_on_retry,
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"    Align check failed to parse, treating as aligned: {e}")
        return True, [], {}
