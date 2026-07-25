"""Translation retry — re-translate missing/mis-mapped units (dup + timestamp detection)."""

from __future__ import annotations

import re
from collections import defaultdict

from light_core import logger
from light_llm.client import OpenAIClient
from light_models import Segment, SubtitleCue

from ..config import TranslateConfig
from .translate import covered_unit_ids, translate_missing


def retry_missing(
    translated_cues: list[SubtitleCue],
    translation_segments: list[Segment],
    config: TranslateConfig,
    usage: dict | None,
    *,
    llm: OpenAIClient | None = None,
    glossary: dict | None = None,
    content_summary: dict | None = None,
) -> tuple[list[SubtitleCue], dict | None]:
    """Retry any translation units that failed LLM parsing.

    After retrying genuinely missing unit_ids, also detects sub-units
    that received mis-mapped translations:
      1. Duplicate text within the same merged-unit group.
      2. Timestamps that differ significantly from the compose segment
         (LLM returned correct-looking unit_id but wrong content).
    """
    MAX_RETRY = 2
    for attempt in range(MAX_RETRY):
        translated_ids = covered_unit_ids(translated_cues)
        missing_ids = {s.unit_id for s in translation_segments} - translated_ids

        # Detect duplicate translations within merged unit groups.
        dup_ids = _find_duplicate_translations(translated_cues)
        missing_ids |= dup_ids

        # Detect timestamp mismatches (LLM mapped wrong content to unit_id).
        ts_mismatch_ids = _find_timestamp_mismatches(translated_cues, translation_segments)
        missing_ids |= ts_mismatch_ids

        if not missing_ids:
            break

        if dup_ids:
            logger.warning(
                f"  ⚠ Found {len(dup_ids)} duplicate translations in merged groups, "
                f"will retry: {', '.join(sorted(dup_ids)[:8])}"
            )
        if ts_mismatch_ids:
            logger.warning(
                f"  ⚠ Found {len(ts_mismatch_ids)} units with mismatched timestamps, "
                f"will retry: {', '.join(sorted(ts_mismatch_ids)[:8])}"
            )

        logger.warning(f"  ⚠ Missing {len(missing_ids)} translations, retry {attempt + 1}/{MAX_RETRY}")

        retry_cues, retry_usage = translate_missing(
            translation_segments, missing_ids, config, llm=llm, glossary=glossary, content_summary=content_summary
        )

        if retry_cues:
            # Replace missing cues with retry results (keep cues that absorbed retried units).
            merged = [
                c for c in translated_cues if c.unit_id not in missing_ids and not set(c.merged_from) & missing_ids
            ]
            merged.extend(retry_cues)
            merged.sort(key=lambda c: c.start)
            translated_cues = merged
            if usage:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[k] = usage.get(k, 0) + retry_usage.get(k, 0)
        else:
            logger.warning(f"    ✗ Still missing {len(missing_ids)} units after retry")
            break

    return translated_cues, usage


def _find_duplicate_translations(translated_cues: list[SubtitleCue]) -> set[str]:
    """Return unit_ids of cues that share near-identical translation text
    with another cue in the same split group.

    Word-level split units (e.g. ``p0007_0``, ``p0007_1``) sometimes
    receive the same translation text for different parts when the LLM
    maps unit_ids incorrectly.  Detecting duplicates lets the retry step
    re-translate the suspect units individually.
    """
    if len(translated_cues) < 2:
        return set()

    groups: dict[str, list[SubtitleCue]] = defaultdict(list)
    for c in translated_cues:
        m = re.match(r"(p\d+)", c.unit_id)
        if m:
            groups[m.group(1)].append(c)

    suspect_ids: set[str] = set()
    for _prefix, cues in groups.items():
        if len(cues) < 2:
            continue
        # Sort by start time so the earlier (likely correct) cue is kept.
        cues.sort(key=lambda c: c.start)
        for i in range(len(cues)):
            text_i = cues[i].text.strip()
            for j in range(i + 1, len(cues)):
                text_j = cues[j].text.strip()
                if not text_i or not text_j:
                    continue
                if text_i == text_j:
                    # Exact duplicate — mark the later one.
                    suspect_ids.add(cues[j].unit_id)
                    break

    return suspect_ids


def _find_timestamp_mismatches(
    translated_cues: list[SubtitleCue],
    translation_segments: list[Segment],
    tolerance: float = 3.0,
) -> set[str]:
    """Return unit_ids of cues whose *start* time deviates from the
    corresponding compose segment by more than *tolerance* seconds.

    When the LLM maps a translation to the wrong sub-unit of a split
    merged group, the cue gets the correct unit_id but the timestamp
    of a different compose segment.  This produces large time gaps in
    the final output even though every unit_id appears to be covered.
    """
    seg_by_id = {s.unit_id: s for s in translation_segments}
    suspect_ids: set[str] = set()
    for c in translated_cues:
        seg = seg_by_id.get(c.unit_id)
        if seg is None:
            continue
        if abs(c.start - seg.start) > tolerance:
            suspect_ids.add(c.unit_id)
    return suspect_ids
