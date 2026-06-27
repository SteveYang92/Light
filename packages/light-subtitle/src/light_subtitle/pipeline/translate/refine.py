"""Translation refinement — targeted re-translation of low-quality segments.

Runs after evaluation identifies problematic translations.  Sends each
low-quality segment (with wider context and diagnostic feedback) back to
the LLM for a corrected translation.

Usage::

    from .refine import refine_translations

    refined = refine_translations(
        low_score_ids={...},
        all_cues=translated_cues,
        all_segments=translation_segments,
        quality_scores=evaluation_scores,
        config=config,
    )
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from light_models import Segment, SubtitleCue

from light_models.punctuation import CJK_CLAUSE_PUNCT, SENTENCE_ENDS

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient, merge_token_usage
from ...llm.prompts import render_prompt
from .evaluate import QualityScore

# ── Configuration ────────────────────────────────────────────────────────────

# Number of context segments on each side for refinement.
CONTEXT_WINDOW = 3

# Max segments per refinement batch (avoids overly long prompts).
REFINE_BATCH_SIZE = 10


# ── Public API ───────────────────────────────────────────────────────────────


def refine_translations(
    low_score_ids: set[str],
    all_cues: list[SubtitleCue],
    all_segments: list[Segment],
    quality_scores: list[QualityScore],
    config: SubtitleConfig,
) -> tuple[list[SubtitleCue], dict | None]:
    """Re-translate low-quality translations with diagnostic feedback.

    For each low-scoring segment, includes up to CONTEXT_WINDOW neighbouring
    segments as context.  The LLM receives the evaluation issues as specific
    instructions for what to fix.

    Returns a list of corrected ``SubtitleCue`` objects — only the ones
    that were successfully refined.  The caller merges these back into
    the main cue list.
    """
    if not config.llm_api_key or not low_score_ids:
        return [], None

    # Build lookup maps.
    cue_map: dict[str, SubtitleCue] = {c.unit_id: c for c in all_cues}
    score_map: dict[str, QualityScore] = {s.unit_id: s for s in quality_scores}

    # Collect refinement tasks: (segment_index, unit_id, source_segment).
    tasks: list[tuple[int, str, Segment]] = []
    for idx, seg in enumerate(all_segments):
        if seg.unit_id in low_score_ids and seg.unit_id in cue_map:
            tasks.append((idx, seg.unit_id, seg))

    if not tasks:
        return [], None

    # ── Group consecutive low-scoring segments for coherent refinement ──
    groups = _group_consecutive(tasks)

    logger.info(
        f"    Refining {len(tasks)} low-quality translations"
        f" in {len(groups)} group(s) (threshold < {config.quality_threshold})..."
    )

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    refined_cues: list[SubtitleCue] = []
    total_usage: dict = {}

    # Process in small batches to amortize LLM overhead.
    for batch_idx in range(0, len(groups), REFINE_BATCH_SIZE):
        batch = groups[batch_idx : batch_idx + REFINE_BATCH_SIZE]
        batch_cues, usage = _refine_batch(batch, all_segments, cue_map, score_map, client, config)
        refined_cues.extend(batch_cues)
        merge_token_usage(total_usage, usage)

    failed = len(low_score_ids) - len(refined_cues)
    if failed > 0:
        logger.warning(f"    ⚠ {failed} refinement(s) failed, keeping originals")

    return refined_cues, total_usage or None


# ── Consecutive grouping ────────────────────────────────────────────────────


def _group_consecutive(tasks: list[tuple[int, str, Segment]]) -> list[list[tuple[int, str, Segment]]]:
    """Group consecutive low-scoring segments for coherent refinement."""
    if not tasks:
        return []
    groups: list[list[tuple[int, str, Segment]]] = []
    current = [tasks[0]]
    for i in range(1, len(tasks)):
        if tasks[i][0] == tasks[i - 1][0] + 1:
            current.append(tasks[i])
        else:
            groups.append(current)
            current = [tasks[i]]
    groups.append(current)
    return groups


# ── Batch refinement ─────────────────────────────────────────────────────────


def _refine_batch(
    groups: list[list[tuple[int, str, Segment]]],
    all_segments: list[Segment],
    cue_map: dict[str, SubtitleCue],
    score_map: dict[str, QualityScore],
    client,
    config: SubtitleConfig,
) -> tuple[list[SubtitleCue], dict]:
    """Refine a batch of low-quality translation groups."""
    prompt = _build_refine_prompt(groups, all_segments, cue_map, score_map, config)

    messages = [{"role": "user", "content": prompt}]

    try:
        response, usage = client.chat(messages, temperature=0.2)
    except Exception as e:
        logger.warning(f"      ⚠ Refine batch failed: {e}")
        return [], {}

    return _parse_refine_response(response, groups, cue_map, config), usage


# ── Prompt construction ──────────────────────────────────────────────────────


def _build_refine_prompt(
    groups: list[list[tuple[int, str, Segment]]],
    all_segments: list[Segment],
    cue_map: dict[str, SubtitleCue],
    score_map: dict[str, QualityScore],
    config: SubtitleConfig,
) -> str:
    """Build a refinement prompt with grouped consecutive low-scoring segments."""
    fixes: list[dict] = []

    for group in groups:
        # Find the range of segment indices in this group.
        group_indices = [seg_idx for seg_idx, _, _ in group]
        min_idx = min(group_indices)
        max_idx = max(group_indices)

        # Context: CONTEXT_WINDOW segments before/after the group.
        ctx_start = max(0, min_idx - CONTEXT_WINDOW)
        ctx_end = min(len(all_segments), max_idx + CONTEXT_WINDOW + 1)

        context: list[dict] = []
        for ci in range(ctx_start, ctx_end):
            cs = all_segments[ci]
            # Skip segments that are in the fix group.
            if any(cs.unit_id == uid for _, uid, _ in group):
                continue
            ctx_cue = cue_map.get(cs.unit_id)
            if ctx_cue:
                context.append(
                    {
                        "unit_id": cs.unit_id,
                        "source": cs.source_text,
                        "translation": ctx_cue.text.replace("\n", "\\n"),
                    }
                )

        entries: list[dict] = []
        all_issues: list[str] = []
        all_suggestions: list[str] = []
        for _seg_idx, unit_id, seg in group:
            old_cue = cue_map[unit_id]
            score = score_map.get(unit_id)
            entries.append(
                {
                    "unit_id": unit_id,
                    "source": seg.source_text,
                    "duration": round(seg.end - seg.start, 1),
                    "max_chars": int((seg.end - seg.start) * 8),
                    "current": old_cue.text.replace("\n", "\\n"),
                }
            )
            if score and score.issues:
                for issue in score.issues:
                    all_issues.append(f"[{unit_id}] {issue}")
            if score and score.suggestion:
                all_suggestions.append(score.suggestion)

        fix = {
            "entries": entries,
            "context": context,
            "issues": all_issues,
            "suggestions": all_suggestions,
        }
        fixes.append(fix)

    return render_prompt(
        "refine.j2",
        target_lang=config.target_lang,
        fixes=fixes,
        glossary=config.glossary,
        content_summary=config.content_summary,
    )


# ── Response parsing ─────────────────────────────────────────────────────────


def _parse_refine_response(
    response: str,
    groups: list[list[tuple[int, str, Segment]]],
    cue_map: dict[str, SubtitleCue],
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    """Parse LLM refinement response into corrected SubtitleCue objects."""
    response = response.strip()

    json_match = re.search(r"\[([\s\S]*)\]", response)
    if json_match:
        data = json.loads(json_match.group(0))
    else:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("      ⚠ Refine: could not parse LLM response")
            return []

    if not isinstance(data, list):
        return []

    # Flatten groups to get all task unit_ids.
    task_ids = {uid for group in groups for _, uid, _ in group}
    refined: list[SubtitleCue] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("unit_id", ""))
        if uid not in task_ids:
            continue

        text = str(item.get("text", "") or "")
        text = text.replace("\\n", "\n").strip()
        if not text:
            continue

        original_cue = cue_map.get(uid)
        if not original_cue:
            continue

        # Apply basic normalization.
        text = _normalize_refined_punctuation(text, config.target_lang)

        refined.append(
            type(original_cue)(
                cue_id=original_cue.cue_id,
                unit_id=uid,
                start=original_cue.start,
                end=original_cue.end,
                text=text,
                lang=config.target_lang,
                speaker=original_cue.speaker,
                words=list(original_cue.words),
            )
        )

    return refined


# ── Helpers ──────────────────────────────────────────────────────────────────


def _normalize_refined_punctuation(text: str, lang: str) -> str:
    """Basic punctuation normalization for refined translations."""
    if lang == "zh" and text:
        last_char = text.rstrip()[-1] if text.rstrip() else ""
        if last_char in CJK_CLAUSE_PUNCT:
            text = text.rstrip()[:-1] + "。"
        elif last_char not in SENTENCE_ENDS:
            text = text.rstrip() + "。"
    return text
