"""Translation refinement — targeted re-translation of low-quality segments.

Runs after evaluation identifies problematic translations.  Sends each
low-quality segment (with wider context and diagnostic feedback) back to
the LLM for a corrected translation.

Usage::

    from .refine import refine_translations

    refined, usage = refine_translations(
        low_score_ids={...},
        all_cues=translated_cues,
        all_segments=translation_segments,
        quality_scores=evaluation_scores,
        config=config,
        llm=client,
    )
"""

from __future__ import annotations

import json
from pathlib import Path

from light_core import logger
from light_core.progress import ProgressCallback
from light_llm.client import OpenAIClient, merge_token_usage
from light_llm.json_extract import extract_json_array
from light_models import Segment, SubtitleCue

from .. import artifacts, export
from ..config import TranslateConfig
from ..prompts import render_prompt
from .evaluate import QualityScore, evaluate_translations, get_low_score_cues, scores_to_dict
from .translate import normalize_punctuation

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
    config: TranslateConfig,
    *,
    llm: OpenAIClient | None = None,
    glossary: dict | None = None,
    content_summary: dict | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[SubtitleCue], dict | None]:
    """Re-translate low-quality translations with diagnostic feedback.

    For each low-scoring segment, includes up to CONTEXT_WINDOW neighbouring
    segments as context.  The LLM receives the evaluation issues as specific
    instructions for what to fix.

    Returns a list of corrected ``SubtitleCue`` objects — only the ones
    that were successfully refined.  The caller merges these back into
    the main cue list.
    """
    if llm is None or not low_score_ids:
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

    client = llm

    refined_cues: list[SubtitleCue] = []
    total_usage: dict = {}
    system_prompt = _render_refine_system_prompt(config, glossary=glossary, content_summary=content_summary)

    # Process in small batches to amortize LLM overhead.
    total_batches = max(1, (len(groups) + REFINE_BATCH_SIZE - 1) // REFINE_BATCH_SIZE)
    for batch_idx in range(0, len(groups), REFINE_BATCH_SIZE):
        batch = groups[batch_idx : batch_idx + REFINE_BATCH_SIZE]
        batch_no = batch_idx // REFINE_BATCH_SIZE + 1
        batch_cues, usage = _refine_batch(batch, all_segments, cue_map, score_map, client, config, system_prompt)
        refined_cues.extend(batch_cues)
        merge_token_usage(total_usage, usage)
        if progress is not None:
            progress(batch_no / total_batches, f"评估精修中... {batch_no}/{total_batches}")

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
    config: TranslateConfig,
    system_prompt: str,
) -> tuple[list[SubtitleCue], dict]:
    """Refine a batch of low-quality translation groups."""
    user_prompt = _build_refine_user_prompt(groups, all_segments, cue_map, score_map, config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response, usage = client.chat(messages, temperature=0.2)
    except Exception as e:
        logger.warning(f"      ⚠ Refine batch failed: {e}")
        return [], {}

    return _parse_refine_response(response, groups, cue_map, config), usage


# ── Prompt construction ──────────────────────────────────────────────────────


def _render_refine_system_prompt(
    config: TranslateConfig,
    *,
    glossary: dict | None = None,
    content_summary: dict | None = None,
) -> str:
    """Build cache-friendly system prompt for refinement."""
    return render_prompt(
        "refine_system.j2",
        target_lang=config.target_lang,
        glossary=config.glossary if glossary is None else glossary,
        content_summary=config.content_summary if content_summary is None else content_summary,
    )


def _build_refine_fixes(
    groups: list[list[tuple[int, str, Segment]]],
    all_segments: list[Segment],
    cue_map: dict[str, SubtitleCue],
    score_map: dict[str, QualityScore],
) -> list[dict]:
    """Build fix-group structures for the refine user prompt."""
    fixes: list[dict] = []

    for group in groups:
        group_indices = [seg_idx for seg_idx, _, _ in group]
        min_idx = min(group_indices)
        max_idx = max(group_indices)

        ctx_start = max(0, min_idx - CONTEXT_WINDOW)
        ctx_end = min(len(all_segments), max_idx + CONTEXT_WINDOW + 1)

        context: list[dict] = []
        for ci in range(ctx_start, ctx_end):
            cs = all_segments[ci]
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

        fixes.append(
            {
                "entries": entries,
                "context": context,
                "issues": all_issues,
                "suggestions": all_suggestions,
            }
        )

    return fixes


def _build_refine_user_prompt(
    groups: list[list[tuple[int, str, Segment]]],
    all_segments: list[Segment],
    cue_map: dict[str, SubtitleCue],
    score_map: dict[str, QualityScore],
    config: TranslateConfig,
) -> str:
    """Build per-batch user prompt for refinement."""
    fixes = _build_refine_fixes(groups, all_segments, cue_map, score_map)
    return render_prompt("refine_user.j2", fixes=fixes)


def _build_refine_prompt(
    groups: list[list[tuple[int, str, Segment]]],
    all_segments: list[Segment],
    cue_map: dict[str, SubtitleCue],
    score_map: dict[str, QualityScore],
    config: TranslateConfig,
) -> str:
    """Legacy single-message prompt builder (tests only)."""
    return (
        _render_refine_system_prompt(config)
        + "\n\n"
        + _build_refine_user_prompt(groups, all_segments, cue_map, score_map, config)
    )


# ── Response parsing ─────────────────────────────────────────────────────────


def _parse_refine_response(
    response: str,
    groups: list[list[tuple[int, str, Segment]]],
    cue_map: dict[str, SubtitleCue],
    config: TranslateConfig,
) -> list[SubtitleCue]:
    """Parse LLM refinement response into corrected SubtitleCue objects."""
    response = response.strip()

    json_fragment = extract_json_array(response)
    if json_fragment is not None:
        data = json.loads(json_fragment)
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
        text = normalize_punctuation(text, config.target_lang, punctuate_blank=True)

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


# ── Evaluate + refine driver ──────────────────────────────────────────────────


def evaluate_and_refine(
    translated_cues: list[SubtitleCue],
    translation_segments: list[Segment],
    config: TranslateConfig,
    tx_dir: Path,
    *,
    llm: OpenAIClient | None = None,
    glossary: dict | None = None,
    content_summary: dict | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[SubtitleCue], dict[str, dict] | None]:
    """Evaluate translation quality and refine low-scoring cues.

    Returns the (possibly refined) list of translated cues and per-step usage.
    """
    if not config.evaluate_enabled or not translated_cues:
        return translated_cues, None

    logger.info("  Evaluating translation quality...")
    quality_scores, eval_usage = evaluate_translations(
        translated_cues, translation_segments, config, llm=llm, glossary=glossary, content_summary=content_summary
    )
    breakdown: dict[str, dict] = {}
    if eval_usage:
        breakdown["translate.evaluate"] = eval_usage

    if not quality_scores:
        return translated_cues, breakdown or None

    avg_score = sum(s.overall for s in quality_scores) / len(quality_scores)
    low_count = len([s for s in quality_scores if s.overall < config.quality_threshold])
    logger.info(
        f"    Quality: avg {avg_score:.2f}, "
        f"{low_count}/{len(quality_scores)} below threshold ({config.quality_threshold})"
    )

    # Snapshot original translations before refinement (for quality.json).
    original_trans = {c.unit_id: c.text for c in translated_cues}

    # ── Refine low-quality translations ──
    low_ids = get_low_score_cues(quality_scores, config.quality_threshold)
    if low_ids:
        for round_num in range(config.max_refine_rounds):
            logger.info(f"    Refine round {round_num + 1}/{config.max_refine_rounds}...")

            refined, refine_usage = refine_translations(
                low_ids,
                translated_cues,
                translation_segments,
                quality_scores,
                config,
                llm=llm,
                glossary=glossary,
                content_summary=content_summary,
                progress=progress,
            )
            if refine_usage:
                merge_token_usage(breakdown.setdefault("translate.refine", {}), refine_usage)

            if not refined:
                break

            # Merge refined cues back.
            refined_map = {c.unit_id: c for c in refined}
            translated_cues = [refined_map.get(c.unit_id, c) for c in translated_cues]

            # Re-evaluate refined cues for next round.
            if round_num < config.max_refine_rounds - 1:
                quality_scores, round_eval_usage = evaluate_translations(
                    translated_cues,
                    translation_segments,
                    config,
                    llm=llm,
                    glossary=glossary,
                    content_summary=content_summary,
                )
                if round_eval_usage:
                    merge_token_usage(breakdown.setdefault("translate.evaluate", {}), round_eval_usage)
                low_ids = get_low_score_cues(quality_scores, config.quality_threshold)
                if not low_ids:
                    logger.info("    All translations now above threshold.")
                    break
        else:
            logger.info(f"    Reached max refine rounds ({config.max_refine_rounds}).")

    # Save quality report (only low-scoring units).
    low_scores = [s for s in quality_scores if s.overall < config.quality_threshold]
    source_map = {s.unit_id: s.source_text for s in translation_segments}
    score_data = scores_to_dict(low_scores)
    for d in score_data:
        d["source"] = source_map.get(d["unit_id"], "")
        d["translation"] = original_trans.get(d["unit_id"], "")
    export.export_json_file(
        {"scores": score_data},
        str(tx_dir / artifacts.QUALITY_JSON),
    )

    return translated_cues, breakdown or None
