"""Translation pipeline — plan cue boundaries → translate → evaluate → refine → save artifacts.

Usage::

    from .translate import run as translate_run
    result = translate_run(segments, source_cues, config, output_dir)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from light_models import Segment, SubtitleCue, Word

from ... import logger
from ...config import SubtitleConfig
from ...usage.tracker import merge_token_usage, pick_usage_fields, save_step_usage
from .. import export
from .. import plan as plan_pipeline
from .context import TranslateContext as TranslateContext
from .evaluate import evaluate_translations, get_low_score_cues, scores_to_dict
from .refine import refine_translations
from .translate import covered_unit_ids, translate_missing
from .translate import load_partial_cues as load_partial_cues
from .translate import run as _translate_live


@dataclass
class TranslateResult:
    """Result of the translation pipeline phase."""

    translated_cues: list[SubtitleCue] = field(default_factory=list)
    usage: dict | None = None


def _plan_units(
    segments: list[Segment],
    config: SubtitleConfig,
    plan_dir: Path,
) -> tuple[list[Segment], dict | None]:
    """Plan cue units with the LLM boundary planner; persist ``plan/plan.json``."""
    plan_dir.mkdir(parents=True, exist_ok=True)
    translation_segments, plan_usage = plan_pipeline.run(segments, config, plan_dir)
    if plan_usage:
        save_step_usage(plan_dir / "usage.json", plan_usage)
    return translation_segments, plan_usage


def load_cached_translation(
    tx_dir: Path,
    config: SubtitleConfig,
) -> tuple[list[SubtitleCue], dict | None]:
    """Load translated cues and usage from cached raw.json / usage.json.

    If ``plan/segment_words.json`` exists, word timing is re-attached to each
    cue by matching ``unit_id``, enabling word-boundary alignment in the pace step.
    """
    raw_path = tx_dir / "raw.json"
    with open(raw_path) as f:
        raw_data = json.load(f)
    translated_cues = [
        SubtitleCue(
            cue_id=c["cue_id"],
            unit_id=c["unit_id"],
            start=c["start"],
            end=c["end"],
            text=c["text"],
            lang=c.get("lang", config.target_lang),
            speaker=c.get("speaker", ""),
            merged_from=c.get("merged_from", []),
        )
        for c in raw_data
    ]

    plan_dir = tx_dir.parent / "plan"
    _attach_words_to_cues(translated_cues, plan_dir)
    _attach_speakers_from_plan(translated_cues, plan_dir)

    usage: dict | None = None
    usage_path = tx_dir / "usage.json"
    if usage_path.exists():
        with open(usage_path) as f:
            usage = json.load(f)
    logger.info(f"  Translation (cached): {len(translated_cues)} cues from raw.json")
    return translated_cues, usage


def _save_translation_artifacts(
    translated_cues: list[SubtitleCue],
    source_cues: list[SubtitleCue],
    usage: dict | None,
    tx_dir: Path,
    breakdown: dict[str, dict] | None = None,
) -> None:
    """Save raw.json, source.json, and usage.json artifacts."""
    export.export_raw_cues(translated_cues, str(tx_dir / "raw.json"))
    export.export_raw_cues(source_cues, str(tx_dir / "source.json"))
    if usage:
        payload = dict(usage)
        if breakdown:
            payload["breakdown"] = breakdown
        logger.info(
            f"  Tokens: {payload.get('total_tokens', '?')} "
            f"(prompt: {payload.get('prompt_tokens', '?')}, "
            f"completion: {payload.get('completion_tokens', '?')})"
        )
        export.export_json_file(payload, str(tx_dir / "usage.json"))


def _segment_words_path(plan_dir: Path) -> Path:
    return plan_dir / "segment_words.json"


def _words_from_unit_chain(unit_ids: list[str], seg_words_map: dict[str, list[dict]]) -> list[Word]:
    """Concatenate word timing for a cue's head unit + ``merged_from`` chain."""
    words: list[Word] = []
    for uid in unit_ids:
        word_dicts = seg_words_map.get(uid)
        if word_dicts:
            words.extend(Word(**w) for w in word_dicts)
    return words


def _attach_words_to_cues(cues: list[SubtitleCue], plan_dir: Path) -> None:
    """Re-attach word timing from ``plan/segment_words.json``.

    Uses ``unit_id`` plus ``merged_from`` (in chain order) so cues that
    absorbed other units get the full ASR span, not just the head unit.
    """
    seg_words_path = _segment_words_path(plan_dir)
    if not seg_words_path.exists():
        return
    with open(seg_words_path, encoding="utf-8") as f:
        seg_words_map = json.load(f)
    for cue in cues:
        unit_chain = [cue.unit_id, *cue.merged_from]
        words = _words_from_unit_chain(unit_chain, seg_words_map)
        if words:
            cue.words = words


def _attach_speakers_from_plan(cues: list[SubtitleCue], plan_dir: Path) -> None:
    """Fill empty cue speakers from ``plan/plan.json`` metadata on resume."""
    plan_path = plan_dir / "plan.json"
    if not plan_path.exists():
        return
    with open(plan_path, encoding="utf-8") as f:
        data = json.load(f)
    speaker_by_unit = {
        item["unit_id"]: item.get("speaker", "") for item in data.get("units", []) if isinstance(item, dict)
    }
    for cue in cues:
        if not cue.speaker:
            cue.speaker = speaker_by_unit.get(cue.unit_id, "")


def _save_translation_segment_words(translation_segments: list[Segment], plan_dir: Path) -> None:
    """Save per-unit word-level timing to ``plan/segment_words.json``."""
    data: dict[str, list[dict]] = {}
    for seg in translation_segments:
        if seg.words:
            data[seg.unit_id] = [
                {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence, "speaker": w.speaker}
                for w in seg.words
            ]
    plan_dir.mkdir(parents=True, exist_ok=True)
    _segment_words_path(plan_dir).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_and_refine(
    translated_cues: list[SubtitleCue],
    translation_segments: list[Segment],
    config: SubtitleConfig,
    tx_dir: Path,
) -> tuple[list[SubtitleCue], dict[str, dict] | None]:
    """Evaluate translation quality and refine low-scoring cues.

    Returns the (possibly refined) list of translated cues and per-step usage.
    """
    if not config.evaluate_enabled or not translated_cues:
        return translated_cues, None

    logger.info("  Evaluating translation quality...")
    quality_scores, eval_usage = evaluate_translations(translated_cues, translation_segments, config)
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
                low_ids, translated_cues, translation_segments, quality_scores, config
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
                quality_scores, round_eval_usage = evaluate_translations(translated_cues, translation_segments, config)
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
        str(tx_dir / "quality.json"),
    )

    return translated_cues, breakdown or None


def _retry_missing_translations(
    translated_cues: list[SubtitleCue],
    translation_segments: list[Segment],
    config: SubtitleConfig,
    usage: dict | None,
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

        retry_cues, retry_usage = translate_missing(translation_segments, missing_ids, config)

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


# ── Public step helpers (used by step_registry) ───────────────────────────────

plan_units = _plan_units
save_segment_words = _save_translation_segment_words
attach_words_to_cues = _attach_words_to_cues
retry_missing = _retry_missing_translations
evaluate_and_refine = _evaluate_and_refine
save_artifacts = _save_translation_artifacts


def load_plan_segments(plan_dir: Path, segments: list[Segment], config: SubtitleConfig) -> list[Segment]:
    """Rebuild translation units from ``plan/plan.json`` when resuming mid-translate.

    Falls back to re-running the planner (LLM calls) only when plan.json
    is absent.
    """
    units = plan_pipeline.load_plan_units(plan_dir)
    if units is not None:
        return units
    return _plan_units(segments, config, plan_dir)[0]


# ── Main entry point ──────────────────────────────────────────────────────────


def run(
    segments: list[Segment],
    source_cues: list[SubtitleCue],
    config: SubtitleConfig,
    output_dir: str | Path,
) -> TranslateResult:
    """Run the full translation pipeline.

    Steps:
      1. Plan cue boundaries (LLM planner).
      2. Translate via LLM.
      3. Retry any missing translations (LLM parse failures).
      4. Evaluate quality + refine low-scoring cues.
      5. Save all artifacts (*raw.json*, *source.json*, *usage.json*, *quality.json*).

    Returns a ``TranslateResult`` with ``(translated_cues, usage)``.
    """
    output_dir = Path(output_dir)
    tx_dir = output_dir / "translations"
    tx_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Plan cue boundaries ─────────────────────────────

    plan_dir = output_dir / "plan"
    translation_segments, _plan_usage = _plan_units(segments, config, plan_dir)

    # Persist word-level timing for resume / pace re-attachment.
    _save_translation_segment_words(translation_segments, plan_dir)

    # Live translation.
    logger.info("  Translating...")
    translated_cues, usage = _translate_live(translation_segments, config, tx_dir)

    logger.info(f"  Translation: {len(translated_cues)} translated cues")

    # ── Step 2: Retry missing translations ──────────────────────────

    translated_cues, usage = _retry_missing_translations(translated_cues, translation_segments, config, usage)

    # ── Step 3: Evaluate + refine ────────────────────────────────────

    translated_cues, eval_breakdown = _evaluate_and_refine(translated_cues, translation_segments, config, tx_dir)

    usage_breakdown: dict[str, dict] = {}
    if usage:
        if usage.get("breakdown"):
            usage_breakdown.update(usage["breakdown"])
        else:
            usage_breakdown["translate.translate"] = pick_usage_fields(usage)
    if eval_breakdown:
        usage_breakdown.update(eval_breakdown)
        for step_usage in eval_breakdown.values():
            merge_token_usage(usage, step_usage)

    # ── Step 4: Save artifacts (final cues) ──────────────────────────

    _save_translation_artifacts(translated_cues, source_cues, usage, tx_dir, breakdown=usage_breakdown or None)

    return TranslateResult(
        translated_cues=translated_cues,
        usage=usage,
    )
