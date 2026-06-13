"""Translation pipeline — compose → split overlong → translate → evaluate → refine → save artifacts.

Usage::

    from .translate import run as translate_run
    result = translate_run(segments, source_cues, config, output_dir)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from light_models import Segment, SubtitleCue, Word

from ... import logger
from ...config import SubtitleConfig
from .. import export
from .compose import compose_segments
from .context import TranslateContext as TranslateContext
from .evaluate import evaluate_translations, get_low_score_cues, scores_to_dict
from .refine import refine_translations
from .split import split_overlong_units
from .translate import load_partial_cues as load_partial_cues
from .translate import run as _translate_live
from .translate import translate_missing


@dataclass
class TranslateResult:
    """Result of the translation pipeline phase."""

    translated_cues: list[SubtitleCue] = field(default_factory=list)
    usage: dict | None = None


def _compose_and_split(
    segments: list[Segment],
    config: SubtitleConfig,
    tx_dir: Path,
) -> list[Segment]:
    """Compose fragments → split overlong units → save debug compose.json."""
    translation_segments = compose_segments(segments)
    logger.info(f"  Compose: {len(segments)} segments → {len(translation_segments)} translation units")

    translation_segments = split_overlong_units(translation_segments, config)
    logger.info(f"  Split overlong: → {len(translation_segments)} units after splitting")

    tx_dir.mkdir(parents=True, exist_ok=True)
    # Debug: save compose results (after splitting).
    compose_out = [
        {
            "unit_id": s.unit_id,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "duration": round(s.end - s.start, 1),
            "speaker": s.speaker,
            "text": s.source_text,
        }
        for s in translation_segments
    ]
    (tx_dir / "compose.json").write_text(json.dumps(compose_out, ensure_ascii=False, indent=2), encoding="utf-8")

    return translation_segments


def load_cached_translation(
    tx_dir: Path,
    config: SubtitleConfig,
) -> tuple[list[SubtitleCue], dict | None]:
    """Load translated cues and usage from cached raw.json / usage.json.

    If *segment_words.json* exists (saved during compose), word timing is
    re-attached to each cue by matching ``unit_id``, enabling word-boundary
    alignment in the pace step.
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
        )
        for c in raw_data
    ]

    # Re-attach word timing from segment_words.json (saved by compose phase).
    seg_words_path = tx_dir / "segment_words.json"
    if seg_words_path.exists():
        with open(seg_words_path) as f:
            seg_words_map = json.load(f)
        for cue in translated_cues:
            word_dicts = seg_words_map.get(cue.unit_id)
            if word_dicts:
                cue.words = [Word(**w) for w in word_dicts]

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
) -> None:
    """Save raw.json, source.json, and usage.json artifacts."""
    export.export_raw_cues(translated_cues, str(tx_dir / "raw.json"))
    export.export_raw_cues(source_cues, str(tx_dir / "source.json"))
    if usage:
        logger.info(
            f"  Tokens: {usage.get('total_tokens', '?')} "
            f"(prompt: {usage.get('prompt_tokens', '?')}, "
            f"completion: {usage.get('completion_tokens', '?')})"
        )
        export.export_json_file(usage, str(tx_dir / "usage.json"))


def _save_translation_segment_words(translation_segments: list[Segment], tx_dir: Path) -> None:
    """Save per-unit word-level timing so cached translation can re-attach words later."""
    data: dict[str, list[dict]] = {}
    for seg in translation_segments:
        if seg.words:
            data[seg.unit_id] = [
                {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence, "speaker": w.speaker}
                for w in seg.words
            ]
    (tx_dir / "segment_words.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_and_refine(
    translated_cues: list[SubtitleCue],
    translation_segments: list[Segment],
    config: SubtitleConfig,
    tx_dir: Path,
) -> list[SubtitleCue]:
    """Evaluate translation quality and refine low-scoring cues.

    Returns the (possibly refined) list of translated cues.
    """
    if not config.evaluate_enabled or not translated_cues:
        return translated_cues

    logger.info("  Evaluating translation quality...")
    quality_scores = evaluate_translations(translated_cues, translation_segments, config)

    if not quality_scores:
        return translated_cues

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

            refined = refine_translations(low_ids, translated_cues, translation_segments, quality_scores, config)

            if not refined:
                break

            # Merge refined cues back.
            refined_map = {c.unit_id: c for c in refined}
            translated_cues = [refined_map.get(c.unit_id, c) for c in translated_cues]

            # Re-evaluate refined cues for next round.
            if round_num < config.max_refine_rounds - 1:
                quality_scores = evaluate_translations(translated_cues, translation_segments, config)
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

    return translated_cues


def _retry_missing_translations(
    translated_cues: list[SubtitleCue],
    translation_segments: list[Segment],
    config: SubtitleConfig,
    usage: dict | None,
) -> tuple[list[SubtitleCue], dict | None]:
    """Retry any translation units that failed LLM parsing."""
    MAX_RETRY = 2
    for attempt in range(MAX_RETRY):
        translated_ids = {c.unit_id for c in translated_cues}
        missing_ids = {s.unit_id for s in translation_segments} - translated_ids
        if not missing_ids:
            break

        logger.warning(f"  ⚠ Missing {len(missing_ids)} translations, retry {attempt + 1}/{MAX_RETRY}")

        retry_cues, retry_usage = translate_missing(translation_segments, missing_ids, config)

        if retry_cues:
            # Replace missing cues with retry results.
            merged = [c for c in translated_cues if c.unit_id not in missing_ids]
            merged.extend(retry_cues)
            translated_cues = merged
            if usage:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[k] = usage.get(k, 0) + retry_usage.get(k, 0)
        else:
            logger.warning(f"    ✗ Still missing {len(missing_ids)} units after retry")
            break

    return translated_cues, usage


# ── Public step helpers (used by step_registry) ───────────────────────────────

compose_and_split = _compose_and_split
save_segment_words = _save_translation_segment_words
retry_missing = _retry_missing_translations
evaluate_and_refine = _evaluate_and_refine
save_artifacts = _save_translation_artifacts


def load_compose_segments(tx_dir: Path, segments: list[Segment], config: SubtitleConfig) -> list[Segment]:
    """Rebuild translation segments from compose.json when resuming mid-translate."""
    compose_path = tx_dir / "compose.json"
    if not compose_path.exists():
        return _compose_and_split(segments, config, tx_dir)

    with open(compose_path, encoding="utf-8") as f:
        compose_data = json.load(f)

    # Directly reconstruct Segment objects from compose.json.
    # No need to re-run compose/split (which makes LLM calls) when
    # the persisted data already contains the correct unit IDs.
    rebuilt: list[Segment] = []
    for item in compose_data:
        rebuilt.append(
            Segment(
                unit_id=item["unit_id"],
                start=item.get("start", 0.0),
                end=item.get("end", 0.0),
                speaker=item.get("speaker", ""),
                source_text=item.get("text", ""),
                words=[],
            )
        )
    return rebuilt


# ── Main entry point ──────────────────────────────────────────────────────────


def run(
    segments: list[Segment],
    source_cues: list[SubtitleCue],
    config: SubtitleConfig,
    output_dir: str | Path,
) -> TranslateResult:
    """Run the full translation pipeline.

    Steps:
      1. Compose segments → split overlong units.
      2. Translate via LLM.
      3. Retry any missing translations (LLM parse failures).
      4. Evaluate quality + refine low-scoring cues.
      5. Save all artifacts (*raw.json*, *source.json*, *usage.json*, *quality.json*).

    Returns a ``TranslateResult`` with ``(translated_cues, usage)``.
    """
    output_dir = Path(output_dir)
    tx_dir = output_dir / "translations"
    tx_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Compose → split overlong ───────────────────────────

    translation_segments = _compose_and_split(segments, config, tx_dir)

    # Persist word-level timing for resume / pace re-attachment.
    _save_translation_segment_words(translation_segments, tx_dir)

    # Live translation.
    logger.info("  Translating...")
    translated_cues, usage = _translate_live(translation_segments, config, tx_dir)

    logger.info(f"  Translation: {len(translated_cues)} translated cues")

    # ── Step 2: Retry missing translations ──────────────────────────

    translated_cues, usage = _retry_missing_translations(translated_cues, translation_segments, config, usage)

    # ── Step 3: Evaluate + refine ────────────────────────────────────

    translated_cues = _evaluate_and_refine(translated_cues, translation_segments, config, tx_dir)

    # ── Step 4: Save artifacts (final cues) ──────────────────────────

    _save_translation_artifacts(translated_cues, source_cues, usage, tx_dir)

    return TranslateResult(
        translated_cues=translated_cues,
        usage=usage,
    )
