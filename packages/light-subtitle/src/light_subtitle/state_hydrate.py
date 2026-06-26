"""Hydrate PipelineState from on-disk artifacts for resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from light_models import Segment, Word

from .config import SubtitleConfig
from .cue_builder import build_source_cues
from .language import detect_source_lang
from .pipeline import context_prep as context_prep_pipeline
from .pipeline import translate as translate_pipeline
from .pipeline.asr.artifacts import audio_wav_path, load_asr_words

if TYPE_CHECKING:
    from .orchestrator import Orchestrator
    from .step_plan import PlanStep


def hydrate_state(orch: Orchestrator, plan: list[PlanStep], start_idx: int) -> None:
    """Replay hydrate handlers through the resume target step inclusive."""
    for step in plan[:start_idx]:
        handler = step.definition.hydrate
        if handler is not None:
            handler(orch)
    if start_idx < len(plan):
        handler = plan[start_idx].definition.hydrate
        if handler is not None:
            handler(orch)


def hydrate_asr_audio(orch: Orchestrator) -> None:
    wav = audio_wav_path(orch.config.output_dir)
    if wav.exists():
        orch.asr_ctx.audio_path = str(wav)


def hydrate_asr_words(orch: Orchestrator) -> None:
    orch.asr_ctx.words = load_asr_words(orch.config)


def hydrate_transcript_words(orch: Orchestrator) -> None:
    orch.state.words = load_words_from_transcript(_out(orch.config) / "transcript.json")


def hydrate_words_after_correct(orch: Orchestrator) -> None:
    out = _out(orch.config)
    post = out / "transcript_correct" / "post_correct.json"
    if post.exists():
        orch.state.words = load_words_from_debug_json(post)
    else:
        hydrate_transcript_words(orch)


def hydrate_words_after_punct(orch: Orchestrator) -> None:
    out = _out(orch.config)
    punct = out / "punct_restore" / "punct_restore.json"
    if punct.exists():
        orch.state.words = load_words_from_debug_json(punct)
    else:
        hydrate_words_after_correct(orch)


def hydrate_segments_from_disk(orch: Orchestrator) -> None:
    out = _out(orch.config)
    if not orch.state.words:
        hydrate_words_after_punct(orch)
    orch.state.source_lang = detect_source_lang(orch.state.words)
    orch.state.segments = load_segments_from_json(out / "segment" / "segment.json", orch.state.words)


def hydrate_context_from_cache(orch: Orchestrator) -> None:
    out = _out(orch.config)
    glossary_path = out / "context" / "glossary.json"
    summary_path = out / "context" / "summary.json"
    if glossary_path.exists() and summary_path.exists():
        cached = context_prep_pipeline.load_cached_context(orch.config.output_dir)
        orch.state.auto_glossary = cached.glossary
        orch.state.content_summary = cached.summary
    orch.state.merged_glossary = context_prep_pipeline.merge_glossary(
        orch.state.auto_glossary,
        orch.config.glossary,
    )
    orch.config.glossary = orch.state.merged_glossary
    orch.config.content_summary = orch.state.content_summary


def hydrate_compose_segments(orch: Orchestrator) -> None:
    """Hydrate composed segments and rebuild ``raw_source_cues`` from them.

    Reads ``compose/compose.json`` (re-running compose+split if absent) and
    ``compose/segment_words.json`` for word-level timing.  The English
    source cues are rebuilt from composed units so the EN track shares the
    same ``unit_id`` graph as the translated track.
    """
    hydrate_segments_from_disk(orch)
    hydrate_context_from_cache(orch)
    compose_dir = _out(orch.config) / "compose"
    orch.state.composed_segments = translate_pipeline.load_compose_segments(
        compose_dir, orch.state.segments, orch.config
    )
    _attach_segment_words(orch.state.composed_segments, compose_dir)
    orch.state.raw_source_cues = build_source_cues(orch.state.composed_segments, orch.state.source_lang)


def _attach_segment_words(segments: list[Segment], compose_dir: Path) -> None:
    """Re-attach word timing from ``segment_words.json`` to composed segments.

    ``load_compose_segments`` rebuilds ``Segment`` objects from
    ``compose.json`` with ``words=[]``; this refills words so pace can do
    word-boundary alignment.  Mirrors the logic in
    ``translate.load_cached_translation``.
    """
    seg_words_path = compose_dir / "segment_words.json"
    if not seg_words_path.exists():
        return
    with open(seg_words_path, encoding="utf-8") as f:
        seg_words_map = json.load(f)
    for seg in segments:
        word_dicts = seg_words_map.get(seg.unit_id)
        if word_dicts:
            seg.words = [Word(**w) for w in word_dicts]


def hydrate_partial_cues(orch: Orchestrator) -> None:
    from .pipeline.translate import load_partial_cues

    tx_dir = _out(orch.config) / "translations"
    partial = tx_dir / "partial.json"
    if partial.exists():
        orch.tx_ctx.translated_cues = load_partial_cues(tx_dir, orch.config)


def hydrate_translated_cues(orch: Orchestrator) -> None:
    hydrate_compose_segments(orch)
    tx_dir = _out(orch.config) / "translations"
    orch.state.translated_cues, orch.state.translation_usage = translate_pipeline.load_cached_translation(
        tx_dir, orch.config
    )


def hydrate_subtitle_export(orch: Orchestrator) -> None:
    hydrate_compose_segments(orch)
    raw = _out(orch.config) / "translations" / "raw.json"
    if raw.exists():
        hydrate_translated_cues(orch)


def _out(config_or_dir: SubtitleConfig | str | Path) -> Path:
    if isinstance(config_or_dir, SubtitleConfig):
        return Path(config_or_dir.output_dir)
    return Path(config_or_dir)


def load_words_from_transcript(path: Path) -> list[Word]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [_word_from_dict(w) for w in data.get("words", [])]


def load_words_from_debug_json(path: Path) -> list[Word]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    words: list[Word] = []
    for seg in data:
        for w in seg.get("words", []):
            words.append(_word_from_dict(w))
    return words


def _word_from_dict(raw: dict) -> Word:
    """Build Word from JSON, ignoring debug-only keys like ``changed``."""
    return Word(
        text=raw["text"],
        start=raw["start"],
        end=raw["end"],
        confidence=raw.get("confidence", 1.0),
        speaker=raw.get("speaker"),
    )


def load_segments_from_json(path: Path, words: list[Word]) -> list[Segment]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments: list[Segment] = []
    for unit in data.get("units", []):
        seg_words = _slice_words_for_unit(words, unit)
        segments.append(
            Segment(
                unit_id=unit["unit_id"],
                start=unit["start"],
                end=unit["end"],
                source_text=unit.get("source_text", ""),
                speaker=unit.get("speaker"),
                words=seg_words,
            )
        )
    return segments


def _slice_words_for_unit(words: list[Word], unit: dict) -> list[Word]:
    start = unit.get("start", 0.0)
    end = unit.get("end", 0.0)
    matched = [w for w in words if w.start >= start - 0.05 and w.end <= end + 0.05]
    if matched:
        return matched
    return list(words)


# Legacy helper for unit tests.
def hydrate_pipeline_state(state: Any, config: SubtitleConfig, start_step_id: str) -> None:
    """Populate state fields for a resume point (test helper)."""

    class _Orch:
        pass

    fake = _Orch()
    fake.state = state
    fake.config = config
    fake.asr_ctx = type("ctx", (), {"audio_path": "", "words": []})()
    fake.tx_ctx = type("ctx", (), {"translated_cues": [], "usage": None})()

    from .step_registry import StepId, build_enabled_definitions

    plan_defs = {d.id.value: d for d in build_enabled_definitions(config)}
    handlers = {
        StepId.CORRECT.value: hydrate_transcript_words,
        StepId.PUNCT.value: hydrate_words_after_correct,
        StepId.SEGMENT.value: hydrate_words_after_punct,
    }
    for step_id, handler in handlers.items():
        if step_id == start_step_id:
            handler(fake)
            return
        if step_id in plan_defs:
            handler(fake)
