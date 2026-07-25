"""Hydrate PipelineState from on-disk artifacts for resume."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from light_models import Segment, Word, word_from_dict
from light_subtitle import artifacts as sub_artifacts
from light_subtitle import context_prep as context_prep_pipeline
from light_subtitle import translate as translate_pipeline
from light_subtitle.cue_builder import build_source_cues
from light_subtitle.language import detect_source_lang
from light_subtitle.translate.join import load_joined_units

from . import artifacts
from .artifacts import audio_wav_path, load_asr_words
from .config import SubtitleConfig
from .llm_client import client_from_config

if TYPE_CHECKING:
    from .orchestrator import Orchestrator
    from .step_plan import PlanStep


def hydrate_state(orch: Orchestrator, plan: list[PlanStep], start_idx: int) -> None:
    """Replay hydrate handlers through the resume target step inclusive.

    The replay range is [0, start_idx] *inclusive*: each step's hydrate
    loads the state its own INPUT artifacts produce, so the resume target
    itself also needs its inputs hydrated (e.g. resuming at ``correct``
    needs transcript.json loaded, which is ``correct``'s own hydrate).
    """
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
        orch.state.audio_path = str(wav)


def hydrate_asr_words(orch: Orchestrator) -> None:
    orch.state.words = load_asr_words(orch.config)


def hydrate_transcript_words(orch: Orchestrator) -> None:
    orch.state.words = artifacts.read_transcript_words(artifacts.transcript_path(_out(orch.config)))


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
    orch.state.segments = sub_artifacts.read_segment_units(sub_artifacts.segment_json_path(out), orch.state.words)


def hydrate_context_from_cache(orch: Orchestrator) -> None:
    out = _out(orch.config)
    glossary_path = out / "context" / "glossary.json"
    summary_path = out / "context" / "summary.json"
    if glossary_path.exists() and summary_path.exists():
        cached = context_prep_pipeline.load_cached_context(orch.config.output_dir)
        orch.state.auto_glossary = cached.glossary
        orch.state.content_summary = cached.summary
    sync_glossary(orch, recompute=True)


def sync_glossary(orch: Orchestrator, *, recompute: bool = False) -> None:
    """Merge auto-extracted + user glossary into ``state.merged_glossary``.

    Single implementation for the three sites that sync glossary state:
    the context step (always recomputes after a fresh extraction), the
    translate guard (fills only when empty), and resume hydration
    (recomputes from cached/auto + user glossaries).  Reads the user's
    initial glossary from ``config.glossary``; never writes config —
    downstream prompt builders receive the merged values from state.
    """
    if recompute or not orch.state.merged_glossary:
        orch.state.merged_glossary = context_prep_pipeline.merge_glossary(
            orch.state.auto_glossary,
            orch.config.glossary,
        )


def hydrate_plan_segments(orch: Orchestrator) -> None:
    """Hydrate planned units and rebuild ``raw_source_cues`` from them.

    Reads ``plan/plan.joined.json`` when a join pass already ran (else
    ``plan/plan.json``, re-running the planner if absent) plus word-level
    timing.  The English source cues are rebuilt from planned units so
    the EN track shares the same ``unit_id`` graph as the translated track.
    """
    hydrate_segments_from_disk(orch)
    hydrate_context_from_cache(orch)
    plan_dir = sub_artifacts.plan_dir(_out(orch.config))
    orch.state.composed_segments = load_joined_units(plan_dir) or translate_pipeline.load_plan_segments(
        plan_dir,
        orch.state.segments,
        orch.config.plan_config(),
        llm=client_from_config(orch.config) if orch.config.llm_api_key else None,
    )
    _attach_segment_words(orch.state.composed_segments, plan_dir)
    orch.state.raw_source_cues = build_source_cues(orch.state.composed_segments, orch.state.source_lang)


def _attach_segment_words(segments: list[Segment], plan_dir: Path) -> None:
    """Re-attach word timing from ``segment_words.json`` to planned units.

    ``load_plan_segments``/``load_joined_units`` rebuild ``Segment``
    objects with ``words=[]``; this refills words so pace can do
    word-boundary alignment.  Shares the map-loading logic with
    ``translate.load_cached_translation`` via :mod:`light_subtitle.artifacts`.
    """
    seg_words_map = sub_artifacts.load_segment_words_map(plan_dir)
    if seg_words_map is None:
        return
    for seg in segments:
        word_dicts = seg_words_map.get(seg.unit_id)
        if word_dicts:
            seg.words = [word_from_dict(w) for w in word_dicts]


def hydrate_partial_cues(orch: Orchestrator) -> None:
    tx_dir = sub_artifacts.translations_dir(_out(orch.config))
    partial = sub_artifacts.partial_cues_path(_out(orch.config))
    if partial.exists():
        orch.state.translated_cues = translate_pipeline.load_partial_cues(tx_dir, orch.config.translate_config())


def hydrate_translated_cues(orch: Orchestrator) -> None:
    hydrate_plan_segments(orch)
    tx_dir = sub_artifacts.translations_dir(_out(orch.config))
    orch.state.translated_cues, orch.state.translation_usage = translate_pipeline.load_cached_translation(
        tx_dir, orch.config.translate_config(), current_segments=orch.state.composed_segments
    )


def hydrate_annotations(orch: Orchestrator) -> None:
    """Fill ``state.annotations`` from disk (or cue fields) for export resume.

    ``--resume-from export`` skips the annotate step; without this hydrate,
    ``--annotate`` alone leaves ``state.annotations`` empty and export writes
    no ``video.annotations.*`` even though ASS/VTT may already exist.
    """
    loaded = sub_artifacts.load_annotations(_out(orch.config))
    if loaded:
        orch.state.annotations = loaded
        return
    if orch.state.translated_cues:
        orch.state.annotations = {c.unit_id: c.annotation for c in orch.state.translated_cues if c.annotation}


def hydrate_annotate_inputs(orch: Orchestrator) -> None:
    hydrate_translated_cues(orch)
    hydrate_annotations(orch)


def hydrate_subtitle_export(orch: Orchestrator) -> None:
    hydrate_plan_segments(orch)
    raw = sub_artifacts.raw_cues_path(_out(orch.config))
    if raw.exists():
        hydrate_translated_cues(orch)
    hydrate_annotations(orch)


def _out(config_or_dir: SubtitleConfig | str | Path) -> Path:
    if isinstance(config_or_dir, SubtitleConfig):
        return Path(config_or_dir.output_dir)
    return Path(config_or_dir)


def load_words_from_debug_json(path: Path) -> list[Word]:
    words: list[Word] = []
    for seg in sub_artifacts.read_json(path):
        for w in seg.get("words", []):
            words.append(word_from_dict(w))
    return words
