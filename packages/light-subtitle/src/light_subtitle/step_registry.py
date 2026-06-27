"""Declarative pipeline step registry — single source of truth for step IDs and behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from light_models import SubtitleCue

from . import logger
from .config import AsrEngine, SubtitleConfig
from .cue_builder import build_source_cues
from .language import detect_source_lang
from .pipeline import annotate as annotate_pipeline
from .pipeline import context_prep as context_prep_pipeline
from .pipeline import export as export_module
from .pipeline import segment, strip_punct, subtitle
from .pipeline import translate as translate_pipeline
from .pipeline.asr import align, diarize, extract_audio, transcribe, whisperx
from .pipeline.asr.artifacts import asr_words_path, audio_wav_path, save_asr_words, save_whisper_cpp_raw
from .pipeline.punct_restore import restore_punctuation
from .pipeline.transcript_correct import correct_transcript
from .pipeline.translate.translate import run as translate_live
from .state_hydrate import (
    hydrate_asr_audio,
    hydrate_asr_words,
    hydrate_compose_segments,
    hydrate_partial_cues,
    hydrate_segments_from_disk,
    hydrate_subtitle_export,
    hydrate_transcript_words,
    hydrate_translated_cues,
    hydrate_words_after_correct,
    hydrate_words_after_punct,
)
from .usage.tracker import merge_token_usage, usage_delta

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

StepRunner = Callable[["Orchestrator"], None]
ArtifactFn = Callable[[SubtitleConfig], tuple[Path, ...]]


class StepId(StrEnum):
    ASR_EXTRACT = "asr.extract"
    ASR_TRANSCRIBE = "asr.transcribe"
    ASR_ALIGN = "asr.align"
    ASR_DIARIZE = "asr.diarize"
    CORRECT = "correct"
    PUNCT = "punct"
    SEGMENT = "segment"
    CONTEXT = "context"
    TRANSLATE_COMPOSE = "translate.compose"
    TRANSLATE_TRANSLATE = "translate.translate"
    TRANSLATE_RETRY = "translate.retry"
    TRANSLATE_EVALUATE = "translate.evaluate"
    TRANSLATE_SAVE = "translate.save"
    ANNOTATE = "annotate"
    SUBTITLE = "subtitle"
    EXPORT = "export"


@dataclass(frozen=True)
class StepDefinition:
    """Metadata and handlers for one pipeline step."""

    id: StepId
    run: StepRunner
    artifacts: ArtifactFn
    progress_start: StepRunner | None = None
    progress_end: StepRunner | None = None
    hydrate: StepRunner | None = None
    enabled: Callable[[SubtitleConfig], bool] = lambda _c: True


def _out(config: SubtitleConfig) -> Path:
    return Path(config.output_dir)


def _tx_dir(config: SubtitleConfig) -> Path:
    return _out(config) / "translations"


def _compose_dir(config: SubtitleConfig) -> Path:
    return _out(config) / "compose"


def _subtitle_artifact_paths(config: SubtitleConfig) -> tuple[Path, ...]:
    if config.target_lang:
        return (
            _tx_dir(config) / "raw.json",
            _out(config) / "segment" / "segment.json",
            _compose_dir(config) / "compose.json",
        )
    return (
        _out(config) / "segment" / "segment.json",
        _compose_dir(config) / "compose.json",
    )


# ── ASR helpers ───────────────────────────────────────────────────────────────


def _asr_progress_start(orch: Orchestrator) -> None:
    orch._progress("asr", 0.0, "提取音频中...")


def _asr_progress_end(orch: Orchestrator) -> None:
    orch._progress("asr", 1.0, f"ASR 完成 ({len(orch.state.words)} 个词)")


def _sync_asr_words(orch: Orchestrator) -> None:
    orch.state.words = orch.asr_ctx.words
    _export_transcript(orch, _out(orch.config))


def _resolve_asr_lang(config: SubtitleConfig) -> str:
    return config.language if config.language != "auto" else "en"


def _transcribe_words(config: SubtitleConfig, audio_path: str) -> list:
    if config.asr == AsrEngine.WHISPERX:
        return whisperx.run(audio_path, language=_resolve_asr_lang(config))
    words = transcribe.run(config, audio_path)
    raw_src = Path(config.output_dir) / "asr" / "whisper_output.json"
    if raw_src.exists():
        save_whisper_cpp_raw(config, raw_src)
    return words


def _run_asr_extract(orch: Orchestrator) -> None:
    orch.asr_ctx.audio_path = extract_audio.run(orch.config)
    logger.info(f"  Extract: {orch.asr_ctx.audio_path}")


def _run_asr_transcribe(orch: Orchestrator) -> None:
    orch.asr_ctx.words = _transcribe_words(orch.config, orch.asr_ctx.audio_path)
    save_asr_words(orch.config, orch.asr_ctx.words)
    logger.info(f"  Transcribe: {len(orch.asr_ctx.words)} words")
    _sync_asr_words(orch)


def _run_asr_align(orch: Orchestrator) -> None:
    orch.asr_ctx.words = align.run(
        orch.asr_ctx.words,
        orch.asr_ctx.audio_path,
        language=_resolve_asr_lang(orch.config),
    )
    save_asr_words(orch.config, orch.asr_ctx.words)
    logger.info(f"  Align: {len(orch.asr_ctx.words)} words")
    _sync_asr_words(orch)


def _run_asr_diarize(orch: Orchestrator) -> None:
    orch.asr_ctx.words = diarize.run(
        orch.asr_ctx.words,
        orch.asr_ctx.audio_path,
        hf_token=orch.config.hf_token,
        model_name=orch.config.diarize_model,
    )
    save_asr_words(orch.config, orch.asr_ctx.words)
    logger.info("  Diarization done.")
    _sync_asr_words(orch)


def _hydrate_asr_align(orch: Orchestrator) -> None:
    hydrate_asr_audio(orch)
    hydrate_asr_words(orch)


# ── Translate helpers ─────────────────────────────────────────────────────────


def _ensure_translate_ready(orch: Orchestrator) -> bool:
    if orch.config.target_lang is None:
        orch._progress("translate", 1.0, "无需翻译")
        return False
    if not orch.config.llm_api_key:
        logger.warning("  Translation skipped (no LLM API key). Using source cues.")
        orch._progress("translate", 1.0, "跳过翻译 (无 API key)")
        return False
    if not orch.state.merged_glossary:
        orch.state.merged_glossary = context_prep_pipeline.merge_glossary(
            orch.state.auto_glossary,
            orch.config.glossary,
        )
    orch.config.glossary = orch.state.merged_glossary
    orch.config.content_summary = orch.state.content_summary
    return True


def _sync_translate_state(orch: Orchestrator) -> None:
    orch.state.translated_cues = orch.tx_ctx.translated_cues
    orch.state.translation_usage = orch.tx_ctx.usage


def _translate_progress_start(orch: Orchestrator) -> None:
    orch._progress("translate", 0.0, "翻译中...")


def _translate_progress_end(orch: Orchestrator) -> None:
    orch._progress("translate", 1.0, f"翻译完成 ({len(orch.state.translated_cues)} 条)")


def _compose_progress_start(orch: Orchestrator) -> None:
    orch._progress("compose", 0.0, "组合翻译单元中...")


def _compose_progress_end(orch: Orchestrator) -> None:
    orch._progress(
        "compose",
        1.0,
        f"组合完成 ({len(orch.state.composed_segments)} 个单元)",
    )


def _run_translate_compose(orch: Orchestrator) -> None:
    """Shared compose+split step.

    Runs for both monolingual English and bilingual runs.  Builds
    ``orch.state.composed_segments`` from the pause-based ``segments`` and
    rebuilds ``raw_source_cues`` from those composed units so the English
    track shares the same ``unit_id`` graph as the translated track.
    """
    compose_dir = _compose_dir(orch.config)
    compose_dir.mkdir(parents=True, exist_ok=True)
    if not orch.state.composed_segments:
        orch.state.composed_segments, split_usage = translate_pipeline.compose_and_split(
            orch.state.segments, orch.config, compose_dir
        )
        if split_usage:
            orch.usage_tracker.record("translate.compose_split", split_usage)
    translate_pipeline.save_segment_words(orch.state.composed_segments, compose_dir)
    orch.state.raw_source_cues = build_source_cues(orch.state.composed_segments, orch.state.source_lang)


def _run_translate_translate(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    tx_dir = _tx_dir(orch.config)
    logger.info("  Translating...")
    orch.tx_ctx.translated_cues, orch.tx_ctx.usage = translate_live(orch.state.composed_segments, orch.config, tx_dir)
    if orch.tx_ctx.usage:
        orch.tx_ctx.usage_breakdown["translate.translate"] = dict(orch.tx_ctx.usage)
        orch.usage_tracker.record("translate.translate", orch.tx_ctx.usage)
    logger.info(f"  Translation: {len(orch.tx_ctx.translated_cues)} translated cues")
    _sync_translate_state(orch)


def _run_translate_retry(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    before_usage = dict(orch.tx_ctx.usage or {})
    orch.tx_ctx.translated_cues, orch.tx_ctx.usage = translate_pipeline.retry_missing(
        orch.tx_ctx.translated_cues,
        orch.state.composed_segments,
        orch.config,
        orch.tx_ctx.usage,
    )
    retry_usage = usage_delta(before_usage, orch.tx_ctx.usage)
    if retry_usage:
        orch.tx_ctx.usage_breakdown["translate.retry"] = retry_usage
        orch.usage_tracker.record("translate.retry", retry_usage)
    _sync_translate_state(orch)


def _run_translate_evaluate(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    orch.tx_ctx.translated_cues, eval_breakdown = translate_pipeline.evaluate_and_refine(
        orch.tx_ctx.translated_cues,
        orch.state.composed_segments,
        orch.config,
        _tx_dir(orch.config),
    )
    if eval_breakdown:
        orch.tx_ctx.usage_breakdown.update(eval_breakdown)
        orch.usage_tracker.record_breakdown(eval_breakdown)
        for step_usage in eval_breakdown.values():
            merge_token_usage(orch.tx_ctx.usage, step_usage)
    _sync_translate_state(orch)


def _run_translate_save(orch: Orchestrator) -> None:
    if not _ensure_translate_ready(orch):
        return
    translate_pipeline.save_artifacts(
        orch.tx_ctx.translated_cues,
        orch.state.raw_source_cues,
        orch.tx_ctx.usage,
        _tx_dir(orch.config),
        breakdown=orch.tx_ctx.usage_breakdown or None,
    )
    _sync_translate_state(orch)


def _hydrate_translate_mid(orch: Orchestrator) -> None:
    hydrate_compose_segments(orch)
    hydrate_partial_cues(orch)


# ── Correct / punct / segment / context / annotate ──────────────────────────


def _correct_progress_start(orch: Orchestrator) -> None:
    orch._progress("correct", 0.0, "转录矫正中...")


def _correct_progress_end(orch: Orchestrator) -> None:
    orch._progress("correct", 1.0, "转录矫正完成")


def _punct_progress_start(orch: Orchestrator) -> None:
    orch._progress("punct", 0.0, "恢复标点中...")


def _punct_progress_end(orch: Orchestrator) -> None:
    orch._progress("punct", 1.0, "标点恢复完成")


def _segment_progress_start(orch: Orchestrator) -> None:
    orch._progress("segment", 0.0, "语义断句中...")


def _segment_progress_end(orch: Orchestrator) -> None:
    orch._progress("segment", 1.0, f"断句完成 ({len(orch.state.segments)} 段)")


def _context_progress_start(orch: Orchestrator) -> None:
    orch._progress("context", 0.0, "提取翻译上下文中...")


def _context_progress_end(orch: Orchestrator) -> None:
    orch._progress("context", 1.0, "翻译上下文就绪")


def _annotate_progress_start(orch: Orchestrator) -> None:
    if orch.config.annotate and orch.state.translated_cues:
        orch._progress("annotate", 0.0, "生成注解中...")


def _annotate_progress_end(orch: Orchestrator) -> None:
    if not orch.config.annotate or not orch.state.translated_cues:
        orch._progress("annotate", 1.0, "无需注解")
    else:
        orch._progress("annotate", 1.0, f"注解完成 ({len(orch.state.annotations)} 条)")


def _format_progress_start(orch: Orchestrator) -> None:
    orch._progress("format", 0.0, "格式化字幕中...")


def _format_progress_end(orch: Orchestrator) -> None:
    orch._progress("format", 1.0, "格式化完成")


def _run_correct(orch: Orchestrator) -> None:
    orch.state.words, usage = correct_transcript(orch.state.words, orch.config, orch.config.output_dir)
    if usage and isinstance(usage.get("breakdown"), dict):
        orch.usage_tracker.record_breakdown(usage["breakdown"])


def _run_punct(orch: Orchestrator) -> None:
    orch.state.words, usage = restore_punctuation(orch.state.words, orch.config, orch.config.output_dir)
    if usage:
        orch.usage_tracker.record("punct", usage)


def _run_segment(orch: Orchestrator) -> None:
    orch.state.source_lang = detect_source_lang(orch.state.words)
    logger.info(f"  Detected language: {orch.state.source_lang}")

    orch.state.segments = segment.run(orch.state.words, orch.config.max_duration)
    logger.info(f"  Segment: {len(orch.state.segments)} segments")

    export_module.export_segments(
        orch.state.words,
        orch.state.segments,
        str(_out(orch.config) / "segment" / "segment.json"),
    )
    # raw_source_cues is built by the compose step from composed units so
    # the English track shares the same unit graph as the translated track.


def _run_context(orch: Orchestrator) -> None:
    if orch.config.context_prep_enabled:
        result, usage = context_prep_pipeline.prepare_context(
            orch.state.segments,
            orch.config,
            orch.config.output_dir,
        )
        orch.state.auto_glossary = result.glossary
        orch.state.content_summary = result.summary
        if usage:
            orch.usage_tracker.record("context", usage)

    orch.state.merged_glossary = context_prep_pipeline.merge_glossary(
        orch.state.auto_glossary,
        orch.config.glossary,
    )
    orch.config.glossary = orch.state.merged_glossary
    orch.config.content_summary = orch.state.content_summary

    if orch.state.content_summary and not orch.config.speaker_names:
        speakers = orch.state.content_summary.get("speakers")
        if isinstance(speakers, dict):
            orch.config.speaker_names = {str(k): str(v) for k, v in speakers.items()}

    logger.info(f"  Translation context: {len(orch.state.merged_glossary)} glossary terms")


def _run_annotate(orch: Orchestrator) -> None:
    if not orch.config.annotate or not orch.state.translated_cues:
        return

    logger.info("  Generating annotations...")
    orch.state.translated_cues, usage = annotate_pipeline.generate_annotations(
        orch.state.translated_cues,
        orch.state.composed_segments or orch.state.segments,
        orch.config,
        orch.config.output_dir,
    )
    if usage:
        orch.usage_tracker.record("annotate", usage)
    orch.state.annotations = {c.unit_id: c.annotation for c in orch.state.translated_cues if c.annotation}
    logger.info(f"  Annotations: {len(orch.state.annotations)} terms annotated")


# ── Subtitle format / export ──────────────────────────────────────────────────


def _export_transcript(orch: Orchestrator, out: Path) -> None:
    export_module.export_transcript(
        orch.state.words,
        orch.state.segments,
        str(out / "transcript.json"),
        source=f"{orch.config.asr.value} {orch.config.whisper_model}",
    )


def _format_source(orch: Orchestrator) -> list[SubtitleCue]:
    orch.config.transcript_words = orch.state.words
    formatted = subtitle.run(orch.state.raw_source_cues, orch.config)
    return strip_punct.strip_chinese_punct(formatted)


def _format_target(orch: Orchestrator) -> list[SubtitleCue]:
    if not orch.state.translated_cues:
        return []
    translate_pipeline.attach_words_to_cues(orch.state.translated_cues, _compose_dir(orch.config))
    orch.config.transcript_words = orch.state.words
    formatted = subtitle.run(orch.state.translated_cues, orch.config)
    return strip_punct.strip_chinese_punct(formatted)


def _write_source_exports(orch: Orchestrator, out: Path, formatted: list[SubtitleCue]) -> None:
    ext = orch.state.source_lang
    export_module.export_srt(formatted, str(out / f"{ext}.srt"))
    export_module.export_vtt(formatted, str(out / f"{ext}.vtt"))
    export_module.export_json(formatted, str(out / "cues.json"))


def _write_translated_exports(orch: Orchestrator, out: Path, formatted: list[SubtitleCue]) -> None:
    ext = orch.config.target_lang if orch.state.translated_cues else orch.state.source_lang
    export_module.export_srt(formatted, str(out / f"{ext}.srt"))
    export_module.export_vtt(formatted, str(out / f"{ext}.vtt"))
    export_module.export_json(formatted, str(out / "cues.json"))
    if orch.config.annotate:
        export_module.export_annotation_ass(
            formatted,
            orch.state.annotations,
            str(out / "annotations.ass"),
            width_pct=orch.config.annotation_width,
            font=orch.config.font,
        )
        export_module.export_annotation_vtt(
            formatted,
            orch.state.annotations,
            str(out / "annotations.vtt"),
        )


def _write_bilingual_exports(
    orch: Orchestrator, out: Path, source_fmt: list[SubtitleCue], target_fmt: list[SubtitleCue]
) -> None:
    src_ext = orch.state.source_lang
    tgt_ext = orch.config.target_lang or "target"
    if orch.state.source_lang == tgt_ext:
        tgt_ext = "target"

    export_module.export_srt(source_fmt, str(out / f"{src_ext}.srt"))
    export_module.export_vtt(source_fmt, str(out / f"{src_ext}.vtt"))

    if target_fmt:
        export_module.export_srt(target_fmt, str(out / f"{tgt_ext}.srt"))
        export_module.export_vtt(target_fmt, str(out / f"{tgt_ext}.vtt"))
        # Composed EN segments carry word-level timing; bilingual ASS uses them
        # to derive each ZH cue's EN text via the shared unit_id graph.
        export_module.export_bilingual_ass(
            source_fmt,
            target_fmt,
            str(out / "bilingual.ass"),
            source_segments=orch.state.composed_segments,
            font=orch.config.font,
        )

    export_module.export_json(source_fmt + target_fmt, str(out / "cues.json"))

    if orch.config.annotate:
        export_module.export_annotation_ass(
            target_fmt,
            orch.state.annotations,
            str(out / "annotations.ass"),
            width_pct=orch.config.annotation_width,
            font=orch.config.font,
        )
        export_module.export_annotation_vtt(
            target_fmt,
            orch.state.annotations,
            str(out / "annotations.vtt"),
        )


def _run_subtitle(orch: Orchestrator) -> None:
    if orch.config.target_lang is None:
        orch._formatted_source = _format_source(orch)
        orch._formatted_target = None
    elif orch.config.bilingual:
        orch._formatted_source = _format_source(orch)
        orch._formatted_target = _format_target(orch) if orch.state.translated_cues else []
    else:
        orch._formatted_source = None
        orch._formatted_target = _format_target(orch)


def _run_export(orch: Orchestrator) -> None:
    out = _out(orch.config)

    if orch.config.target_lang is None:
        _write_source_exports(orch, out, orch._formatted_source or _format_source(orch))
    elif orch.config.bilingual:
        _write_bilingual_exports(
            orch,
            out,
            orch._formatted_source or _format_source(orch),
            orch._formatted_target if orch._formatted_target is not None else _format_target(orch),
        )
    else:
        _write_translated_exports(orch, out, orch._formatted_target or _format_target(orch))

    _export_transcript(orch, out)


# ── Step definition list ────────────────────────────────────────────────────────


def build_step_definitions(config: SubtitleConfig) -> list[StepDefinition]:
    """Return ordered step definitions for *config* (before enabled filtering)."""
    return [
        StepDefinition(
            id=StepId.ASR_EXTRACT,
            run=_run_asr_extract,
            artifacts=lambda _c: (),
            progress_start=_asr_progress_start,
            hydrate=hydrate_asr_audio,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.ASR_TRANSCRIBE,
            run=_run_asr_transcribe,
            artifacts=lambda c: (audio_wav_path(c.output_dir),),
            progress_end=_asr_progress_end,
            hydrate=hydrate_asr_audio,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.ASR_ALIGN,
            run=_run_asr_align,
            artifacts=lambda c: (asr_words_path(c), audio_wav_path(c.output_dir)),
            progress_end=_asr_progress_end,
            hydrate=_hydrate_asr_align,
            enabled=lambda c: c.asr == AsrEngine.WHISPER_CPP,
        ),
        StepDefinition(
            id=StepId.ASR_DIARIZE,
            run=_run_asr_diarize,
            artifacts=lambda c: (asr_words_path(c), audio_wav_path(c.output_dir)),
            progress_end=_asr_progress_end,
            hydrate=_hydrate_asr_align,
            enabled=lambda c: c.diarize,
        ),
        StepDefinition(
            id=StepId.CORRECT,
            run=_run_correct,
            artifacts=lambda c: (_out(c) / "transcript.json",),
            progress_start=_correct_progress_start,
            progress_end=_correct_progress_end,
            hydrate=hydrate_transcript_words,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.PUNCT,
            run=_run_punct,
            artifacts=lambda c: (_out(c) / "transcript.json",),
            progress_start=_punct_progress_start,
            progress_end=_punct_progress_end,
            hydrate=hydrate_words_after_correct,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.SEGMENT,
            run=_run_segment,
            artifacts=lambda c: (_out(c) / "transcript.json",),
            progress_start=_segment_progress_start,
            progress_end=_segment_progress_end,
            hydrate=hydrate_words_after_punct,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.CONTEXT,
            run=_run_context,
            artifacts=lambda c: (_out(c) / "segment" / "segment.json",),
            progress_start=_context_progress_start,
            progress_end=_context_progress_end,
            hydrate=hydrate_segments_from_disk,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_COMPOSE,
            run=_run_translate_compose,
            artifacts=lambda c: (_out(c) / "segment" / "segment.json", _compose_dir(c) / "compose.json"),
            progress_start=_compose_progress_start,
            progress_end=_compose_progress_end,
            hydrate=hydrate_compose_segments,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.TRANSLATE_TRANSLATE,
            run=_run_translate_translate,
            artifacts=lambda c: (_compose_dir(c) / "compose.json",),
            progress_start=_translate_progress_start,
            hydrate=hydrate_compose_segments,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_RETRY,
            run=_run_translate_retry,
            artifacts=lambda c: (_compose_dir(c) / "compose.json",),
            hydrate=_hydrate_translate_mid,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_EVALUATE,
            run=_run_translate_evaluate,
            artifacts=lambda c: (_tx_dir(c) / "partial.json",),
            hydrate=_hydrate_translate_mid,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key and c.evaluate_enabled),
        ),
        StepDefinition(
            id=StepId.TRANSLATE_SAVE,
            run=_run_translate_save,
            artifacts=lambda c: (_tx_dir(c) / "partial.json",),
            progress_end=_translate_progress_end,
            hydrate=_hydrate_translate_mid,
            enabled=lambda c: bool(c.target_lang and c.llm_api_key),
        ),
        StepDefinition(
            id=StepId.ANNOTATE,
            run=_run_annotate,
            artifacts=lambda c: (_tx_dir(c) / "raw.json",),
            progress_start=_annotate_progress_start,
            progress_end=_annotate_progress_end,
            hydrate=hydrate_translated_cues,
            enabled=lambda c: bool(c.annotate and c.target_lang),
        ),
        StepDefinition(
            id=StepId.SUBTITLE,
            run=_run_subtitle,
            artifacts=_subtitle_artifact_paths,
            progress_start=_format_progress_start,
            hydrate=hydrate_subtitle_export,
            enabled=lambda _c: True,
        ),
        StepDefinition(
            id=StepId.EXPORT,
            run=_run_export,
            artifacts=_subtitle_artifact_paths,
            progress_end=_format_progress_end,
            hydrate=hydrate_subtitle_export,
            enabled=lambda _c: True,
        ),
    ]


def build_enabled_definitions(config: SubtitleConfig) -> list[StepDefinition]:
    """Return enabled steps in pipeline order."""
    return [d for d in build_step_definitions(config) if d.enabled(config)]


# ── ASR phase boundary ─────────────────────────────────────────────────────────

ASR_STEP_IDS: frozenset[StepId] = frozenset(
    {
        StepId.ASR_EXTRACT,
        StepId.ASR_TRANSCRIBE,
        StepId.ASR_ALIGN,
        StepId.ASR_DIARIZE,
    }
)
