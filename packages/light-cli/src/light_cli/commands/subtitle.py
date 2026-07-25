"""``light subtitle`` — transcript.json → subtitle files (srt/vtt/ass).

Thin wrapper over :mod:`light_subtitle`: segment → plan (compose) →
translate (when ``--target-lang``) → layout → export.  Artifact names and
formats match the pipeline (``video.<lang>.srt``, ``video.bilingual.ass``,
``cues.json``, ``transcript.json``), so the output interchanges with
``light-subtitle`` runs, ``light-qc``, and the web backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from light_core import logger
from light_models import SubtitleCue
from light_subtitle import artifacts as sub_artifacts
from light_subtitle import export as export_module
from light_subtitle import segment
from light_subtitle import subtitle as subtitle_format
from light_subtitle import translate as translate_pipeline
from light_subtitle.config import LayoutConfig, PlanConfig, SegmentConfig, TranslateConfig
from light_subtitle.cue_builder import build_source_cues
from light_subtitle.language import detect_source_lang
from light_subtitle.style.config import SubtitleStyleConfig
from light_subtitle.subtitle import strip_punct

from ..artifacts import cues_path, read_transcript_words, transcript_path
from .common import LlmApiKey, LlmBaseUrl, LlmModel, build_client, resolve_api_key


def _format(cues: list[SubtitleCue], layout: LayoutConfig, llm, words) -> list[SubtitleCue]:
    formatted, _usage = subtitle_format.run(cues, layout, llm=llm, transcript_words=words)
    return strip_punct.strip_chinese_punct(formatted)


def subtitle(
    input_path: Annotated[str, typer.Option("-i", "--input", help="transcript.json (light-transcript.v1)")],
    output_dir: Annotated[str, typer.Option("-o", "--output", help="Output directory")] = "./output",
    target_lang: Annotated[
        str,
        typer.Option("--target-lang", help="Target language for translation (e.g. zh). Empty = source-only"),
    ] = "",
    bilingual: Annotated[bool, typer.Option("--bilingual", help="Output both source and translated subtitles")] = False,
    style_config: Annotated[
        str,
        typer.Option("--style-config", help="YAML style overrides for bilingual subtitle boxes"),
    ] = "",
    font: Annotated[str, typer.Option("--font", help="Subtitle font for ASS export")] = "PingFang SC",
    llm_base_url: LlmBaseUrl = "https://api.deepseek.com",
    llm_model: LlmModel = "deepseek-v4-flash",
    llm_api_key: LlmApiKey = "",
):
    """Build subtitle files from a transcript.json (no ASR, no video needed)."""
    if target_lang and not resolve_api_key(llm_api_key):
        raise typer.BadParameter("--target-lang requires an LLM API key (--llm-api-key or DEEPSEEK_API_KEY).")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    words = read_transcript_words(input_path)
    if not words:
        raise typer.BadParameter(f"No words found in {input_path}.")
    source_lang = detect_source_lang(words)
    logger.info(f"  Detected language: {source_lang}")

    # ── Segment → plan (compose) ──────────────────────────────────────────────
    seg_config = SegmentConfig()
    segments = segment.run(words, seg_config.max_duration, seg_config.max_chars_per_line)
    export_module.export_segments(words, segments, str(sub_artifacts.segment_json_path(out)))
    logger.info(f"  Segment: {len(segments)} segments")

    # LLM improves boundary planning / CPS compression; optional here.
    llm = build_client(llm_base_url, llm_model, llm_api_key) if resolve_api_key(llm_api_key) else None

    plan_dir = sub_artifacts.plan_dir(out)
    composed, _plan_usage = translate_pipeline.plan_units(segments, PlanConfig(), plan_dir, llm=llm)
    translate_pipeline.save_segment_words(composed, plan_dir)
    raw_source_cues = build_source_cues(composed, source_lang)

    # ── Translate (optional) ──────────────────────────────────────────────────
    translated_cues: list[SubtitleCue] = []
    if target_lang:
        translated_cues, _tx_usage = translate_pipeline.run(
            composed,
            TranslateConfig(target_lang=target_lang),
            sub_artifacts.translations_dir(out),
            llm=llm,
        )
        translate_pipeline.attach_words_to_cues(translated_cues, plan_dir)
        logger.info(f"  Translation: {len(translated_cues)} translated cues")

    # ── Layout + export ───────────────────────────────────────────────────────
    layout = LayoutConfig(target_lang=target_lang or None)
    style = SubtitleStyleConfig.load_yaml(style_config) if style_config else SubtitleStyleConfig()

    if not target_lang:
        source_fmt = _format(raw_source_cues, layout, llm, words)
        export_module.export_srt(source_fmt, str(sub_artifacts.sidecar_path(out, f"{source_lang}.srt")))
        export_module.export_vtt(source_fmt, str(sub_artifacts.sidecar_path(out, f"{source_lang}.vtt")))
        export_module.export_json(source_fmt, str(cues_path(out)))
    elif bilingual:
        source_fmt = _format(raw_source_cues, layout, llm, words)
        target_fmt = _format(translated_cues, layout, llm, words) if translated_cues else []
        export_module.export_srt(source_fmt, str(sub_artifacts.sidecar_path(out, f"{source_lang}.srt")))
        export_module.export_vtt(source_fmt, str(sub_artifacts.sidecar_path(out, f"{source_lang}.vtt")))
        if target_fmt:
            export_module.export_srt(target_fmt, str(sub_artifacts.sidecar_path(out, f"{target_lang}.srt")))
            export_module.export_vtt(target_fmt, str(sub_artifacts.sidecar_path(out, f"{target_lang}.vtt")))
            export_module.export_bilingual_ass(
                source_fmt,
                target_fmt,
                str(sub_artifacts.sidecar_path(out, "bilingual.ass")),
                source_segments=composed,
                font=font,
                style=style,
            )
            export_module.export_bilingual_vtt(
                source_fmt,
                target_fmt,
                str(sub_artifacts.sidecar_path(out, "bilingual.vtt")),
                source_segments=composed,
            )
        export_module.export_json(source_fmt + target_fmt, str(cues_path(out)))
    else:
        target_fmt = _format(translated_cues, layout, llm, words)
        export_module.export_srt(target_fmt, str(sub_artifacts.sidecar_path(out, f"{target_lang}.srt")))
        export_module.export_vtt(target_fmt, str(sub_artifacts.sidecar_path(out, f"{target_lang}.vtt")))
        export_module.export_json(target_fmt, str(cues_path(out)))

    export_module.export_transcript(words, segments, str(transcript_path(out)), source=f"transcript:{input_path}")
    typer.echo(f"subtitles: {out}")
