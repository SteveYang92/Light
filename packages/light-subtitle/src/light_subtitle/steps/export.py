"""Export step — write SRT/VTT/ASS/JSON subtitle files and transcript.json.

Thin orchestration over :mod:`light_subtitle.pipeline.export` (which owns
the actual file formats).  Formatting lives in :mod:`.subtitle`; this
module only decides which tracks exist and writes them out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from light_models import SubtitleCue

from .. import artifacts
from ..pipeline import export as export_module
from .subtitle import _format_source, _format_target, _wants_bilingual_exports

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator


def _export_transcript(orch: Orchestrator, out: Path) -> None:
    export_module.export_transcript(
        orch.state.words,
        orch.state.segments,
        str(artifacts.transcript_path(out)),
        source=f"{orch.config.asr.value} {orch.config.whisper_model}",
    )


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
        from ..utils.ffmpeg import probe_video_size

        export_module.export_bilingual_ass(
            source_fmt,
            target_fmt,
            str(out / "bilingual.ass"),
            source_segments=orch.state.composed_segments,
            font=orch.config.font,
            style=orch.config.style,
            frame_size=probe_video_size(orch.config.input_path),
        )
        export_module.export_bilingual_vtt(
            source_fmt,
            target_fmt,
            str(out / "bilingual.vtt"),
            source_segments=orch.state.composed_segments,
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


def _run_export(orch: Orchestrator) -> None:
    out = Path(orch.config.output_dir)

    if orch.config.target_lang is None:
        _write_source_exports(orch, out, orch.state.formatted_source_cues or _format_source(orch))
    elif _wants_bilingual_exports(orch):
        _write_bilingual_exports(
            orch,
            out,
            orch.state.formatted_source_cues or _format_source(orch),
            orch.state.formatted_target_cues if orch.state.formatted_target_cues is not None else _format_target(orch),
        )
    else:
        _write_translated_exports(orch, out, orch.state.formatted_target_cues or _format_target(orch))

    _export_transcript(orch, out)
