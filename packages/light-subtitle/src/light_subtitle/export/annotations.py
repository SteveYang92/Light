"""Annotation (secondary subtitle) export — ASS with box positioning and WebVTT."""

from __future__ import annotations

import re
from pathlib import Path

from light_models import SubtitleCue

from ..style.fonts import ASS_V4_PLUS_STYLE_FORMAT, annotation_style_line
from .formats import _normalize_plain_subtitle_text, _resolved_font

# Leading ※ markers (ASS export adds one; avoid duplicates from conversion or LLM)
_ANNOTATION_MARKER_RE = re.compile(r"^\s*(?:※\s*)+")


def strip_annotation_marker(text: str) -> str:
    """Remove leading ※ markers from annotation body text."""
    return _ANNOTATION_MARKER_RE.sub("", text).strip()


def format_annotation_display(text: str) -> str:
    """Normalize annotation text to exactly one leading ※ marker."""
    body = strip_annotation_marker(text)
    if not body:
        return ""
    return f"※ {body}"


def _annotation_timed_entries(cues: list[SubtitleCue], annotations: dict[str, str]) -> list[dict]:
    """Content-driven annotation timing, shared by the ASS and VTT writers.

    Phase 1: base duration proportional to text length (5 CPS, min 40 chars).
    Phase 2: extend toward the next annotation's start (cap: reading time ×
    1.3, gap 0.3 s).
    """
    CPS = 5.0
    MIN_LEN = 40
    entries: list[dict] = []
    for cue in cues:
        annotation = annotations.get(cue.unit_id)
        if not annotation:
            continue
        body = strip_annotation_marker(annotation)
        if not body:
            continue
        base_end = cue.start + max(len(body), MIN_LEN) / CPS
        entries.append(
            {
                "start": cue.start,
                "end": max(cue.end, base_end),
                "text": body,
            }
        )

    if not entries:
        return entries

    GAP = 0.3
    for i in range(len(entries)):
        reading_time = entries[i]["end"] - entries[i]["start"]
        extension_cap = entries[i]["start"] + reading_time * 1.3
        if i < len(entries) - 1:
            next_start = entries[i + 1]["start"]
            entries[i]["end"] = min(next_start - GAP, extension_cap)
        else:
            entries[i]["end"] = max(entries[i]["end"], extension_cap)
    return entries


def export_annotation_ass(
    cues: list[SubtitleCue], annotations: dict[str, str], output_path: str, width_pct: int = 30, font: str | None = None
) -> None:
    """Export secondary subtitle annotations as ASS with top-left positioning and dark background.

    Display duration is content-driven (see :func:`_annotation_timed_entries`).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    entries = _annotation_timed_entries(cues, annotations)
    if not entries:
        return

    font_name = _resolved_font(font)

    # ── Write ASS ──
    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("Title: Light Subtitle Annotations\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n\n")

        f.write("[V4+ Styles]\n")
        f.write(ASS_V4_PLUS_STYLE_FORMAT)
        right_margin = max(10, 1920 * (100 - width_pct) // 100)
        f.write(annotation_style_line(font_name, right_margin))
        f.write("\n")

        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        from light_text import seconds_to_ass

        for entry in entries:
            start = seconds_to_ass(entry["start"])
            end = seconds_to_ass(entry["end"])
            text = entry["text"].replace("\n", "\\N")
            f.write(f"Dialogue: 0,{start},{end},Annotation,,0,0,0,,{format_annotation_display(text)}\n")


def export_annotation_vtt(cues: list[SubtitleCue], annotations: dict[str, str], output_path: str) -> None:
    """Export secondary subtitle annotations as WebVTT with top-left positioning.

    Uses VTT's built-in ::cue positioning: line:0% places the cue at the top,
    align:start for left alignment. Compatible with Video.js and all browsers.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    entries = _annotation_timed_entries(cues, annotations)
    if not entries:
        return

    from light_text import seconds_to_vtt

    with open(output, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, entry in enumerate(entries):
            start = seconds_to_vtt(entry["start"])
            end = seconds_to_vtt(entry["end"])
            text = format_annotation_display(_normalize_plain_subtitle_text(entry["text"]))
            f.write(f"{i + 1}\n")
            f.write(f"{start} --> {end} align:start line:0%\n")
            f.write(f"{text}\n\n")
