"""Text/JSON subtitle format writers — SRT, WebVTT, mono ASS, cues.json, misc JSON."""

from __future__ import annotations

import json
from pathlib import Path

from light_models import SubtitleCue, seconds_to_srt, seconds_to_vtt

from ...style.fonts import FontConfig, default_style_line, resolve_font


def _resolved_font(font: str | None) -> str:
    """Resolve *font* through the system fallback chain."""
    if font is None:
        return resolve_font(FontConfig())
    return resolve_font(FontConfig(primary=font))


def _normalize_plain_subtitle_text(text: str) -> str:
    """Convert ASS-style escaped line breaks before writing text-based subtitle formats."""
    return text.replace("\\N", "\n").replace("\\n", "\n")


def export_json_file(data: dict, output_path: str) -> None:
    """Write an arbitrary dict as JSON (used for usage stats etc.)."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_srt(cues: list[SubtitleCue], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            start = seconds_to_srt(cue.start)
            end = seconds_to_srt(cue.end)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{_normalize_plain_subtitle_text(cue.text)}\n\n")


def export_vtt(cues: list[SubtitleCue], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, cue in enumerate(cues, 1):
            start = seconds_to_vtt(cue.start)
            end = seconds_to_vtt(cue.end)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{_normalize_plain_subtitle_text(cue.text)}\n\n")


def export_json(
    cues: list[SubtitleCue], output_path: str, media_info: dict | None = None, speakers: list[dict] | None = None
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "media": media_info or {},
        "speakers": speakers or [],
        "cues": [
            {
                "id": i + 1,
                "cue_id": cue.cue_id,
                "unit_id": cue.unit_id,
                "start": cue.start,
                "end": cue.end,
                "speaker": cue.speaker,
                "lang": cue.lang,
                "text": cue.text,
                "qc": cue.qc,
            }
            for i, cue in enumerate(cues)
        ],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_ass(cues: list[SubtitleCue], output_path: str, font: str | None = None) -> None:
    """Basic ASS export — mono-language."""
    font_name = _resolved_font(font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Italic, Alignment\n")
        f.write(default_style_line(font_name))
        f.write("\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        from light_models import seconds_to_ass

        for cue in cues:
            start = seconds_to_ass(cue.start)
            end = seconds_to_ass(cue.end)
            text = cue.text.replace("\n", "\\N")
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
