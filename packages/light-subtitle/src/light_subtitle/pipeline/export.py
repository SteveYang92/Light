import json
import re
from datetime import UTC, datetime
from pathlib import Path

from light_models import Segment, SubtitleCue, Word, is_cjk, seconds_to_srt, seconds_to_vtt

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


def export_ass(cues: list[SubtitleCue], output_path: str) -> None:
    """Basic ASS export — mono-language."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Italic, Alignment\n")
        f.write("Style: Default,Arial,20,&H00FFFFFF,&H00000000,0,0,2\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        from light_models import seconds_to_ass

        for cue in cues:
            start = seconds_to_ass(cue.start)
            end = seconds_to_ass(cue.end)
            text = cue.text.replace("\n", "\\N")
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def export_bilingual_ass(zh_cues: list[SubtitleCue], en_cues: list[SubtitleCue], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Italic, Alignment\n")
        f.write("Style: ZH,Arial,20,&H00FFFFFF,&H00000000,0,0,2\n")
        f.write("Style: EN,Arial,18,&H00FFFF00,&H00000000,0,0,2\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        from light_models import seconds_to_ass

        max_cues = max(len(zh_cues), len(en_cues))
        for i in range(max_cues):
            if i < len(zh_cues):
                zc = zh_cues[i]
                start = seconds_to_ass(zc.start)
                end = seconds_to_ass(zc.end)
                text = zc.text.replace("\n", "\\N")
                f.write(f"Dialogue: 0,{start},{end},ZH,,0,0,0,,{text}\n")
            if i < len(en_cues):
                ec = en_cues[i]
                start = seconds_to_ass(ec.start)
                end = seconds_to_ass(ec.end)
                text = ec.text.replace("\n", "\\N")
                f.write(f"Dialogue: 1,{start},{end},EN,,0,0,10,,{text}\n")


def export_transcript(
    words: list[Word], segments: list[Segment], output_path: str, source: str = "whisper.cpp"
) -> None:
    """Export a standardized transcript.json with word-level timestamps.

    This is the canonical transcription format consumed by light-qc's
    ``--transcript`` parameter.  It is ASR-agnostic: the pipeline
    normalises whisper.cpp output into a flat word list, so replacing
    the ASR backend does not require any changes in light-qc.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build word-index lookup for segment word_range.
    word_index: dict[int, int] = {}
    for i, w in enumerate(words):
        word_index[id(w)] = i

    data = {
        "format": "light-transcript.v1",
        "source": source,
        "language": _detect_lang_from_words(words),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "words": [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
                "speaker": w.speaker,
            }
            for w in words
        ],
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker,
                "text": s.source_text,
                "word_range": [
                    word_index[id(s.words[0])] if s.words else 0,
                    word_index[id(s.words[-1])] if s.words else 0,
                ],
            }
            for s in segments
        ],
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _detect_lang_from_words(words: list[Word]) -> str:
    zh_count = 0
    total = 0
    for w in words:
        for ch in w.text:
            if is_cjk(ch):
                zh_count += 1
                total += 1
            elif ch.isalpha():
                total += 1
    if total == 0:
        return "en"
    return "zh" if zh_count / total >= 0.4 else "en"


def export_raw_cues(cues: list[SubtitleCue], output_path: str) -> None:
    """Export raw translated cues as JSON for LLM output review."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "cue_id": c.cue_id,
            "unit_id": c.unit_id,
            "start": c.start,
            "end": c.end,
            "text": c.text,
            "lang": c.lang,
        }
        for c in cues
    ]
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_annotation_ass(
    cues: list[SubtitleCue], annotations: dict[str, str], output_path: str, width_pct: int = 30
) -> None:
    """Export secondary subtitle annotations as ASS with top-left positioning and dark background.

    Display duration is content-driven:
    Phase 1: base timing proportional to text length (4 CPS, min 50 chars → 12.5 s).
    Phase 2: extend toward next annotation start (cap: base × 1.5, spacing 0.1 s).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: collect annotated entries with content-driven base timing ──
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
        return

    # ── Phase 2: extend toward next annotation (cap: reading_time × 1.3) ──
    GAP = 0.3

    for i in range(len(entries)):
        reading_time = entries[i]["end"] - entries[i]["start"]
        extension_cap = entries[i]["start"] + reading_time * 1.3

        if i < len(entries) - 1:
            next_start = entries[i + 1]["start"]
            entries[i]["end"] = min(next_start - GAP, extension_cap)
        else:
            entries[i]["end"] = max(entries[i]["end"], extension_cap)

    # ── Phase 3: write ASS ──
    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("Title: Light Subtitle Annotations\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n\n")

        f.write("[V4+ Styles]\n")
        f.write(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
            " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
            " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
            " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )
        right_margin = max(10, 1920 * (100 - width_pct) // 100)
        f.write(
            "Style: Annotation,PingFangSC-Regular,40,&H00FFFFFF,&H00000000,"
            "&H00000000,&H00000000,-1,0,0,0,100,100,0,0,"
            f"1,3,2,7,10,{right_margin},10,1\n\n"
        )

        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        from light_models import seconds_to_ass

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
        entries.append({"start": cue.start, "end": max(cue.end, base_end), "text": body})

    if not entries:
        return

    # Extend toward next annotation
    GAP = 0.3
    for i in range(len(entries)):
        reading_time = entries[i]["end"] - entries[i]["start"]
        cap = entries[i]["start"] + reading_time * 1.3
        if i < len(entries) - 1:
            entries[i]["end"] = min(entries[i + 1]["start"] - GAP, cap)
        else:
            entries[i]["end"] = max(entries[i]["end"], cap)

    from light_models import seconds_to_vtt

    with open(output, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, entry in enumerate(entries):
            start = seconds_to_vtt(entry["start"])
            end = seconds_to_vtt(entry["end"])
            text = format_annotation_display(_normalize_plain_subtitle_text(entry["text"]))
            f.write(f"{i + 1}\n")
            f.write(f"{start} --> {end} align:start line:0%\n")
            f.write(f"{text}\n\n")


def export_segments(words: list[Word], segments: list[Segment], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_words": len(words),
        "total_units": len(segments),
        "words": [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
                "speaker": w.speaker,
            }
            for w in words
        ],
        "units": [
            {
                "unit_id": s.unit_id,
                "start": s.start,
                "end": s.end,
                "duration": round(s.end - s.start, 3),
                "speaker": s.speaker,
                "word_count": len(s.words),
                "source_text": s.source_text,
                "word_range": {
                    "from": s.words[0].text if s.words else "",
                    "to": s.words[-1].text if s.words else "",
                },
            }
            for s in segments
        ],
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
