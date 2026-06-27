import json
import re
from datetime import UTC, datetime
from pathlib import Path

from light_models import Segment, SubtitleCue, Word, is_cjk, seconds_to_srt, seconds_to_vtt

from ..fonts import (
    ASS_V4_PLUS_STYLE_FORMAT,
    FontConfig,
    annotation_style_line,
    bilingual_style_line,
    default_style_line,
    resolve_font,
)

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


def export_bilingual_ass(
    en_cues: list[SubtitleCue],
    zh_cues: list[SubtitleCue],
    output_path: str,
    source_segments: list[Segment] | None = None,
    font: str | None = None,
) -> None:
    """Export bilingual ASS with ZH as the anchor and EN derived from segment words.

    Each ZH cue becomes one ASS ``Dialogue`` (ZH line on top, EN line below,
    joined with ``\\N``), keeping the ZH cue's own time window.  ZH is the
    anchor: it is never lost or duplicated, and its display window is never
    cut or stretched by EN.

    The EN text for each ZH cue is the joined words of the composed EN
    segment(s) sharing its ``unit_id`` (and any units listed in
    ``merged_from``).  ZH and EN share the composed-unit graph, so the
    unit_id match is exact — every ZH gets precisely the EN words that
    produced it, never repeated and never split across ZH boundaries.

    When ``source_segments`` is None (only in tests), the function falls back
    to time-window overlap: each unresolved ZH claims the single EN cue with
    the largest overlap (EN-anchored exclusive assignment, so EN is not
    repeated).  ZH cues with no match become ZH-only Dialogues; EN
    segments/cues referenced by no ZH become EN-only Dialogues.

    Parameter order is ``(en_cues, zh_cues)`` to match the call site
    ``export_bilingual_ass(source_fmt, target_fmt, ...)``.
    """
    from light_models import seconds_to_ass

    font_name = _resolved_font(font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seg_by_unit: dict[str, Segment] = {s.unit_id: s for s in source_segments} if source_segments else {}

    def _en_text_for_zh(zc: SubtitleCue) -> str | None:
        """Join EN words from the segment(s) matching this ZH cue's unit_id + merged_from."""
        unit_ids = [zc.unit_id, *zc.merged_from]
        chunks: list[str] = []
        for uid in unit_ids:
            seg = seg_by_unit.get(uid)
            if seg and seg.words:
                chunks.append("".join(w.text for w in seg.words).strip())
        return " ".join(c for c in chunks if c) or None

    def _overlap(a: SubtitleCue, b: SubtitleCue) -> float:
        """Time overlap in seconds between two cues (0 if disjoint)."""
        return max(0.0, min(a.end, b.end) - max(a.start, b.start))

    # ── Resolve EN text for each ZH cue ──────────────────────────────────
    used_seg_units: set[str] = set()
    used_en_idx: set[int] = set()
    zh_en_text: list[str | None] = [None] * len(zh_cues)

    # Primary: segment words via unit_id (+ merged_from).
    unresolved: list[int] = []
    for zi, zc in enumerate(zh_cues):
        en_text = _en_text_for_zh(zc)
        if en_text is not None:
            zh_en_text[zi] = en_text
            used_seg_units.add(zc.unit_id)
            used_seg_units.update(zc.merged_from)
        else:
            unresolved.append(zi)

    # Fallback: EN-anchored exclusive assignment by largest overlap.  Process
    # candidate (overlap, zh, en) triples in descending overlap so each EN
    # cue goes to the ZH it fits best; ties broken by earliest ZH.  This keeps
    # EN from being repeated across ZH groups.
    if unresolved and en_cues:
        candidates: list[tuple[float, int, int]] = []
        for zi in unresolved:
            zc = zh_cues[zi]
            for ei, ec in enumerate(en_cues):
                if ei in used_en_idx:
                    continue
                ov = _overlap(zc, ec)
                if ov > 0:
                    candidates.append((ov, zi, ei))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        for _, zi, ei in candidates:
            if zh_en_text[zi] is None and ei not in used_en_idx:
                zh_en_text[zi] = " ".join(en_cues[ei].text.split())
                used_en_idx.add(ei)

    # ── Build groups: one per ZH cue + leftover EN-only ──────────────────
    # (zh_cue_or_None, en_text_or_None, start, end).  ZH groups use the ZH
    # cue's window; EN-only groups use the segment/cue window.
    groups: list[tuple[SubtitleCue | None, str | None, float, float]] = []
    for zi, zc in enumerate(zh_cues):
        groups.append((zc, zh_en_text[zi], zc.start, zc.end))

    if source_segments:
        for seg in source_segments:
            if seg.unit_id in used_seg_units or not seg.words:
                continue
            en_text = "".join(w.text for w in seg.words).strip()
            if en_text:
                groups.append((None, en_text, seg.start, seg.end))
    else:
        for ei, ec in enumerate(en_cues):
            if ei not in used_en_idx:
                groups.append((None, " ".join(ec.text.split()), ec.start, ec.end))

    # ── Write ASS ────────────────────────────────────────────────────────
    # ZH groups keep their full window (anchor).  EN-only groups are clamped
    # to avoid overlapping adjacent ZH groups, and dropped if fully covered.
    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write(ASS_V4_PLUS_STYLE_FORMAT)
        # One unified style: resolved font, white primary, bottom-aligned
        # (Alignment=2, MarginV=BILINGUAL_MARGIN_V).  EN uses a smaller font via
        # the {fs14} inline override.  Black outline (2px) + soft shadow keep the
        # white text legible on any background.
        f.write(bilingual_style_line(font_name))
        f.write("\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        groups.sort(key=lambda g: g[2])
        _GAP = 0.01
        clamped: list[tuple[SubtitleCue | None, str | None, float, float]] = []
        prev_end = -1.0
        for i, (zc, en_text, s, e) in enumerate(groups):
            if zc is None:
                if prev_end > 0:
                    s = max(s, prev_end + _GAP)
                if i + 1 < len(groups):
                    nxt_start = groups[i + 1][2]
                    if e > nxt_start - _GAP:
                        e = max(s + _GAP, nxt_start - _GAP)
                if s >= e:
                    continue
            clamped.append((zc, en_text, s, e))
            prev_end = e

        for zc, en_text, start_s, end_s in clamped:
            parts: list[str] = []
            if zc is not None:
                parts.append(zc.text.replace("\n", "\\N"))
            if en_text:
                parts.append(f"{{\\fs14}}{en_text}")
            text = "\\N".join(parts)
            f.write(f"Dialogue: 0,{seconds_to_ass(start_s)},{seconds_to_ass(end_s)},Bilingual,,0,0,0,,{text}\n")


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
            **({"merged_from": c.merged_from} if c.merged_from else {}),
        }
        for c in cues
    ]
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_annotation_ass(
    cues: list[SubtitleCue], annotations: dict[str, str], output_path: str, width_pct: int = 30, font: str | None = None
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

    font_name = _resolved_font(font)

    # ── Phase 3: write ASS ──
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
