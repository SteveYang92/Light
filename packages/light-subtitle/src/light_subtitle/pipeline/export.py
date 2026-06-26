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


def export_bilingual_ass(en_cues: list[SubtitleCue], zh_cues: list[SubtitleCue], output_path: str) -> None:
    """Export bilingual ASS by merging EN/ZH into one Dialogue per semantic unit.

    Each bilingual pair is emitted as a **single** ASS Dialogue whose text is
    the ZH line(s) followed by the EN line(s), joined with ``\\N``.  Using one
    Dialogue (one style, bottom-aligned, ``MarginV=0``) lets ASS treat the
    whole block as one unit that grows upward as lines are added — so ZH and
    EN are always vertically adjacent with **no gap and no overlap**,
    regardless of how many lines each track takes.  This is the only layout
    that handles arbitrary line-count combinations cleanly, because ASS
    ``MarginV`` is static and cannot adapt to per-cue line counts.

    Pairing is by ``effective_unit_ids`` (head ``unit_id`` + ``merged_from``
    chain), not by positional index, so it copes with:

    * EN fan-out — one composed English unit may split into several display
      cues (karaoke-style); all sub-cues share the parent unit_id and are
      grouped with the one matching ZH cue.  Their texts are concatenated in
      time order as the EN portion.
    * ZH display-merge — one ZH cue whose ``merged_from`` lists absorbed unit
      ids; it groups with every EN cue whose unit_id intersects.
    * combined fan-out + merge.
    * unmatched cues (no shared unit ids) — emitted as a solo ZH or EN line.

    Note: parameter order is ``(en_cues, zh_cues)`` to match the call site
    ``export_bilingual_ass(source_fmt, target_fmt, ...)`` where source is
    the EN track and target is the ZH translation.
    """
    from light_models import seconds_to_ass
    from light_models.cue_utils import effective_unit_ids

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Group cues by shared effective unit ids ───────────────────────────
    # A group = all display cues (EN + ZH) that belong to one semantic unit.
    # EN fan-out sub-cues share the parent unit_id → same group.  A ZH
    # display-merge cue intersects several EN cues via merged_from → all in
    # one group.  We group by "any unit-id overlap" which is correct because
    # unit ids are unique per composed unit and merges only chain adjacent
    # ids, so overlap is an equivalence relation in practice.
    en_sets = [effective_unit_ids(c) for c in en_cues]
    zh_sets = [effective_unit_ids(c) for c in zh_cues]

    # Assign each EN cue to a group keyed by the first ZH cue it intersects.
    # EN cues that match no ZH cue get their own group; ZH cues with no EN
    # match also get their own group.  Order groups by ZH cue order first
    # (so the file reads top-to-bottom in ZH/semantic order), then append
    # leftover EN-only groups at the end in EN order.
    en_grouped: list[bool] = [False] * len(en_cues)
    groups: list[tuple[list[SubtitleCue], list[SubtitleCue]]] = []  # (zh, en)
    for zi, zc in enumerate(zh_cues):
        zs = zh_sets[zi]
        matching_en = [ec for ei, ec in enumerate(en_cues) if not en_grouped[ei] and en_sets[ei] & zs]
        for ei in range(len(en_cues)):
            if not en_grouped[ei] and en_sets[ei] & zs:
                en_grouped[ei] = True
        groups.append(([zc], matching_en))
    # Leftover EN cues with no ZH match → EN-only groups.
    for ei, ec in enumerate(en_cues):
        if not en_grouped[ei]:
            groups.append(([], [ec]))

    # ── Write one Dialogue per group ──────────────────────────────────────
    # ZH lines first (top), EN lines after (bottom), joined with \N.  Within
    # each track, cues are already in time order from the pipeline; we sort
    # the EN members of a group by start to be safe for fan-out sub-cues.
    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Italic, Alignment\n")
        # One unified style for the merged bilingual block.  Both languages
        # share PingFangSC-Regular + white primary so the pair reads as one
        # consistent subtitle (movie convention: translation on top, smaller
        # original below).  Bottom-aligned (Alignment=2), MarginV=0 → ASS
        # grows the whole block upward as lines are added, so ZH and EN stay
        # vertically adjacent with no gap and no overlap for any line count.
        # A single Fontsize is required by ASS per style; the within-block
        # size contrast (ZH larger, EN smaller) is achieved with the ``fs``
        # override tag on the EN portion of each Dialogue line.
        f.write("Style: Bilingual,PingFangSC-Regular,20,&H00FFFFFF,&H00000000,0,0,2\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for zh_members, en_members in groups:
            # Time window = union of all cues in the group, so the whole
            # bilingual block is shown for the full span of its components.
            all_cues = zh_members + en_members
            if not all_cues:
                continue
            start_s = min(c.start for c in all_cues)
            end_s = max(c.end for c in all_cues)
            start = seconds_to_ass(start_s)
            end = seconds_to_ass(end_s)

            parts: list[str] = []
            for zc in zh_members:
                parts.append(zc.text.replace("\n", "\\N"))
            # EN portion: smaller font via {fs14}, sorted by start for fan-out order.
            if en_members:
                en_sorted = sorted(en_members, key=lambda c: c.start)
                # Force EN to a single display line: drop all newlines from every
                # EN cue (intra-cue \n and any inter-cue separators) and join the
                # per-cue texts with a space.  Fan-out sub-cues concatenate in
                # time order on one line.  ASS will word-wrap if the line is wider
                # than the screen, but the logical cue text stays one line so the
                # bilingual block is at most ZH lines + 1 EN line.
                en_words: list[str] = []
                for ec in en_sorted:
                    en_words.extend(ec.text.split())
                en_text = " ".join(en_words)
                parts.append(f"{{\\fs14}}{en_text}")
            text = "\\N".join(parts)

            f.write(f"Dialogue: 0,{start},{end},Bilingual,,0,0,0,,{text}\n")


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
