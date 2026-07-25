"""Bilingual ASS/VTT export — ZH-anchored groups with EN derived from segment words."""

from __future__ import annotations

from pathlib import Path

from light_core import logger
from light_models import Segment, SubtitleCue
from light_text import seconds_to_vtt

from ..style.box import (
    PILFontMeasurer,
    TextMeasurer,
    boxed_script_info,
    boxed_style_lines,
    build_bilingual_boxed_events,
)
from ..style.config import PLAY_RES_X, PLAY_RES_Y, SubtitleStyleConfig, play_res_for_frame
from ..style.fonts import ASS_V4_PLUS_STYLE_FORMAT, bilingual_ass_en_font_tag, bilingual_style_line
from .formats import _normalize_plain_subtitle_text, _resolved_font

# (zh_cue_or_None, en_text_or_None, start, end)
BilingualGroup = tuple[SubtitleCue | None, str | None, float, float]


def _build_bilingual_groups(
    en_cues: list[SubtitleCue],
    zh_cues: list[SubtitleCue],
    source_segments: list[Segment] | None = None,
) -> list[BilingualGroup]:
    """Build ZH-anchored bilingual cue groups shared by ASS and VTT export.

    Each group is ``(zh_cue | None, en_text | None, start, end)``.  ZH groups
    keep the ZH cue window; EN-only leftovers are clamped to avoid overlapping
    adjacent ZH groups.
    """
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
        return max(0.0, min(a.end, b.end) - max(a.start, b.start))

    used_seg_units: set[str] = set()
    used_en_idx: set[int] = set()
    zh_en_text: list[str | None] = [None] * len(zh_cues)

    unresolved: list[int] = []
    for zi, zc in enumerate(zh_cues):
        en_text = _en_text_for_zh(zc)
        if en_text is not None:
            zh_en_text[zi] = en_text
            used_seg_units.add(zc.unit_id)
            used_seg_units.update(zc.merged_from)
        else:
            unresolved.append(zi)

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

    groups: list[BilingualGroup] = []
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

    groups.sort(key=lambda g: g[2])
    _GAP = 0.01
    clamped: list[BilingualGroup] = []
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
    return clamped


# Single-line marker between ZH block (may contain ``\n``) and EN block.
# Cannot use ``\n\n`` — that is the WebVTT cue delimiter.
BILINGUAL_VTT_MARKER = "<<EN>>"


def _bilingual_plain_text(zc: SubtitleCue | None, en_text: str | None) -> str:
    """ZH block then EN block for text-based bilingual exports."""
    parts: list[str] = []
    if zc is not None:
        parts.append(_normalize_plain_subtitle_text(zc.text))
    if en_text:
        parts.append(en_text)
    if len(parts) == 2:
        return f"{parts[0]}\n{BILINGUAL_VTT_MARKER}\n{parts[1]}"
    return parts[0] if parts else ""


def export_bilingual_ass(
    en_cues: list[SubtitleCue],
    zh_cues: list[SubtitleCue],
    output_path: str,
    source_segments: list[Segment] | None = None,
    font: str | None = None,
    style: SubtitleStyleConfig | None = None,
    measurer: TextMeasurer | None = None,
    frame_size: tuple[int, int] | None = None,
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

    When *style* has ``box_enabled`` (the default), the exported file is
    self-contained: PlayRes matched to *frame_size* aspect (default 1920×1080),
    per-language rounded background boxes drawn as vector shapes sized by
    measuring the actual burn font (Pillow), and one text Dialogue per visual
    line.  Falls back to the plain outline style when the font file cannot be
    located (with a warning).  *measurer* injects a fake ``TextMeasurer`` for
    tests.  *frame_size* is ``(width, height)`` of the target video — required
    for non-16:9 frames so box drawings stay aligned with text under libass.
    """
    from light_text import seconds_to_ass

    from ..style.fonts import resolve_font_file  # local: tests patch the module attr

    font_name = _resolved_font(font)
    cfg = style or SubtitleStyleConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = _build_bilingual_groups(en_cues, zh_cues, source_segments)
    play_res = play_res_for_frame(*frame_size) if frame_size is not None else None

    if cfg.box_enabled:
        font_file = None if measurer is not None else resolve_font_file(font_name)
        if measurer is not None or font_file is not None:
            m = measurer if measurer is not None else PILFontMeasurer(font_file, family=font_name)
            play_x, play_y = play_res if play_res is not None else (PLAY_RES_X, PLAY_RES_Y)
            with open(output, "w", encoding="utf-8") as f:
                f.write(boxed_script_info(play_x, play_y))
                f.write("[V4+ Styles]\n")
                f.write(ASS_V4_PLUS_STYLE_FORMAT)
                for line in boxed_style_lines(font_name, cfg):
                    f.write(line + "\n")
                f.write("\n[Events]\n")
                f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
                for line in build_bilingual_boxed_events(groups, font_name, m, cfg, play_res=play_res):
                    f.write(line + "\n")
            return
        logger.warning(f"  未找到字体文件用于盒尺寸测量 ({font_name})，双语 ASS 回退为无盒样式")

    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write(ASS_V4_PLUS_STYLE_FORMAT)
        f.write(bilingual_style_line(font_name))
        f.write("\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for zc, en_text, start_s, end_s in groups:
            parts: list[str] = []
            if zc is not None:
                parts.append(zc.text.replace("\n", "\\N"))
            if en_text:
                parts.append(f"{bilingual_ass_en_font_tag()}{en_text}")
            text = "\\N".join(parts)
            f.write(f"Dialogue: 0,{seconds_to_ass(start_s)},{seconds_to_ass(end_s)},Bilingual,,0,0,0,,{text}\n")


def export_bilingual_vtt(
    en_cues: list[SubtitleCue],
    zh_cues: list[SubtitleCue],
    output_path: str,
    source_segments: list[Segment] | None = None,
) -> None:
    """Export bilingual WebVTT for web playback (ZH line, then EN line per cue)."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = _build_bilingual_groups(en_cues, zh_cues, source_segments)

    with open(output, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        cue_idx = 1
        for zc, en_text, start_s, end_s in groups:
            text = _bilingual_plain_text(zc, en_text)
            if not text:
                continue
            f.write(f"{cue_idx}\n")
            f.write(f"{seconds_to_vtt(start_s)} --> {seconds_to_vtt(end_s)}\n")
            f.write(f"{text}\n\n")
            cue_idx += 1
