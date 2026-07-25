r"""Rounded background boxes for bilingual ASS subtitles.

Geometry is computed in a PlayRes design space (Y = 1080, X matches the video
aspect — see ``style.config.play_res_for_frame``).  Matching aspects keeps
libass X/Y scales equal so vector boxes and fonts stay aligned; a fixed 16:9
PlayRes on a non-16:9 frame (e.g. 3324×2160) compresses drawings horizontally
relative to text.  Each language block (ZH / EN) gets ONE rounded-rect drawing
event wrapping all its lines (block box, Netflix/Apple style), plus one text
Dialogue per visual line so vertical placement never depends on libass
line-pitch matching the measurer's.  Text events use absolute ``\pos`` rather
than margins: renderer-level margins (mpv/IINA fullscreen letterbox) only shift
margin-positioned text, which would detach it from the absolutely-placed box
drawings.

Text measurement uses Pillow against the actual font file, so box sizes wrap
the rendered glyphs (wrap effect).  A measurer with the same duck type can be
injected in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from light_core import logger
from light_text import seconds_to_ass

from .config import PLAY_RES_X, PLAY_RES_Y, SubtitleStyleConfig

# ── Text measurement ────────────────────────────────────


class TextMeasurer(Protocol):
    """Duck type for text measurement (PIL-backed in production, fake in tests)."""

    def line_width(self, text: str, size: int) -> float: ...
    def line_pitch(self, size: int) -> float: ...


class PILFontMeasurer:
    """Measure text with the actual burn font via Pillow.

    ASS Fontsize F is the font's line height (ascender + descender = F), not
    the em size — so the PIL face is loaded at ``F * em_scale`` where
    ``em_scale = em / (ascender + descender)`` of the font.  Line pitch equals
    F itself (times the configured spacing multiplier).

    TTC collections are scanned for the face matching *family* (falls back to
    face 0 when no family matches).
    """

    def __init__(self, font_path: str | Path, family: str | None = None) -> None:
        from PIL import ImageFont

        self._image_font = ImageFont
        self._path = str(font_path)
        self._index = self._pick_face_index(family)
        probe = self._image_font.truetype(self._path, 100, index=self._index)
        ascent, descent = probe.getmetrics()
        self._em_scale = 100.0 / (ascent + descent)
        self._cache: dict[int, object] = {}

    def _pick_face_index(self, family: str | None) -> int:
        if not family:
            return 0
        for index in range(16):
            try:
                font = self._image_font.truetype(self._path, 20, index=index)
            except OSError:
                break
            if font.getname()[0].lower() == family.lower():
                return index
        logger.warning(f"  字体集合中未匹配到 {family} 字面，使用 face 0: {self._path}")
        return 0

    def _font(self, size: int):
        em = max(1, round(size * self._em_scale))
        if em not in self._cache:
            self._cache[em] = self._image_font.truetype(self._path, em, index=self._index)
        return self._cache[em]

    def line_width(self, text: str, size: int) -> float:
        return float(self._font(size).getlength(text))

    def line_pitch(self, size: int) -> float:
        # ASS Fontsize IS the rendered line height (ascender + descender).
        return float(size)


# ── Wrapping ────────────────────────────────────────────

# Latin word (with inner apostrophe/hyphen), whitespace, or any single char (CJK).
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|\s+|.", re.S)


def wrap_line(text: str, measurer: TextMeasurer, size: int, max_width: float) -> list[str]:
    """Greedy wrap: CJK breaks per character, Latin breaks at word boundaries."""
    lines: list[str] = []
    current = ""
    for token in _TOKEN_RE.findall(text):
        if token.isspace():
            token = " "
        if not current and token == " ":
            continue
        candidate = current + token
        if current and measurer.line_width(candidate, size) > max_width and token != " ":
            lines.append(current.rstrip())
            current = "" if token == " " else token
            if token == " ":
                continue
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines or [""]


def wrap_block(text: str, measurer: TextMeasurer, size: int, max_width: float) -> list[str]:
    """Wrap each explicit line break separately, then concatenate."""
    out: list[str] = []
    for part in text.split("\n"):
        part = part.strip()
        if part:
            out.extend(wrap_line(part, measurer, size, max_width))
    return out


# ── Rounded rectangle drawing ───────────────────────────

# Cubic Bézier control distance for a quarter circle.
_KAPPA = 0.5522847498


def rounded_rect_path(x0: int, y0: int, x1: int, y1: int, r: int) -> str:
    """ASS drawing path for a rounded rectangle (clockwise from top edge)."""
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if r == 0:
        return f"m {x0} {y0} l {x1} {y0} l {x1} {y1} l {x0} {y1}"
    c = round(_KAPPA * r)
    return (
        f"m {x0 + r} {y0} "
        f"l {x1 - r} {y0} b {x1 - r + c} {y0} {x1} {y0 + r - c} {x1} {y0 + r} "
        f"l {x1} {y1 - r} b {x1} {y1 - r + c} {x1 - r + c} {y1} {x1 - r} {y1} "
        f"l {x0 + r} {y1} b {x0 + r - c} {y1} {x0} {y1 - r + c} {x0} {y1 - r} "
        f"l {x0} {y0 + r} b {x0} {y0 + r - c} {x0 + r - c} {y0} {x0 + r} {y0}"
    )


# ── Layout ──────────────────────────────────────────────


@dataclass
class _LangBlock:
    """One language's laid-out lines plus its enclosing box."""

    lines: list[str]
    style_name: str
    size: int
    pitch: float
    pad_h: float
    pad_v: float
    radius: int
    base_bottom_y: float = 0.0  # frame y of the bottom line's line-box bottom
    box: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1

    def lay_out(self, text_bottom_y: float, measurer: TextMeasurer, center_x: int) -> None:
        """Position the block so its bottom text line ends at *text_bottom_y* (frame coords)."""
        width = max(measurer.line_width(line, self.size) for line in self.lines)
        half = width / 2 + self.pad_h
        # libass bottom-aligns each line's line box (ascender+descender = one
        # line height) at the line's \pos y; the configured leading only
        # lives *between* lines, so the block is (n-1) pitches + 1 line
        # height tall — using n * pitch here would leave extra space at the
        # top and the text would sit low inside the box.
        height = (len(self.lines) - 1) * self.pitch + measurer.line_pitch(self.size) + 2 * self.pad_v
        y1 = text_bottom_y + self.pad_v
        self.box = (
            round(center_x - half),
            round(y1 - height),
            round(center_x + half),
            round(y1),
        )
        self.base_bottom_y = text_bottom_y


def _make_block(
    lines: list[str], style_name: str, size: int, measurer: TextMeasurer, config: SubtitleStyleConfig
) -> _LangBlock:
    return _LangBlock(
        lines=lines,
        style_name=style_name,
        size=size,
        pitch=measurer.line_pitch(size) * config.line_spacing,
        pad_h=config.pad_h_scale * size,
        pad_v=config.pad_v_scale * size,
        radius=round(config.corner_radius_scale * measurer.line_pitch(size)),
    )


def _alpha_tag(opacity: float) -> str:
    """ASS alpha override for *opacity* (0..1, 1 = opaque)."""
    return f"\\1a&H{round((1 - opacity) * 255):02X}&"


# ── Public API ──────────────────────────────────────────

# groups: (zh_cue | None, en_text | None, start, end) — same shape as export's BilingualGroup,
# declared structurally here so style/ never imports pipeline/ (circular import).
BilingualGroupLike = tuple[object | None, str | None, float, float]

BOX_STYLE_NAME = "Box"
ZH_STYLE_NAME = "BilingualZh"
EN_STYLE_NAME = "BilingualEn"


def boxed_style_lines(font_name: str, config: SubtitleStyleConfig) -> list[str]:
    """V4+ style lines for the boxed bilingual layout."""
    common = "&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,0,0"
    return [
        f"Style: {ZH_STYLE_NAME},{font_name},{config.zh_font_size},{common},2,10,10,0,1",
        f"Style: {EN_STYLE_NAME},{font_name},{config.en_font_size},{common},2,10,10,0,1",
        f"Style: {BOX_STYLE_NAME},{font_name},1,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,0,0,1,0,0,0,1",
    ]


def boxed_script_info(play_res_x: int = PLAY_RES_X, play_res_y: int = PLAY_RES_Y) -> str:
    """Script Info header pinning PlayRes (WrapStyle 2 = no re-wrap)."""
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
    )


def build_bilingual_boxed_events(
    groups: list[BilingualGroupLike],
    font_name: str,
    measurer: TextMeasurer,
    config: SubtitleStyleConfig,
    play_res: tuple[int, int] | None = None,
) -> list[str]:
    """Build Dialogue lines: one rounded box per language block + one text event per line.

    *play_res* is ``(PlayResX, PlayResY)``; defaults to 1920×1080.  Pass the
    value from ``play_res_for_frame(video_w, video_h)`` so boxes stay aligned
    on non-16:9 frames.
    """
    play_res_x, play_res_y = play_res if play_res is not None else (PLAY_RES_X, PLAY_RES_Y)
    center_x = play_res_x // 2
    max_text_width = play_res_x - 2 * config.margin_lr
    events: list[str] = []

    for zc, en_text, start_s, end_s in groups:
        start = seconds_to_ass(start_s)
        end = seconds_to_ass(end_s)

        blocks: list[_LangBlock] = []
        en_block: _LangBlock | None = None
        zh_block: _LangBlock | None = None

        if en_text and en_text.strip():
            en_lines = wrap_block(" ".join(en_text.split()), measurer, config.en_font_size, max_text_width)
            en_block = _make_block(en_lines, EN_STYLE_NAME, config.en_font_size, measurer, config)
            en_block.lay_out(play_res_y - config.margin_v, measurer, center_x)
            blocks.append(en_block)

        if zc is not None and getattr(zc, "text", "").strip():
            zh_lines = wrap_block(zc.text, measurer, config.zh_font_size, max_text_width)
            zh_block = _make_block(zh_lines, ZH_STYLE_NAME, config.zh_font_size, measurer, config)
            if en_block is not None:
                en_box_top = en_block.box[1]
                zh_text_bottom = en_box_top - config.block_gap - zh_block.pad_v
            else:
                zh_text_bottom = play_res_y - config.margin_v
            zh_block.lay_out(zh_text_bottom, measurer, center_x)
            blocks.append(zh_block)

        for block in blocks:
            x0, y0, x1, y1 = block.box
            drawing = (
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\c&H000000&{_alpha_tag(config.bg_opacity)}}}"
                f"{rounded_rect_path(x0, y0, x1, y1, block.radius)}"
            )
            events.append(f"Dialogue: 0,{start},{end},{BOX_STYLE_NAME},,0,0,0,,{drawing}")
            for k, line in enumerate(block.lines):
                # Absolute \pos (not MarginV): renderer-level margins, e.g.
                # mpv/IINA extending the frame into fullscreen letterbox bars,
                # shift margin-positioned text but not \pos text or drawings —
                # absolute placement keeps text glued to its box everywhere.
                # First reading line goes on top (smallest y).
                y = round(block.base_bottom_y - (len(block.lines) - 1 - k) * block.pitch)
                events.append(
                    f"Dialogue: 1,{start},{end},{block.style_name},,0,0,0,,{{\\an2\\pos({center_x},{y})}}{line}"
                )

    return events
