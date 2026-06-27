"""Font resolution and ASS style patching for subtitle export and pack."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import logger

# CJK-first chain; Latin fallback at the end.
DEFAULT_FALLBACKS: tuple[str, ...] = (
    "PingFang SC",
    "PingFangSC-Regular",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "Arial",
)

DEFAULT_FONT = "PingFang SC"

# ASS V4+ style header shared by bilingual and annotation exports.
ASS_V4_PLUS_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
    " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
    " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
    " Alignment, MarginL, MarginR, MarginV, Encoding\n"
)


@dataclass(frozen=True)
class FontConfig:
    """Preferred font plus built-in fallbacks for system resolution."""

    primary: str = DEFAULT_FONT
    fallbacks: tuple[str, ...] = DEFAULT_FALLBACKS


def _normalize_font_name(name: str) -> str:
    """Lowercase and strip spaces/hyphens for fuzzy family comparison."""
    return re.sub(r"[\s\-_]+", "", name.lower())


def _parse_fc_family(raw: str) -> str | None:
    """Extract a single family name from ``fc-match`` output.

    fontconfig may return comma-separated aliases, e.g.
    ``PingFang SC,蘋方-簡,苹方-简``.  ASS ``Style:`` lines also use commas as
    field separators, so only the first family must be used as Fontname.
    """
    line = raw.strip()
    if not line:
        return None
    return line.split(",")[0].strip() or None


def _fc_match_family(candidate: str) -> str | None:
    """Return font family from ``fc-match`` for *candidate*, or None on failure."""
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{family}\n", candidate],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _parse_fc_family(result.stdout)


def _font_matches(candidate: str, matched_family: str) -> bool:
    """True when *matched_family* is a reasonable match for *candidate*."""
    cand_norm = _normalize_font_name(candidate)
    match_norm = _normalize_font_name(matched_family)
    if not cand_norm or not match_norm:
        return False
    return cand_norm in match_norm or match_norm in cand_norm


def _candidate_chain(config: FontConfig) -> tuple[str, ...]:
    """Deduplicated ordered list: primary first, then fallbacks."""
    seen: set[str] = set()
    chain: list[str] = []
    for name in (config.primary, *config.fallbacks):
        key = _normalize_font_name(name)
        if key in seen:
            continue
        seen.add(key)
        chain.append(name)
    return tuple(chain)


def resolve_font(config: FontConfig | None = None) -> str:
    """Pick the first available font from *config*'s candidate chain.

    Uses ``fc-match`` when available (Linux / Homebrew fontconfig).  Without
    it (typical macOS without fontconfig), returns *primary* and relies on
    libass CoreText fontselect at render time.
    """
    cfg = config or FontConfig()
    chain = _candidate_chain(cfg)

    if shutil.which("fc-match") is None:
        return cfg.primary

    for candidate in chain:
        matched = _fc_match_family(candidate)
        if matched and _font_matches(candidate, matched):
            if _normalize_font_name(matched) != _normalize_font_name(cfg.primary):
                logger.info(f"  字体回退: {cfg.primary} → {matched}")
            return matched

    logger.warning(f"  未在系统中匹配到字体链，使用首选: {cfg.primary}")
    return cfg.primary


def patch_ass_styles(
    text: str,
    font_name: str,
    style_names: set[str] | None = None,
) -> str:
    """Replace Fontname (field 2) on ``Style:`` lines in ASS *text*.

    When *style_names* is None, all ``Style:`` lines are patched.  Dialogue
    lines and inline override tags (e.g. ``{\\fs14}``) are left unchanged.
    """
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.startswith("Style:"):
            out_lines.append(line)
            continue
        # Strip trailing newline for parsing; re-append later.
        newline = ""
        body = line
        if body.endswith("\r\n"):
            newline = "\r\n"
            body = body[:-2]
        elif body.endswith("\n"):
            newline = "\n"
            body = body[:-1]

        fields = body.split(",", 22)
        if len(fields) < 2:
            out_lines.append(line)
            continue
        style_name = fields[0].removeprefix("Style:").strip()
        if style_names is not None and style_name not in style_names:
            out_lines.append(line)
            continue
        fields[1] = font_name
        out_lines.append(",".join(fields) + newline)
    return "".join(out_lines)


def write_patched_ass(src: Path, font_name: str, dst: Path) -> None:
    """Read *src* ASS, patch Style Fontname fields, write to *dst*."""
    dst.write_text(patch_ass_styles(src.read_text(encoding="utf-8"), font_name), encoding="utf-8")


def bilingual_style_line(font_name: str) -> str:
    """Return the Bilingual ASS style line for *font_name*."""
    return (
        f"Style: Bilingual,{font_name},20,&H00FFFFFF,&H00FFFFFF,"
        "&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,0,1\n"
    )


def annotation_style_line(font_name: str, right_margin: int) -> str:
    """Return the Annotation ASS style line for *font_name*."""
    return (
        f"Style: Annotation,{font_name},40,&H00FFFFFF,&H00000000,"
        "&H00000000,&H00000000,-1,0,0,0,100,100,0,0,"
        f"1,3,2,7,10,{right_margin},10,1\n"
    )


def default_style_line(font_name: str) -> str:
    """Return a minimal mono-language Default style line (legacy V4 header)."""
    return f"Style: Default,{font_name},20,&H00FFFFFF,&H00000000,0,0,2\n"
