"""Subtitle style configuration (extensible).

Geometry uses a PlayRes design space with height pinned at 1080 and width
chosen to match the video frame's aspect ratio (default 1920 = 16:9).  libass
scales PlayRes → frame isotropically when aspects match; a mismatched aspect
(e.g. 1920×1080 PlayRes on a 3324×2160 frame) stretches drawings and fonts
differently so background boxes no longer wrap the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Default PlayRes design space (16:9).  Non-16:9 videos override PlayResX via
# ``play_res_for_frame``.  See ``style.box`` for usage.
PLAY_RES_X = 1920
PLAY_RES_Y = 1080


def play_res_for_frame(width: int, height: int) -> tuple[int, int]:
    """Return ``(PlayResX, PlayResY)`` matching *width*/*height* aspect.

    PlayResY stays at :data:`PLAY_RES_Y` so font sizes keep their 1080p design
    meaning; PlayResX scales so X/Y scale factors to the frame are equal.
    """
    if width <= 0 or height <= 0:
        return PLAY_RES_X, PLAY_RES_Y
    play_x = max(1, round(PLAY_RES_Y * width / height))
    return play_x, PLAY_RES_Y


@dataclass(frozen=True)
class SubtitleStyleConfig:
    """Bilingual boxed-subtitle theme; sizes in PlayRes pixels (Y = 1080)."""

    box_enabled: bool = True
    """Master switch for rounded background boxes on bilingual subtitles."""

    bg_opacity: float = 0.70
    """Background box opacity (0..1); mapped to ASS alpha at export."""

    corner_radius_scale: float = 0.25
    """Corner radius as a fraction of the language's line height."""

    pad_h_scale: float = 0.70
    """Horizontal box padding as a fraction of the language's font size."""

    pad_v_scale: float = 0.12
    """Vertical box padding as a fraction of the language's font size."""

    block_gap: int = 2
    """Vertical gap between the ZH box and the EN box."""

    zh_font_size: int = 65
    """ZH font size in PlayRes pixels (Y = 1080 design space)."""

    en_font_size: int = 39
    """EN font size in PlayRes pixels (Y = 1080 design space)."""

    margin_v: int = 75
    """Bottom margin of the EN text block."""

    margin_lr: int = 40
    """Left/right safe margins; text wraps at PlayResX - 2 * margin_lr."""

    line_spacing: float = 1.12
    """Multiplier on the font's line height (ASS Fontsize) when stacking lines."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form extension bucket for future style surfaces."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleStyleConfig:
        """Build a config from a partial dict; unknown keys go to *extra*."""
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs)
        if extra:
            object.__setattr__(cfg, "extra", {**cfg.extra, **extra})
        return cfg

    @classmethod
    def load_yaml(cls, path: str | Path) -> SubtitleStyleConfig:
        """Load style overrides from a YAML file (flat mapping of field names)."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"样式配置必须是 YAML mapping: {path}")
        return cls.from_dict(data)
