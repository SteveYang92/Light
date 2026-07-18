"""Subtitle style configuration (extensible).

All geometry lives in a fixed 1920x1080 PlayRes design space; libass scales
proportionally to the actual video resolution at render time.  Fields are the
extension points — new subtitle surfaces (mono, annotation) add fields here
without changing the structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Fixed PlayRes design space (16:9).  See ``style.box`` for usage.
PLAY_RES_X = 1920
PLAY_RES_Y = 1080


@dataclass(frozen=True)
class SubtitleStyleConfig:
    """Bilingual boxed-subtitle theme; sizes in 1920x1080 PlayRes pixels."""

    box_enabled: bool = True
    """Master switch for rounded background boxes on bilingual subtitles."""

    bg_opacity: float = 0.70
    """Background box opacity (0..1); mapped to ASS alpha at export."""

    corner_radius_scale: float = 0.25
    """Corner radius as a fraction of the language's line height."""

    pad_h_scale: float = 0.45
    """Horizontal box padding as a fraction of the language's font size."""

    pad_v_scale: float = 0.12
    """Vertical box padding as a fraction of the language's font size."""

    block_gap: int = 2
    """Vertical gap between the ZH box and the EN box."""

    zh_font_size: int = 65
    """ZH font size in 1080p PlayRes pixels."""

    en_font_size: int = 39
    """EN font size in 1080p PlayRes pixels."""

    margin_v: int = 75
    """Bottom margin of the EN text block."""

    margin_lr: int = 40
    """Left/right safe margins; text wraps at PLAY_RES_X - 2 * margin_lr."""

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
