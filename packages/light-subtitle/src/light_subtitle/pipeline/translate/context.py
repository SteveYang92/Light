"""Translation pipeline sub-step context."""

from __future__ import annotations

from dataclasses import dataclass, field

from light_models import Segment, SubtitleCue


@dataclass
class TranslateContext:
    """Shared state between translation sub-steps."""

    translation_segments: list[Segment] = field(default_factory=list)
    translated_cues: list[SubtitleCue] = field(default_factory=list)
    usage: dict | None = None
