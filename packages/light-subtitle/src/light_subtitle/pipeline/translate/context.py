"""Translation pipeline sub-step context."""

from __future__ import annotations

from dataclasses import dataclass, field

from light_models import SubtitleCue


@dataclass
class TranslateContext:
    """Shared state between translation sub-steps.

    Composed segments live on ``OrchestratorState.composed_segments`` so
    they are shared with the English source formatting path; this context
    only holds translation-specific outputs.
    """

    translated_cues: list[SubtitleCue] = field(default_factory=list)
    usage: dict | None = None
    usage_breakdown: dict[str, dict] = field(default_factory=dict)
