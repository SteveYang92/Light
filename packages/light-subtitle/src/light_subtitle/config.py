"""Small per-stage configs for the subtitle capability package.

Each dataclass carries exactly the fields its pipeline stage reads —
defaults mirror :class:`light_cli.config.SubtitleConfig`.  The CLI
orchestration layer builds these from its unified ``SubtitleConfig``
via adapter methods; LLM access is NOT part of these configs (an
``OpenAIClient`` is passed explicitly, ``None`` meaning "LLM
unavailable" — the equivalent of the old empty-``llm_api_key`` gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LayoutConfig", "PlanConfig", "SegmentConfig", "TranslateConfig"]


@dataclass
class SegmentConfig:
    """Pause-based semantic segmentation (``light_subtitle.segment``)."""

    max_duration: float = 7.0
    max_chars_per_line: int = 42


@dataclass
class PlanConfig:
    """LLM cue-boundary planning (``light_subtitle.plan``)."""

    max_duration: float = 7.0
    min_duration: float = 0.8
    llm_temperature: float = 0.4


@dataclass
class TranslateConfig:
    """Translation, evaluation/refine, and the join pass."""

    target_lang: str | None = None
    glossary: dict[str, str] = field(default_factory=dict)
    content_summary: dict | None = None
    cps_limit: int = 9
    max_duration: float = 7.0
    evaluate_enabled: bool = False
    quality_threshold: float = 0.7
    max_refine_rounds: int = 1
    llm_temperature: float = 0.4


@dataclass
class LayoutConfig:
    """Display formatting — layout (断句) + pace (对时) + CPS compression."""

    cps_limit: int = 9
    cps_limit_en: int = 25
    max_lines: int = 2
    max_lines_zh: int = 1
    max_chars_per_line_zh: int = 40
    max_chars_per_line_en: int = 42
    min_duration: float = 0.8
    max_duration: float = 7.0
    reading_padding: float = 0.3
    optimize_entry_points: bool = False
    target_lang: str | None = None
    llm_temperature: float = 0.4
