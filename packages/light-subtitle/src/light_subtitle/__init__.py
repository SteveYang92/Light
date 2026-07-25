"""light-subtitle — subtitle capability package.

Language-aware segmentation and line-breaking, display styling (fonts,
themes, boxed ASS rendering), cue construction, subtitle/transcript
export, the LLM pipeline stages (cue planning, translation, display
formatting, context prep, annotation), and the bundled LLM prompt
templates used by the subtitle pipeline:

- :mod:`light_subtitle.config` — small per-stage configs (``SegmentConfig`` …).
- :mod:`light_subtitle.artifacts` — subtitle-domain artifact paths and JSON (de)serialization.
- :mod:`light_subtitle.segment` — split word streams into semantic segments.
- :mod:`light_subtitle.language` — English/CJK break finders and line splitters.
- :mod:`light_subtitle.style` — ``SubtitleStyleConfig``, font resolution, box geometry.
- :mod:`light_subtitle.cue_builder` — build source ``SubtitleCue`` objects from segments.
- :mod:`light_subtitle.export` — SRT/VTT/ASS/JSON writers (mono, bilingual, annotations, transcript).
- :mod:`light_subtitle.plan` — LLM cue-boundary planning (+ deterministic fallback).
- :mod:`light_subtitle.translate` — translation, alignment check, evaluate/refine, join, caches.
- :mod:`light_subtitle.subtitle` — display formatting (layout/pace/compress) and punctuation stripping.
- :mod:`light_subtitle.context_prep` — glossary + content-summary extraction before translation.
- :mod:`light_subtitle.annotate` — LLM-generated secondary subtitle annotations.
- :mod:`light_subtitle.prompts` — bundled ``.j2`` prompt templates (``render_prompt``).
"""

from __future__ import annotations
