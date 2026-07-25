"""Subtitle display formatting — layout (断句) → pace (对时), plus text post-processing.

Mental model::
    1. prepare (断句) — split text into viewer-friendly screens
    2. correct  (对时) — assign each screen a comfortable display window

Note: `_align_to_units` has been removed. Translation now uses one-to-one
mapping (one segment → one cue), and timestamps are assigned at translate
time directly from source segments.  No post-hoc alignment needed.

Display-time punctuation stripping lives in :mod:`.strip_punct`.
"""

from __future__ import annotations

from light_llm.client import OpenAIClient

from ..config import LayoutConfig
from . import layout, pace


def run(cues, config: LayoutConfig, *, llm: OpenAIClient | None = None, transcript_words=None) -> tuple[list, dict]:
    """Format cues into display-ready subtitles.

    Two-phase pipeline (顺序有因果关系，不可调换):

    1. prepare (断句) — split text into display screens, then merge
       adjacent cues that are too short to stand alone.  This merge
       MUST run before pace because pace's gap/CPS calculations work
       on the final cue structure.

    2. correct (对时) — duration fix, gap resolution, CPS enforcement
       (borrow time, then compress over-limit translations via LLM),
       min-gap guard, reading padding.

    *transcript_words* feeds pace's entry-point optimization.  *llm*
    enables the CPS compression pass (``None`` disables it).  Returns
    ``(cues, usage)`` — usage collects compression token usage (empty
    dict when LLM compression did not run).
    """
    cues = layout.prepare(cues, config)  # 断句
    cues, usage = pace.correct(cues, config, llm=llm, transcript_words=transcript_words)  # 对时
    return cues, usage
