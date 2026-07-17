"""Subtitle formatting — prepare → correct.

Mental model::
    1. prepare (断句) — split text into viewer-friendly screens
    2. correct  (对时) — assign each screen a comfortable display window

Note: `_align_to_units` has been removed. Translation now uses one-to-one
mapping (one segment → one cue), and timestamps are assigned at translate
time directly from source segments.  No post-hoc alignment needed.
"""

from . import layout, pace


def run(cues, config, usage_out: dict | None = None) -> list:
    """Format cues into display-ready subtitles.

    Two-phase pipeline (顺序有因果关系，不可调换):

    1. prepare (断句) — split text into display screens, then merge
       adjacent cues that are too short to stand alone.  This merge
       MUST run before pace because pace's gap/CPS calculations work
       on the final cue structure.

    2. correct (对时) — duration fix, gap resolution, CPS enforcement
       (borrow time, then compress over-limit translations via LLM),
       min-gap guard, reading padding.  *usage_out* collects compression
       token usage when provided.
    """
    cues = layout.prepare(cues, config)  # 断句
    cues = pace.correct(cues, config, usage_out)  # 对时
    return cues
