"""Subtitle formatting — prepare → correct → repair.

Mental model::
    1. prepare (断句) — split text into viewer-friendly screens
    2. correct  (对时) — assign each screen a comfortable display window
    3. repair   (修边) — fix visual artifacts across screen boundaries

Note: `_align_to_units` has been removed. Translation now uses one-to-one
mapping (one segment → one cue), and timestamps are assigned at translate
time directly from source segments.  No post-hoc alignment needed.
"""

from . import layout, pace, polish


def run(cues, config) -> list:
    """Format cues into display-ready subtitles.

    Three-phase pipeline (顺序有因果关系，不可调换):

    1. prepare (断句) — split text into display screens, then merge
       adjacent cues that are too short to stand alone.  This merge
       MUST run before pace because pace's gap/CPS calculations work
       on the final cue structure.

    2. correct (对时) — duration fix, gap resolution, CPS enforcement,
       min-gap guard, reading padding.

    3. repair (修边) — cross-cue text repairs (orphan chars, leading
       punct, split words/names, conjunction absorption).  Runs after
       pace because it doesn't affect timing boundaries.
    """
    cues = layout.prepare(cues, config)  # 断句
    cues = pace.correct(cues, config)  # 对时
    cues = polish.repair(cues)  # 修边
    return cues
