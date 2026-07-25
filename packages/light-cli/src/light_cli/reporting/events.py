"""Progress event types for pipeline reporting (renderer-agnostic).

Pure data — no I/O, stdlib only (NO rich/typer imports allowed here).
Renderers (Plain/Rich/Web SSE/GUI) consume these; producers (runner,
orchestrator, steps) emit them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

# ── Run-level stage names ─────────────────────────────────────────────────────
#
# Pipeline-step stages (asr/correct/punct/segment/context/compose/translate/
# annotate/format) live in steps/progress.py; these are the stages owned by
# the runner itself.  Values are an external contract (backend SSE).

STAGE_DOWNLOAD = "download"
STAGE_SPLIT = "split"
STAGE_MERGE = "merge"
STAGE_DONE = "done"

RUN_STAGES = frozenset({STAGE_DOWNLOAD, STAGE_SPLIT, STAGE_MERGE, STAGE_DONE})


@dataclass(frozen=True)
class SegmentRef:
    """Identity of one video segment within a split run.

    ``index`` is the 0-based ordinal (0..total-1); ``tag`` is the display
    label derived from the segment directory name via
    ``video_split.segment_tag`` (e.g. ``"seg2"`` or ``"chunk_1"``).
    """

    index: int
    total: int
    tag: str


class StageStatus(StrEnum):
    """Lifecycle status of one stage within a run or a segment."""

    started = "started"
    progress = "progress"
    finished = "finished"
    failed = "failed"
    skipped = "skipped"


@dataclass(frozen=True)
class ProgressEvent:
    """One stage transition within a run.

    ``segment`` is None for run-level stages (download/split/merge/done);
    pipeline stages carry the emitting segment (None on single-segment
    runs — RunModel normalizes those to an implicit 1/1 segment).
    """

    stage: str
    status: StageStatus
    progress: float
    message: str
    segment: SegmentRef | None = None
    ts: float = field(default_factory=time.time)


class RunKind(StrEnum):
    """Lifecycle kind of the whole run."""

    started = "started"
    finished = "finished"
    failed = "failed"


@dataclass(frozen=True)
class RunEvent:
    """Run lifecycle event (start/finish/failure of the whole run)."""

    kind: RunKind
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
