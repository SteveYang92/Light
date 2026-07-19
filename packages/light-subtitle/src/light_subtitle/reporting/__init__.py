"""Renderer-agnostic progress reporting — events, run model, reporter adapters.

Zero-dependency shared layer (no rich/typer at import time): CLI, Web SSE,
and future GUI renderers all consume :class:`ProgressEvent`/:class:`RunEvent`
streams.  :class:`PlainReporter` (non-TTY) and :class:`RichReporter` (TTY,
lazy rich import) are the built-in renderers.
"""

from .events import (
    RUN_STAGES,
    STAGE_DONE,
    STAGE_DOWNLOAD,
    STAGE_MERGE,
    STAGE_SPLIT,
    ProgressEvent,
    RunEvent,
    RunKind,
    SegmentRef,
    StageStatus,
)
from .labels import STAGE_LABELS, STATUS_ICONS, stage_label
from .model import RunModel, RunSnapshot, SegmentView, StageView
from .reporter import CallableReporter, CompositeReporter, Reporter, as_reporter
from .text import PlainReporter

__all__ = [
    "RUN_STAGES",
    "STAGE_DONE",
    "STAGE_DOWNLOAD",
    "STAGE_LABELS",
    "STAGE_MERGE",
    "STAGE_SPLIT",
    "STATUS_ICONS",
    "CallableReporter",
    "CompositeReporter",
    "PlainReporter",
    "ProgressEvent",
    "Reporter",
    "RunEvent",
    "RunKind",
    "RunModel",
    "RunSnapshot",
    "SegmentRef",
    "SegmentView",
    "StageStatus",
    "StageView",
    "as_reporter",
    "stage_label",
]
