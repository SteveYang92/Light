"""RunModel — fold progress events into a renderable run state.

Shared by CLI/Web/GUI renderers.  No I/O, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .events import RUN_STAGES, ProgressEvent, RunEvent, RunKind, SegmentRef, StageStatus

# Implicit segment for pipeline stages emitted without a SegmentRef
# (single-segment runs): displays as 1/1 with an empty tag.
_IMPLICIT_SEGMENT = SegmentRef(index=0, total=1, tag="")


@dataclass(frozen=True)
class StageView:
    """Latest known state of one stage (run-level or within a segment)."""

    stage: str
    status: StageStatus
    progress: float
    message: str
    ts: float
    started_ts: float | None = None  # ts of the `started` event (elapsed-time display)


@dataclass(frozen=True)
class SegmentView:
    """One segment and its stage states (first-seen order)."""

    ref: SegmentRef
    stages: tuple[StageView, ...]


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable render view of the run so far."""

    run_kind: RunKind | None
    run_payload: dict
    run_stages: tuple[StageView, ...]  # download/split/merge/done, first-seen order
    segments: tuple[SegmentView, ...]  # first-seen order


class RunModel:
    """Consume ProgressEvent/RunEvent streams into ordered stage tables.

    Stage order within each table is first-seen; later events for the same
    stage update its status/progress/message in place.
    """

    def __init__(self) -> None:
        self._run_kind: RunKind | None = None
        self._run_payload: dict = {}
        self._run_stages: dict[str, StageView] = {}
        self._segments: dict[SegmentRef, dict[str, StageView]] = {}

    def apply(self, event: ProgressEvent | RunEvent) -> None:
        if isinstance(event, RunEvent):
            self._run_kind = event.kind
            self._run_payload = dict(event.payload)
            return
        if event.segment is None and event.stage in RUN_STAGES:
            previous = self._run_stages.get(event.stage)
            self._run_stages[event.stage] = self._to_view(event, previous)
            return
        segment = event.segment if event.segment is not None else _IMPLICIT_SEGMENT
        table = self._segments.setdefault(segment, {})
        previous = table.get(event.stage)
        table[event.stage] = self._to_view(event, previous)

    @staticmethod
    def _to_view(event: ProgressEvent, previous: StageView | None) -> StageView:
        started_ts = event.ts if event.status == StageStatus.started else (previous.started_ts if previous else None)
        return StageView(
            stage=event.stage,
            status=event.status,
            progress=event.progress,
            message=event.message,
            ts=event.ts,
            started_ts=started_ts,
        )

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            run_kind=self._run_kind,
            run_payload=dict(self._run_payload),
            run_stages=tuple(self._run_stages.values()),
            segments=tuple(
                SegmentView(ref=ref, stages=tuple(stages.values())) for ref, stages in self._segments.items()
            ),
        )
