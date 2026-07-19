"""Tests for the reporting layer — RunModel state machine and reporter adapters."""

from __future__ import annotations

from light_subtitle.reporting import (
    STAGE_DOWNLOAD,
    CallableReporter,
    CompositeReporter,
    ProgressEvent,
    RunEvent,
    RunKind,
    RunModel,
    SegmentRef,
    StageStatus,
    as_reporter,
)


def _ev(stage: str, status: StageStatus, progress: float, message: str = "", segment: SegmentRef | None = None):
    return ProgressEvent(stage=stage, status=status, progress=progress, message=message, segment=segment, ts=1.0)


class TestSingleSegmentLifecycle:
    def test_full_lifecycle_snapshot(self):
        model = RunModel()
        model.apply(RunEvent(RunKind.started, {"slug": "demo"}, ts=0.0))
        model.apply(_ev(STAGE_DOWNLOAD, StageStatus.started, 0.0, "下载中…"))
        model.apply(_ev(STAGE_DOWNLOAD, StageStatus.finished, 1.0, "下载完成"))
        model.apply(_ev("asr", StageStatus.started, 0.0, "提取音频中..."))
        model.apply(_ev("asr", StageStatus.finished, 1.0, "ASR 完成"))
        model.apply(_ev("translate", StageStatus.finished, 1.0, "翻译完成"))

        snap = model.snapshot()
        assert snap.run_kind == RunKind.started
        assert snap.run_payload == {"slug": "demo"}

        # Run-level stages in first-seen order; segment=None pipeline stages
        # normalize into the implicit 1/1 segment.
        assert [s.stage for s in snap.run_stages] == [STAGE_DOWNLOAD]
        assert snap.run_stages[0].status == StageStatus.finished
        assert snap.run_stages[0].progress == 1.0

        assert len(snap.segments) == 1
        seg = snap.segments[0]
        assert seg.ref == SegmentRef(index=0, total=1, tag="")
        assert [s.stage for s in seg.stages] == ["asr", "translate"]
        assert seg.stages[0].status == StageStatus.finished
        assert seg.stages[0].progress == 1.0
        assert seg.stages[0].message == "ASR 完成"


class TestMultiSegmentInterleaved:
    def test_segments_tracked_independently(self):
        model = RunModel()
        s1 = SegmentRef(index=0, total=2, tag="seg1")
        s2 = SegmentRef(index=1, total=2, tag="seg2")

        # Interleaved: seg1 starts ASR, seg2 starts ASR, seg1 finishes ASR,
        # seg2 starts translate, seg1 finishes translate, seg2 finishes ASR.
        model.apply(_ev("asr", StageStatus.started, 0.0, segment=s1))
        model.apply(_ev("asr", StageStatus.started, 0.0, segment=s2))
        model.apply(_ev("asr", StageStatus.finished, 1.0, segment=s1))
        model.apply(_ev("translate", StageStatus.started, 0.0, segment=s2))
        model.apply(_ev("translate", StageStatus.finished, 1.0, segment=s1))
        model.apply(_ev("asr", StageStatus.finished, 1.0, segment=s2))

        snap = model.snapshot()
        assert [seg.ref for seg in snap.segments] == [s1, s2]  # first-seen order

        seg1, seg2 = snap.segments
        assert [(s.stage, s.status) for s in seg1.stages] == [
            ("asr", StageStatus.finished),
            ("translate", StageStatus.finished),
        ]
        assert [(s.stage, s.status) for s in seg2.stages] == [
            ("asr", StageStatus.finished),
            ("translate", StageStatus.started),
        ]
        # seg2's translate is mid-flight: progress preserved, not finished.
        assert seg2.stages[1].progress == 0.0


class TestFailedAndSkipped:
    def test_failed_and_skipped_surface_in_snapshot(self):
        model = RunModel()
        model.apply(_ev("asr", StageStatus.started, 0.0))
        model.apply(_ev("asr", StageStatus.failed, 0.0, "RuntimeError: boom"))
        model.apply(_ev("translate", StageStatus.skipped, 1.0, "无需翻译"))

        seg = model.snapshot().segments[0]
        assert seg.stages[0].status == StageStatus.failed
        assert seg.stages[0].message == "RuntimeError: boom"
        assert seg.stages[1].status == StageStatus.skipped
        assert seg.stages[1].progress == 1.0

    def test_run_failed_kind(self):
        model = RunModel()
        model.apply(RunEvent(RunKind.failed, {"error": "boom"}, ts=0.0))
        snap = model.snapshot()
        assert snap.run_kind == RunKind.failed
        assert snap.run_payload == {"error": "boom"}


class TestCallableReporter:
    def test_forwards_plain_triples_for_all_statuses(self):
        calls: list[tuple] = []
        reporter = CallableReporter(lambda s, p, m: calls.append((s, p, m)))

        reporter.emit(_ev("asr", StageStatus.started, 0.0, "提取音频中...", segment=SegmentRef(0, 2, "seg1")))
        reporter.emit(_ev("asr", StageStatus.progress, 0.5, "half"))
        reporter.emit(_ev("asr", StageStatus.finished, 1.0, "done"))
        reporter.emit(_ev("asr", StageStatus.failed, 0.5, "boom"))
        reporter.emit(_ev("translate", StageStatus.skipped, 1.0, "skip"))

        # Every ProgressEvent degrades to (stage, progress, message);
        # segment and status are dropped, progress passes through as-is.
        assert calls == [
            ("asr", 0.0, "提取音频中..."),
            ("asr", 0.5, "half"),
            ("asr", 1.0, "done"),
            ("asr", 0.5, "boom"),
            ("translate", 1.0, "skip"),
        ]

    def test_drops_run_events(self):
        calls: list[tuple] = []
        reporter = CallableReporter(lambda s, p, m: calls.append((s, p, m)))
        reporter.emit(RunEvent(RunKind.started, {"slug": "x"}, ts=0.0))
        assert calls == []


class TestAsReporter:
    def test_none_returns_noop_reporter(self):
        as_reporter(None).emit(_ev("asr", StageStatus.started, 0.0))  # must not raise

    def test_reporter_passes_through(self):
        reporter = CompositeReporter()
        assert as_reporter(reporter) is reporter

    def test_callable_wrapped(self):
        calls: list[tuple] = []
        reporter = as_reporter(lambda s, p, m: calls.append((s, p, m)))
        assert isinstance(reporter, CallableReporter)
        reporter.emit(_ev("asr", StageStatus.finished, 1.0, "ok"))
        assert calls == [("asr", 1.0, "ok")]

    def test_composite_fans_out(self):
        a: list[tuple] = []
        b: list[tuple] = []
        composite = CompositeReporter(
            CallableReporter(lambda s, p, m: a.append((s, p))),
            CallableReporter(lambda s, p, m: b.append((s, p))),
        )
        composite.emit(_ev("asr", StageStatus.finished, 1.0))
        assert a == b == [("asr", 1.0)]
