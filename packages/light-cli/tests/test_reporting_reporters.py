"""Tests for PlainReporter and RichReporter (model-driven, no pixel checks)."""

from __future__ import annotations

import io
import time

from light_cli.reporting import (
    STAGE_DOWNLOAD,
    PlainReporter,
    ProgressEvent,
    RunEvent,
    RunKind,
    SegmentRef,
    StageStatus,
)
from light_cli.reporting.labels import STAGE_LABELS
from light_cli.steps import progress as steps_progress


def _ev(stage: str, status: StageStatus, progress: float, message: str = "", segment: SegmentRef | None = None):
    return ProgressEvent(stage=stage, status=status, progress=progress, message=message, segment=segment, ts=1.0)


def _drive_plain(events) -> str:
    stream = io.StringIO()
    reporter = PlainReporter(stream=stream)
    for event in events:
        reporter.emit(event)
    return stream.getvalue()


class TestPlainReporter:
    def test_transition_lines_and_summary(self):
        out = _drive_plain(
            [
                RunEvent(RunKind.started, {"slug": "demo"}, ts=0.0),
                _ev(STAGE_DOWNLOAD, StageStatus.started, 0.0, ""),
                _ev(STAGE_DOWNLOAD, StageStatus.finished, 1.0, "下载完成"),
                _ev("translate", StageStatus.started, 0.0, "翻译中..."),
                _ev("translate", StageStatus.finished, 1.0, "翻译完成 (561 条)"),
                RunEvent(RunKind.finished, {"slug": "demo", "log": "output/demo/pipeline_x.log"}, ts=1.0),
            ]
        )
        lines = out.splitlines()
        assert "── Light ──" in lines
        assert "输出: demo" in lines
        assert "▶ 下载" in lines
        assert "✓ 下载 — 下载完成" in lines
        assert "▶ 翻译 — 翻译中..." in lines
        assert "✓ 翻译 — 翻译完成 (561 条)" in lines
        assert "── 完成 ──" in lines
        assert "日志: output/demo/pipeline_x.log" in lines
        assert any(line.startswith("耗时: ") for line in lines)
        assert "\x1b[" not in out

    def test_segment_prefix_and_dedup(self):
        s2 = SegmentRef(index=1, total=5, tag="seg2")
        out = _drive_plain(
            [
                _ev("asr", StageStatus.started, 0.0, "提取音频中...", segment=s2),
                _ev("asr", StageStatus.started, 0.0, "提取音频中...", segment=s2),  # idempotent re-emit
                _ev("asr", StageStatus.finished, 1.0, "ASR 完成", segment=s2),
                _ev("translate", StageStatus.skipped, 1.0, "无需翻译", segment=s2),
                _ev("asr", StageStatus.failed, 0.5, "RuntimeError: boom", segment=s2),
            ]
        )
        lines = out.splitlines()
        assert lines.count("[seg2/5] ▶ 语音转录 — 提取音频中...") == 1
        assert "[seg2/5] ✓ 语音转录 — ASR 完成" in lines
        assert "[seg2/5] – 翻译 — 无需翻译" in lines
        assert "[seg2/5] ✗ 语音转录 — RuntimeError: boom" in lines

    def test_progress_events_throttled(self):
        events = [_ev("asr", StageStatus.started, 0.0, "提取音频中...")]
        for pct in (0.1, 0.2, 0.3, 0.4):
            events.append(_ev("asr", StageStatus.progress, pct, "working"))
        out = _drive_plain(events)
        # All four progress events land inside the 1s throttle window → 1 bar line.
        bar_lines = [line for line in out.splitlines() if line.startswith("[=")]
        assert len(bar_lines) == 1
        assert bar_lines[0].endswith("10% 语音转录")
        assert "\x1b[" not in out


class TestRichReporter:
    def test_model_and_renderable(self):
        from light_cli.reporting.rich_ui import RichReporter
        from rich.console import Console

        console = Console(file=io.StringIO(), force_terminal=False, width=100)
        reporter = RichReporter(console=console, live=False)
        s1 = SegmentRef(index=0, total=2, tag="seg1")
        s2 = SegmentRef(index=1, total=2, tag="seg2")
        for event in (
            RunEvent(RunKind.started, {"slug": "demo", "mode": "translate→zh", "input": "video.mp4"}, ts=0.0),
            _ev(STAGE_DOWNLOAD, StageStatus.finished, 1.0, "下载完成"),
            _ev("asr", StageStatus.finished, 1.0, "ASR 完成", segment=s1),
            _ev("translate", StageStatus.started, 0.0, "翻译中...", segment=s2),
        ):
            reporter.emit(event)

        snap = reporter.model().snapshot()
        assert len(snap.segments) == 2
        assert snap.segments[1].stages[0].status == StageStatus.started

        # Renderable must build without a terminal and produce readable text.
        console.print(reporter._renderable())
        text = console.file.getvalue()
        assert "模式: translate→zh" in text
        assert "输入: video.mp4" in text
        assert "耗时:" in text
        assert "下载" in text
        assert "seg1/2" in text or "seg2/2" in text

    def test_single_segment_table_and_footer(self):
        from light_cli.reporting.rich_ui import RichReporter
        from rich.console import Console

        console = Console(file=io.StringIO(), force_terminal=False, width=100)
        reporter = RichReporter(console=console, live=False)
        for event in (
            _ev("asr", StageStatus.started, 0.0, "提取音频中..."),
            _ev("asr", StageStatus.finished, 1.0, "ASR 完成"),
            _ev("translate", StageStatus.skipped, 1.0, "无需翻译"),
            RunEvent(RunKind.finished, {"slug": "demo", "artifacts": ["demo.zh.srt"], "usage": "1.2k tok"}, ts=1.0),
        ):
            reporter.emit(event)
        console.print(reporter._renderable())
        text = console.file.getvalue()
        assert "语音转录" in text
        assert "无需翻译" in text
        assert "产物: demo.zh.srt" in text
        assert "用量: 1.2k tok" in text

    def test_header_shows_live_elapsed(self):
        from light_cli.reporting.rich_ui import RichReporter
        from rich.console import Console

        console = Console(file=io.StringIO(), force_terminal=False, width=100)
        reporter = RichReporter(console=console, live=False)
        reporter.emit(RunEvent(RunKind.started, {"slug": "demo", "mode": "mono"}, ts=0.0))
        console.print(reporter._renderable())
        text = console.file.getvalue()
        assert "耗时:" in text
        assert "模式: mono" in text

    def test_stage_meta_width_stable_with_or_without_progress(self):
        from light_cli.reporting.model import StageView
        from light_cli.reporting.rich_ui import _ELAPSED_WIDTH, _PROGRESS_SLOT_WIDTH, RichReporter

        reporter = RichReporter(live=False)
        now = time.time()
        started = StageView(
            stage="asr",
            status=StageStatus.started,
            progress=0.0,
            message="提取音频中...",
            ts=now,
            started_ts=now - 12,
        )
        progressing = StageView(
            stage="translate",
            status=StageStatus.progress,
            progress=0.4,
            message="批次 3/10...",
            ts=now,
            started_ts=now - 65,
        )
        finished = StageView(
            stage="download",
            status=StageStatus.finished,
            progress=1.0,
            message="下载完成",
            ts=now,
            started_ts=now - 8,
        )
        expected = _PROGRESS_SLOT_WIDTH + 1 + _ELAPSED_WIDTH
        assert len(reporter._stage_meta(started)) == expected
        assert len(reporter._stage_meta(progressing)) == expected
        assert len(reporter._stage_meta(finished)) == expected
        assert "40%" in reporter._stage_meta(progressing)
        assert "=" in reporter._stage_meta(progressing)

    def test_in_flight_elapsed_advances_on_rerender(self):
        from unittest.mock import patch

        from light_cli.reporting.rich_ui import RichReporter
        from rich.console import Console

        console = Console(file=io.StringIO(), force_terminal=False, width=120)
        reporter = RichReporter(console=console, live=False)
        started_wall = 1_000_000.0
        reporter.emit(
            ProgressEvent(
                stage="asr",
                status=StageStatus.started,
                progress=0.0,
                message="提取音频中...",
                segment=None,
                ts=started_wall,
            )
        )
        with patch("light_cli.reporting.rich_ui.time.time", return_value=started_wall + 5):
            meta_early = reporter._stage_meta(reporter.model().snapshot().segments[0].stages[0])
        with patch("light_cli.reporting.rich_ui.time.time", return_value=started_wall + 42):
            meta_late = reporter._stage_meta(reporter.model().snapshot().segments[0].stages[0])
        assert "5s" in meta_early
        assert "42s" in meta_late
        assert meta_early != meta_late


class TestStageLabels:
    def test_all_stage_constants_have_labels(self):
        from light_cli.reporting import events as run_events

        step_stages = {getattr(steps_progress, name) for name in dir(steps_progress) if name.startswith("STAGE_")}
        run_stages = set(run_events.RUN_STAGES)
        missing = (step_stages | run_stages) - set(STAGE_LABELS)
        assert not missing, f"stages missing Chinese labels: {missing}"


class TestElapsed:
    def test_started_not_finished_returns_non_negative(self):
        from light_cli.reporting.model import StageView
        from light_cli.reporting.rich_ui import _elapsed

        now = time.time()
        view = StageView(
            stage="download",
            status=StageStatus.started,
            progress=0.0,
            message="下载中…",
            ts=now - 5,
            started_ts=now - 5,
        )
        result = _elapsed(view, now)
        assert result == "5s"

    def test_finished_returns_ts_minus_started(self):
        from light_cli.reporting.model import StageView
        from light_cli.reporting.rich_ui import _elapsed

        ts = time.time()
        view = StageView(
            stage="download",
            status=StageStatus.finished,
            progress=1.0,
            message="下载完成",
            ts=ts,
            started_ts=ts - 3,
        )
        result = _elapsed(view, ts)
        assert result == "3s"

    def test_no_started_ts_returns_empty(self):
        from light_cli.reporting.model import StageView
        from light_cli.reporting.rich_ui import _elapsed

        view = StageView(
            stage="download",
            status=StageStatus.started,
            progress=0.0,
            message="下载中…",
            ts=time.time(),
            started_ts=None,
        )
        assert _elapsed(view, time.time()) == ""

    def test_clock_mismatch_clamped_to_zero(self):
        from light_cli.reporting.model import StageView
        from light_cli.reporting.rich_ui import _elapsed

        ts = time.time()
        view = StageView(
            stage="download",
            status=StageStatus.started,
            progress=0.0,
            message="下载中…",
            ts=ts,
            started_ts=ts + 100,
        )
        assert _elapsed(view, ts) == "0s"


class TestRtf:
    def test_footer_with_duration_shows_rtf(self):
        stream = io.StringIO()
        reporter = PlainReporter(stream=stream)
        reporter.emit(RunEvent(RunKind.started, {"slug": "demo"}, ts=0.0))
        reporter.emit(
            RunEvent(
                RunKind.finished,
                {"slug": "demo", "duration": 300.0, "log": "output/demo/pipeline.log"},
                ts=0.1,
            )
        )
        text = stream.getvalue()
        assert "RTF:" in text
        assert "时长" in text

    def test_footer_without_duration_no_rtf(self):
        stream = io.StringIO()
        reporter = PlainReporter(stream=stream)
        reporter.emit(RunEvent(RunKind.started, {"slug": "demo"}, ts=0.0))
        reporter.emit(
            RunEvent(
                RunKind.finished,
                {"slug": "demo", "log": "output/demo/pipeline.log"},
                ts=0.1,
            )
        )
        text = stream.getvalue()
        assert "RTF:" not in text
        assert "耗时:" in text

    def test_rtf_rounded_to_two_decimals(self):
        stream = io.StringIO()
        reporter = PlainReporter(stream=stream)
        reporter.emit(RunEvent(RunKind.started, {"slug": "demo"}, ts=0.0))
        reporter.emit(
            RunEvent(
                RunKind.finished,
                {"slug": "demo", "duration": 100.0},
                ts=0.1,
            )
        )
        text = stream.getvalue()
        for line in text.splitlines():
            if line.startswith("RTF:"):
                rtf_section = line.split("（")[0]
                rtf_str = rtf_section.split()[1]
                rtf_val = float(rtf_str)
                assert rtf_val == round(rtf_val, 2)
