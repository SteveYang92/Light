"""Plain-text progress reporter — for non-TTY, CI, and --verbose output.

Prints one line per state transition (never ANSI escapes, safe to capture
from subprocesses).  Fine-grained ``status=progress`` events are throttled
to one progress-bar line per second per stage.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

from .events import ProgressEvent, RunEvent, RunKind, SegmentRef, StageStatus
from .labels import STATUS_ICONS, stage_label
from .model import RunModel

_PROGRESS_BAR_WIDTH = 10
_PROGRESS_THROTTLE_SEC = 1.0


class PlainReporter:
    """Reporter that prints plain-text lines on state transitions."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._model = RunModel()
        self._lock = threading.Lock()
        self._started_ts: float | None = None  # monotonic ts of first event (summary elapsed)
        # (segment, stage) → last printed (status, message); dedupes idempotent re-emits
        self._last_line: dict[tuple[SegmentRef | None, str], tuple[StageStatus, str]] = {}
        # (segment, stage) → monotonic ts of last printed progress-bar line
        self._last_progress_line: dict[tuple[SegmentRef | None, str], float] = {}

    # ── Reporter protocol ─────────────────────────────────────────────

    def emit(self, event: ProgressEvent | RunEvent) -> None:
        with self._lock:
            if self._started_ts is None:
                self._started_ts = time.monotonic()
            self._model.apply(event)
            if isinstance(event, RunEvent):
                self._print_run_event(event)
            else:
                self._print_progress_event(event)

    def model(self) -> RunModel:
        """Exposed for tests/renderers that want the current snapshot."""
        return self._model

    def close(self) -> None:
        """Protocol parity with RichReporter (plain printing needs no teardown)."""

    # ── internals ─────────────────────────────────────────────────────

    def _print(self, text: str) -> None:
        print(text, file=self._stream, flush=True)

    @staticmethod
    def _segment_prefix(segment: SegmentRef | None) -> str:
        """``[seg2/5] `` for real multi-segment refs; empty otherwise."""
        if segment is None or (segment.total <= 1 and not segment.tag):
            return ""
        return f"[{segment.tag or segment.index + 1}/{segment.total}] "

    def _print_progress_event(self, event: ProgressEvent) -> None:
        key = (event.segment, event.stage)
        label = stage_label(event.stage)
        prefix = self._segment_prefix(event.segment)

        if event.status == StageStatus.progress:
            now = time.monotonic()
            if now - self._last_progress_line.get(key, 0.0) < _PROGRESS_THROTTLE_SEC:
                return
            self._last_progress_line[key] = now
            filled = round(event.progress * _PROGRESS_BAR_WIDTH)
            bar = "=" * filled + "-" * (_PROGRESS_BAR_WIDTH - filled)
            self._print(f"{prefix}[{bar}] {round(event.progress * 100)}% {label}")
            return

        marker = (event.status, event.message)
        if self._last_line.get(key) == marker:
            return
        self._last_line[key] = marker

        icon = STATUS_ICONS[event.status]
        if event.message:
            self._print(f"{prefix}{icon} {label} — {event.message}")
        else:
            self._print(f"{prefix}{icon} {label}")

    def _print_run_event(self, event: RunEvent) -> None:
        payload = event.payload
        if event.kind == RunKind.started:
            self._print("── Light ──")
            seen_output = False
            for key, label in (("mode", "模式"), ("input", "输入"), ("output", "输出")):
                value = payload.get(key)
                if value:
                    self._print(f"{label}: {value}")
                    seen_output = seen_output or key == "output"
            if not seen_output and payload.get("slug"):
                self._print(f"输出: {payload['slug']}")
        elif event.kind == RunKind.finished:
            self._print("── 完成 ──")
            output = payload.get("output")
            if output:
                self._print(f"输出: {output}")
            artifacts = payload.get("artifacts")
            if artifacts:
                self._print("产物: " + ", ".join(str(a) for a in artifacts))
            usage = payload.get("usage")
            if usage:
                self._print(f"用量: {usage}")
            log = payload.get("log")
            if log:
                self._print(f"日志: {log}")
            elapsed = time.monotonic() - self._started_ts if self._started_ts is not None else 0.0
            self._print(f"耗时: {_format_elapsed(elapsed)}")
            duration = payload.get("duration")
            if duration and duration > 0:
                rtf = elapsed / duration
                self._print(f"RTF: {rtf:.2f}（耗时 {_format_elapsed(elapsed)} / 时长 {_format_elapsed(duration)}）")
        elif event.kind == RunKind.failed:
            self._print(f"✗ 运行失败: {payload.get('error', '')}")
            self._print("提示: 可用 --resume 或 --resume-from 续跑")


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"
