"""Rich terminal progress reporter — live-updating TTY view.

``rich`` is imported lazily (never at module top level), so importing this
module is safe in headless environments; instantiating :class:`RichReporter`
still requires rich (a project dependency).
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING

from .events import ProgressEvent, RunEvent, RunKind, StageStatus
from .labels import ICON_WAITING, STATUS_ICONS, stage_label
from .model import RunModel, RunSnapshot, SegmentView, StageView

if TYPE_CHECKING:
    from rich.console import Console, RenderableType

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_PROGRESS_BAR_WIDTH = 18


class RichReporter:
    """Reporter rendering the run as a live Rich layout (TTY).

    *console* may be injected (tests, capture); by default a Console bound
    to the real ``sys.stdout`` is created so later stream redirections by
    the logger do not affect the live view.  Set ``live=False`` to drive
    the model without a Live session (tests).
    """

    def __init__(self, console: Console | None = None, *, live: bool = True, refresh_per_second: float = 4.0) -> None:
        self._model = RunModel()
        self._lock = threading.Lock()
        self._console = console if console is not None else _default_console()
        self._live_enabled = live
        self._refresh_per_second = refresh_per_second
        self._live = None
        self._frame = 0
        self._started_ts: float | None = None

    # ── Reporter protocol ─────────────────────────────────────────────

    def emit(self, event: ProgressEvent | RunEvent) -> None:
        with self._lock:
            if self._started_ts is None:
                self._started_ts = time.monotonic()
            self._model.apply(event)
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            if not self._live_enabled:
                return
            if self._live is None:
                self._start_live()
            # Live.update requires a RenderableType — a callable is not renderable
            # and raises NotRenderableError on refresh (blank/frozen TTY).
            # refresh=True so the finished footer (incl. RTF) paints immediately
            # rather than waiting for the next auto-refresh tick before close().
            self._live.update(self._renderable(), refresh=True)

    def close(self) -> None:
        """Stop the Live session (final frame stays on screen)."""
        with self._lock:
            if self._live is not None:
                self._live.stop()
                self._live = None

    def model(self) -> RunModel:
        """Exposed for tests/renderers that want the current snapshot."""
        return self._model

    def _start_live(self) -> None:
        from rich.live import Live

        self._live = Live(
            self._renderable(),
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=False,
        )
        self._live.start()

    # ── rendering (rich imports stay inside) ──────────────────────────

    def _renderable(self) -> RenderableType:
        from rich.console import Group
        from rich.panel import Panel

        snap = self._model.snapshot()
        parts = [self._header(snap)]
        if len(snap.segments) <= 1:
            parts.append(self._stage_table(snap))
        else:
            parts.append(self._segments_table(snap))
        if snap.run_kind == RunKind.finished:
            parts.append(self._footer(snap))
        elif snap.run_kind == RunKind.failed:
            parts.append(self._failed_footer(snap))
        return Panel(Group(*parts), border_style="dim", padding=(0, 1))

    def _header(self, snap: RunSnapshot) -> RenderableType:
        from rich.text import Text

        payload = snap.run_payload
        lines: list[str] = []
        for key, label in (("mode", "模式"), ("input", "输入"), ("output", "输出")):
            value = payload.get(key)
            if value:
                lines.append(f"{label}: {value}")
        if not any(k in payload for k in ("mode", "input", "output")) and payload.get("slug"):
            lines.append(f"输出: {payload['slug']}")
        return Text("\n".join(lines) if lines else "Light", style="bold")

    def _stage_table(self, snap: RunSnapshot) -> RenderableType:
        """Single-segment (or implicit 1/1) stage checklist."""
        from rich.table import Table

        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_column(width=2)
        table.add_column(style="bold", no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        table.add_column(no_wrap=True, justify="right")
        for view in (*snap.run_stages, *(snap.segments[0].stages if snap.segments else ())):
            table.add_row(
                self._status_icon(view),
                stage_label(view.stage),
                view.message,
                self._stage_meta(view),
            )
        return table

    def _segments_table(self, snap: RunSnapshot) -> RenderableType:
        """Multi-segment view: run-level rows + one compact row per segment."""
        from rich.table import Table

        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_column(width=2)
        table.add_column(no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        table.add_column(no_wrap=True, justify="right")
        for view in snap.run_stages:
            table.add_row(self._status_icon(view), stage_label(view.stage), view.message, "")
        for segment in snap.segments:
            table.add_row(*self._segment_row(segment))
        return table

    def _segment_row(self, segment: SegmentView) -> tuple[str, str, str, str]:
        ref = segment.ref
        tag = f"{ref.tag or ref.index + 1}/{ref.total}"
        done = sum(1 for s in segment.stages if s.status == StageStatus.finished)
        current = next((s for s in reversed(segment.stages) if s.status in _ACTIVE_STATUSES), None)
        if current is not None:
            icon = self._status_icon(current)
            label = stage_label(current.stage)
            meta = self._stage_meta(current)
        else:
            icon = ICON_WAITING
            label = "等待"
            meta = ""
        detail = f"({done})" if done else ""
        return icon, f"{tag} {label}", detail, meta

    def _status_icon(self, view: StageView) -> str:
        if view.status in (StageStatus.started, StageStatus.progress):
            return _SPINNER_FRAMES[self._frame]
        return STATUS_ICONS[view.status]

    def _stage_meta(self, view: StageView) -> str:
        """Progress bar (for fractional progress) + elapsed time."""
        parts: list[str] = []
        if view.status == StageStatus.progress and 0.0 < view.progress < 1.0:
            filled = round(view.progress * _PROGRESS_BAR_WIDTH)
            bar = "=" * filled + "-" * (_PROGRESS_BAR_WIDTH - filled)
            parts.append(f"[{bar}] {round(view.progress * 100)}%")
        elapsed = _elapsed(view, time.time())
        if elapsed:
            parts.append(elapsed)
        return " ".join(parts)

    def _footer(self, snap: RunSnapshot) -> RenderableType:
        from rich.text import Text

        payload = snap.run_payload
        lines = ["── 完成 ──"]
        output = payload.get("output")
        if output:
            lines.append(f"输出: {output}")
        artifacts = payload.get("artifacts")
        if artifacts:
            lines.append("产物: " + ", ".join(str(a) for a in artifacts))
        usage = payload.get("usage")
        if usage:
            lines.append(f"用量: {usage}")
        log = payload.get("log")
        if log:
            lines.append(f"日志: {log}")
        if self._started_ts is not None:
            elapsed = time.monotonic() - self._started_ts
            lines.append(f"耗时: {_format_elapsed(elapsed)}")
            duration = payload.get("duration")
            if duration and duration > 0:
                rtf = elapsed / duration
                lines.append(f"RTF: {rtf:.2f}（耗时 {_format_elapsed(elapsed)} / 时长 {_format_elapsed(duration)}）")
        return Text("\n".join(lines))

    def _failed_footer(self, snap: RunSnapshot) -> RenderableType:
        from rich.text import Text

        payload = snap.run_payload
        lines = [f"✗ 失败: {payload.get('error', '')}"]
        log = payload.get("log")
        if log:
            lines.append(f"日志: {log}")
        lines.append("提示: 可用 --resume 或 --resume-from 续跑")
        return Text("\n".join(lines), style="bold red")


_ACTIVE_STATUSES = frozenset({StageStatus.started, StageStatus.progress, StageStatus.failed})


def _elapsed(view: StageView, now: float) -> str:
    """Stage elapsed time: finished → ts - started_ts; in-flight → now - started_ts."""
    if view.started_ts is None:
        return ""
    if view.status in (StageStatus.finished, StageStatus.failed, StageStatus.skipped):
        end = view.ts
    else:
        end = now
    return _format_elapsed(max(0.0, end - view.started_ts))


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _default_console() -> Console:
    from rich.console import Console

    return Console(file=sys.stdout)
