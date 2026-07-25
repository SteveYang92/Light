"""Reporter protocol and adapters (fan-out, legacy 3-tuple callable)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .events import ProgressEvent, RunEvent

Event = ProgressEvent | RunEvent

# Anything :func:`as_reporter` accepts.
ProgressInput = "Reporter | Callable[[str, float, str], None] | None"


@runtime_checkable
class Reporter(Protocol):
    """Consumer of progress events (renderer or downstream sink)."""

    def emit(self, event: Event) -> None: ...


class CompositeReporter:
    """Fan out each event to every child reporter."""

    def __init__(self, *reporters: Reporter) -> None:
        self._reporters = reporters

    def emit(self, event: Event) -> None:
        for reporter in self._reporters:
            reporter.emit(event)


class CallableReporter:
    """Adapt the legacy ``fn(stage, progress, message)`` contract.

    Every ProgressEvent is forwarded as the plain 3-tuple — segment and
    status are dropped; the event's own progress value passes through for
    all statuses (started/progress/finished, and failed/skipped as-is).
    RunEvents have no legacy representation and are dropped.
    """

    def __init__(self, fn: Callable[[str, float, str], None]) -> None:
        self._fn = fn

    def emit(self, event: Event) -> None:
        if isinstance(event, ProgressEvent):
            self._fn(event.stage, event.progress, event.message)


class _NullReporter:
    def emit(self, event: Event) -> None:
        pass


_NULL_REPORTER = _NullReporter()


def as_reporter(obj: Reporter | Callable[[str, float, str], None] | None) -> Reporter:
    """Normalize *obj* to a :class:`Reporter`.

    Reporters pass through unchanged; callables are wrapped in
    :class:`CallableReporter`; None yields a no-op reporter.
    """
    if obj is None:
        return _NULL_REPORTER
    if isinstance(obj, Reporter):
        return obj
    if callable(obj):
        return CallableReporter(obj)
    raise TypeError(f"Cannot adapt {type(obj).__name__} to Reporter")
