"""Shared progress-reporting primitives."""

from __future__ import annotations

from collections.abc import Callable

ProgressCallback = Callable[[float, str | None], None]
"""Progress callback: invoked with a 0-1 fraction of the calling operation and an optional message."""
