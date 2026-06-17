"""Pipeline logging — console + file output for every run.

Writes timestamped log files to the output directory alongside
the standard ``typer.echo`` console output.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import ParamSpec, TypeVar

import typer

P = ParamSpec("P")
R = TypeVar("R")

_file_logger: contextvars.ContextVar[logging.Logger | None] = contextvars.ContextVar(
    "light_subtitle_file_logger",
    default=None,
)


def init(output_dir: str | Path) -> None:
    """Initialize logging for a pipeline run.

    Creates a timestamped log file at ``{output_dir}/pipeline_{ts}.log``.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    log_path = Path(output_dir) / f"pipeline_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"light-subtitle-{Path(output_dir).name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    _file_logger.set(logger)


def bind_file_logger(logger: logging.Logger | None) -> contextvars.Token[logging.Logger | None]:
    """Bind a file logger in the current thread (for worker threads)."""
    return _file_logger.set(logger)


def current_file_logger() -> logging.Logger | None:
    """Return the file logger bound in the current thread, if any."""
    return _file_logger.get()


def run_with_file_logger[P, R](fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Callable[[], R]:
    """Wrap *fn* so a worker thread inherits the submitter's file logger."""

    parent_logger = _file_logger.get()

    def task() -> R:
        token: contextvars.Token[logging.Logger | None] | None = None
        if parent_logger is not None:
            token = _file_logger.set(parent_logger)
        try:
            return fn(*args, **kwargs)
        finally:
            if token is not None:
                _file_logger.reset(token)

    return task


def info(msg: str) -> None:
    """Echo to console AND append to the run log file."""
    typer.echo(msg)
    logger = _file_logger.get()
    if logger is not None:
        logger.info(msg)


def warning(msg: str) -> None:
    """Echo to console AND append to the run log file at WARNING level."""
    typer.echo(msg)
    logger = _file_logger.get()
    if logger is not None:
        logger.warning(msg)
