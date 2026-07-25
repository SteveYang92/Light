"""Pipeline logging — console + file output for every run.

Writes timestamped log files to the output directory alongside
the standard ``typer.echo`` console output.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import ParamSpec, TypeVar

import typer

P = ParamSpec("P")
R = TypeVar("R")

_file_logger: contextvars.ContextVar[logging.Logger | None] = contextvars.ContextVar(
    "light_cli_file_logger",
    default=None,
)

# Console echo switch (module-level; single bool assignment is atomic
# under the GIL, so no lock is needed).  File logging is unaffected.
_console_echo = True


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


def log_path() -> Path | None:
    """Return the bound pipeline log file path, or None when uninitialized."""
    flog = _file_logger.get()
    if flog is None or not flog.handlers:
        return None
    handler = flog.handlers[0]
    return Path(handler.baseFilename) if isinstance(handler, logging.FileHandler) else None


def set_console_echo(enabled: bool) -> None:
    """Enable/disable console echo for :func:`info`/:func:`warning`.

    File logging is unaffected.  Default is True (backend/pack behavior).
    """
    global _console_echo
    _console_echo = bool(enabled)


@contextlib.contextmanager
def capture_external_output() -> Iterator[None]:
    """Redirect stdout/stderr into the current pipeline log file.

    Third-party libraries (whisperx, pyannote) print/tqdm/log straight to
    the terminal; wrapping their calls here sends that output to the run's
    log file instead.  Without a bound file logger the output is discarded
    (devnull) — it is noise we intentionally drop.  RichReporter holds a
    reference to the real stdout and is unaffected.
    """
    flog = _file_logger.get()
    owns_target = flog is None or not flog.handlers
    target = open(os.devnull, "w", encoding="utf-8") if owns_target else flog.handlers[0].stream
    with contextlib.redirect_stdout(target), contextlib.redirect_stderr(target):
        try:
            yield
        finally:
            target.flush()
            if owns_target:
                target.close()


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
    """Echo to console (when enabled) AND append to the run log file."""
    if _console_echo:
        typer.echo(msg)
    logger = _file_logger.get()
    if logger is not None:
        logger.info(msg)


def warning(msg: str) -> None:
    """Echo to console (when enabled) AND append to the run log file at WARNING level."""
    if _console_echo:
        typer.echo(msg)
    logger = _file_logger.get()
    if logger is not None:
        logger.warning(msg)
