"""Pipeline logging — console + file output for every run.

Writes timestamped log files to the output directory alongside
the standard ``typer.echo`` console output.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import typer

_log: logging.Logger | None = None


def init(output_dir: str | Path) -> None:
    """Initialize logging for a pipeline run.

    Creates a timestamped log file at ``{output_dir}/pipeline_{ts}.log``.
    Call once at pipeline startup, before any log calls.
    """
    global _log

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    log_path = Path(output_dir) / f"pipeline_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _log = logging.getLogger("light-subtitle")
    _log.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    _log.handlers.clear()
    _log.addHandler(handler)


def info(msg: str) -> None:
    """Echo to console AND append to the run log file."""
    typer.echo(msg)
    if _log is not None:
        _log.info(msg)


def warning(msg: str) -> None:
    """Echo to console AND append to the run log file at WARNING level."""
    typer.echo(msg)
    if _log is not None:
        _log.warning(msg)
