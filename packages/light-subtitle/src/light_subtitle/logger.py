"""Pipeline logging — console + file output for every run.

Writes timestamped log files to the output directory alongside
the standard ``typer.echo`` console output.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import typer

_log = threading.local()


def init(output_dir: str | Path) -> None:
    """Initialize logging for a pipeline run.

    Creates a timestamped log file at ``{output_dir}/pipeline_{ts}.log``.
    Thread-safe: each thread gets its own file handler keyed by output_dir.
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
    _log.logger = logger


def info(msg: str) -> None:
    """Echo to console AND append to the run log file."""
    typer.echo(msg)
    logger = getattr(_log, "logger", None)
    if logger is not None:
        logger.info(msg)


def warning(msg: str) -> None:
    """Echo to console AND append to the run log file at WARNING level."""
    typer.echo(msg)
    logger = getattr(_log, "logger", None)
    if logger is not None:
        logger.warning(msg)
