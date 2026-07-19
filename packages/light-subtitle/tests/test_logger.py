"""Tests for pipeline logger context propagation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from light_subtitle import logger


def test_worker_threads_inherit_file_logger(tmp_path):
    """ThreadPoolExecutor workers should write to the same pipeline log file."""
    logger.init(tmp_path)

    def worker() -> None:
        logger.info("  Layout merge hint: mu0001 → merge_with_next | 'test'")

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(logger.run_with_file_logger(worker)).result()

    log_files = list(tmp_path.glob("pipeline_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8")
    assert "Layout merge hint" in content


def test_console_echo_toggle(tmp_path, capsys):
    """set_console_echo(False) suppresses echo but keeps file logging."""
    logger.init(tmp_path)
    logger.set_console_echo(False)
    try:
        logger.info("  hidden from console")
        logger.warning("  hidden warning too")
        assert capsys.readouterr().out == ""
    finally:
        logger.set_console_echo(True)

    logger.info("  visible again")
    assert "visible again" in capsys.readouterr().out

    content = next(tmp_path.glob("pipeline_*.log")).read_text(encoding="utf-8")
    assert "hidden from console" in content
    assert "hidden warning too" in content


def test_capture_external_output_writes_to_log_file(tmp_path, capsys):
    """print() inside capture_external_output lands in the pipeline log."""
    logger.init(tmp_path)
    with logger.capture_external_output():
        print("noise-from-library")
    content = next(tmp_path.glob("pipeline_*.log")).read_text(encoding="utf-8")
    assert "noise-from-library" in content

    # stdout is restored after the context manager.
    print("back-to-console")
    assert "back-to-console" in capsys.readouterr().out


def test_capture_external_output_without_logger_goes_devnull(capsys):
    """Without a bound file logger, captured output is discarded."""
    token = logger.bind_file_logger(None)
    try:
        with logger.capture_external_output():
            print("discarded-noise")
        assert "discarded-noise" not in capsys.readouterr().out
    finally:
        logger._file_logger.reset(token)


def test_log_path(tmp_path):
    """log_path returns the bound pipeline log file path (None when unbound)."""
    token = logger.bind_file_logger(None)
    try:
        assert logger.log_path() is None
    finally:
        logger._file_logger.reset(token)

    logger.init(tmp_path)
    path = logger.log_path()
    assert path is not None
    assert path.parent == tmp_path
    assert path.name.startswith("pipeline_")
