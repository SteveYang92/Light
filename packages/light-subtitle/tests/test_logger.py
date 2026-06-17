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
