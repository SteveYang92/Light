"""Tests for LLM parallel warmup helper."""

from __future__ import annotations

from light_subtitle.llm.parallel import run_parallel_with_warmup


def test_run_parallel_with_warmup_runs_lowest_index_first() -> None:
    order: list[int] = []

    def make_task(idx: int):
        def task() -> int:
            order.append(idx)
            return idx

        return task

    tasks = [(make_task(2), 2), (make_task(0), 0), (make_task(1), 1)]
    results = run_parallel_with_warmup(tasks, max_workers=2)

    assert results == {0: 0, 1: 1, 2: 2}
    assert order[0] == 0


def test_run_parallel_with_warmup_empty() -> None:
    assert run_parallel_with_warmup([], max_workers=4) == {}
