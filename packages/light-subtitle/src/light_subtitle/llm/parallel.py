"""Helpers for parallel LLM batch execution with prompt-cache warmup."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_parallel_with_warmup[T](
    tasks: list[tuple[Callable[[], T], int]],
    *,
    max_workers: int,
) -> dict[int, T]:
    """Run *tasks* with the lowest-index task serially first to warm prompt cache.

    Each task is ``(callable, result_index)``. Returns ``{index: result}``.
    """
    if not tasks:
        return {}

    by_index = {idx: fn for fn, idx in tasks}
    results: dict[int, T] = {}

    warmup_idx = min(by_index)
    results[warmup_idx] = by_index[warmup_idx]()

    remaining = [(fn, idx) for fn, idx in tasks if idx != warmup_idx]
    if not remaining:
        return results

    workers = min(max_workers, len(remaining))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn): idx for fn, idx in remaining}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return results
