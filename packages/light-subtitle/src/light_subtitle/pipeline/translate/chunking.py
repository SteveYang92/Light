"""Batch chunking for translation — group pending segments into LLM batches."""

from __future__ import annotations

from light_models import Segment

from .protocol import _parse_split_part


def _split_group_extent(pending: list[Segment], index: int) -> tuple[int, int]:
    """Return ``[start, end)`` indices of the split_group containing ``pending[index]``."""
    parsed = _parse_split_part(pending[index].unit_id)
    if parsed is None:
        return index, index + 1
    group_id = parsed[0]
    start = index
    while start > 0:
        prev = _parse_split_part(pending[start - 1].unit_id)
        if prev and prev[0] == group_id:
            start -= 1
        else:
            break
    end = index + 1
    while end < len(pending):
        nxt = _parse_split_part(pending[end].unit_id)
        if nxt and nxt[0] == group_id:
            end += 1
        else:
            break
    return start, end


def _chunk_pending_segments(pending: list[Segment], chunk_size: int) -> list[list[Segment]]:
    """Chunk pending segments; never split a ``split_group`` across batches.

    A split_group may occupy a batch larger than *chunk_size* when the whole
    group does not fit in the remaining space of the current batch.
    """
    if not pending:
        return []

    chunks: list[list[Segment]] = []
    i = 0
    while i < len(pending):
        chunk: list[Segment] = []
        while i < len(pending):
            g_start, g_end = _split_group_extent(pending, i)
            g_len = g_end - g_start
            if g_len > 1:
                group_slice = pending[g_start:g_end]
                if chunk and len(chunk) + g_len > chunk_size:
                    break
                if not chunk and g_len > chunk_size:
                    chunks.append(group_slice)
                    i = g_end
                    chunk = []
                    break
                chunk.extend(group_slice)
                i = g_end
                continue
            if len(chunk) >= chunk_size:
                break
            chunk.append(pending[i])
            i += 1
        if chunk:
            chunks.append(chunk)
    return chunks


def _adjust_chunk_end(pending: list[Segment], start: int, end: int, chunk_size: int) -> int:
    """Extend *end* so a split_group at the boundary stays in one batch (may exceed *chunk_size*)."""
    if end >= len(pending):
        return end

    g_start, g_end = _split_group_extent(pending, end - 1)
    if g_end <= end:
        g_start, g_end = _split_group_extent(pending, end)
    if g_end - g_start <= 1:
        return end
    if g_start < start:
        return end
    return g_end
