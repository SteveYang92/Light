"""Split-part unit-id protocol — parse and classify ``pNNNN_K`` identifiers.

Word-level splits of an overlong planned unit are suffixed with a part
index (``p0007_0``, ``p0007_1``, …).  These helpers are the single source
for parsing that scheme and for sentence-end detection used by the
split-aware payload/normalization logic.
"""

from __future__ import annotations

import re

from light_models import Segment

_SPLIT_PART_RE = re.compile(r"^(p\d+)_(\d+)$")
_EN_SENTENCE_END = frozenset(".!?…")


def _parse_split_part(unit_id: str) -> tuple[str, int] | None:
    """Return ``(split_group_id, part_index)`` for units like ``p0007_0`` or ``p0007_1``."""
    match = _SPLIT_PART_RE.match(unit_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _split_group_part_counts(segments: list[Segment]) -> dict[str, int]:
    """Map split group id to number of parts (max index + 1)."""
    max_index: dict[str, int] = {}
    for segment in segments:
        parsed = _parse_split_part(segment.unit_id)
        if parsed is None:
            continue
        group_id, part_index = parsed
        max_index[group_id] = max(max_index.get(group_id, 0), part_index + 1)
    return max_index


def _is_last_split_part(unit_id: str, part_counts: dict[str, int]) -> bool | None:
    """Return whether *unit_id* is the last part of its split group, or None if not split."""
    parsed = _parse_split_part(unit_id)
    if parsed is None:
        return None
    group_id, part_index = parsed
    count = part_counts.get(group_id, part_index + 1)
    return part_index >= count - 1


def _source_ends_sentence(source_text: str) -> bool:
    """True when English source ends with sentence-final punctuation."""
    stripped = source_text.rstrip()
    return bool(stripped) and stripped[-1] in _EN_SENTENCE_END
