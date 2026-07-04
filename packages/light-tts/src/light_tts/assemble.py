from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlacedSegment:
    start: float
    samples: np.ndarray
    sample_rate: int


def assemble_timeline(
    segments: list[PlacedSegment],
    total_duration: float,
    sample_rate: int,
    *,
    crossfade_ms: float = 50.0,
    replace_on_overlap: bool = False,
) -> np.ndarray:
    """Place segments on a timeline with optional linear crossfade at segment starts."""
    total_samples = max(1, int(total_duration * sample_rate))
    timeline = np.zeros(total_samples, dtype=np.float32)
    cf = max(0, int(sample_rate * crossfade_ms / 1000.0))

    for seg in sorted(segments, key=lambda s: s.start):
        if seg.sample_rate != sample_rate:
            raise ValueError(f"Sample rate mismatch: {seg.sample_rate} != {sample_rate}")
        start_idx = max(0, int(seg.start * sample_rate))
        chunk = seg.samples.astype(np.float32)
        end_idx = min(total_samples, start_idx + len(chunk))
        chunk = chunk[: end_idx - start_idx]
        if len(chunk) == 0:
            continue

        for i, sample in enumerate(chunk):
            idx = start_idx + i
            if idx >= total_samples:
                break
            prev = timeline[idx]
            if prev == 0.0 or replace_on_overlap:
                timeline[idx] = sample
                continue
            if cf > 0 and i < cf:
                alpha = (i + 1) / cf
                timeline[idx] = prev * (1.0 - alpha) + sample * alpha
            else:
                timeline[idx] = np.clip(prev + sample, -1.0, 1.0)

    return timeline
