"""Detect optimal split points at silence boundaries for long videos.

Ports the logic from ``x-subtitle/scripts/silence_split.py`` into the
``light-subtitle`` package so that long-video splitting happens in-process.

Usage::

    from light_subtitle.video_split import should_split, compute_split_points, split_video

    if should_split(video_path, threshold=2700):
        points = compute_split_points(video_path, target_duration=2700)
        segments = split_video(video_path, points, overlap=10)
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .utils.ffmpeg import probe_duration

# ── Scoring weights (silence quality vs proximity) ──────

PROXIMITY_WEIGHT = 0.4
DURATION_WEIGHT = 0.6
MAX_GAP_SEC = 5.0


@dataclass(frozen=True)
class SilenceInterval:
    start: float
    end: float
    duration: float


# ── Public API ──────────────────────────────────────────


def should_split(video_path: Path, threshold: float = 2700) -> bool:
    """Return True when *video_path* is longer than *threshold* seconds."""
    return probe_duration(str(video_path)) > threshold


def compute_split_points(
    video_path: Path,
    target_duration: float = 2700,
    window: float = 60,
    noise_db: float = -35,
) -> list[float]:
    """Find silence-aligned split points for *video_path*.

    Returns ``[0.0, P1, P2, …, PN, duration]`` where each interior point is
    placed at a nearby silence boundary.

    Works in two passes:
      1. Prefer long silences (≥1.5 s) within a wide window (± *window*).
      2. Fall back to any silence within a narrow window (±30 s).
    """
    duration = probe_duration(str(video_path))
    if duration <= target_duration:
        return [0.0, duration]

    intervals = _detect_silence_intervals(video_path, noise_db)
    N = int((duration + target_duration - 1) // target_duration)

    points: list[float] = [0.0]
    for k in range(1, N):
        target = k * (duration / N)
        best = _best_silence(target, intervals, window)
        points.append(best if best is not None else target)
    points.append(duration)
    return points


def split_video(
    video_path: Path,
    split_points: list[float],
    overlap: float = 10,
    *,
    seg_dir_template: str = ".seg",
) -> list[Path]:
    """Extract segments from *video_path* using *split_points*.

    Creates ``<parent>/.seg1/video.<ext>``, ``.seg2/``, … with copy-codec
    ffmpeg (fast, keyframe-aligned).  Returns the list of segment directories.

    Also saves ``split_points.json`` in the parent directory so the merge
    step can compute exact boundary offsets.
    """
    import json as _json

    parent = video_path.parent
    ext = video_path.suffix
    N = len(split_points) - 1  # number of segments

    seg_dirs: list[Path] = []
    for k in range(N):
        seg_dir = parent / f"{seg_dir_template}{k + 1}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg_dirs.append(seg_dir)

        seg_path = seg_dir / f"video{ext}"

        if k == 0:
            # First segment: from 0 to P1 + overlap
            to_val = split_points[1] + overlap
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    "0",
                    "-i",
                    str(video_path),
                    "-to",
                    f"{to_val:.3f}",
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-fflags",
                    "+genpts",
                    str(seg_path),
                ],
                check=True,
                capture_output=True,
            )
        elif k < N - 1:
            # Middle: Pk - overlap → Pk+1 + overlap
            ss_start = split_points[k] - overlap
            t_dur = split_points[k + 1] - split_points[k] + 2 * overlap
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{ss_start:.3f}",
                    "-i",
                    str(video_path),
                    "-t",
                    f"{t_dur:.3f}",
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-fflags",
                    "+genpts",
                    str(seg_path),
                ],
                check=True,
                capture_output=True,
            )
        else:
            # Last: PN-1 - overlap → end
            ss_start = split_points[k] - overlap
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{ss_start:.3f}",
                    "-i",
                    str(video_path),
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-fflags",
                    "+genpts",
                    str(seg_path),
                ],
                check=True,
                capture_output=True,
            )

    # Persist split points for the merge step.
    (parent / "split_points.json").write_text(
        _json.dumps({"split_points": split_points, "overlap": overlap}, indent=2),
        encoding="utf-8",
    )

    return seg_dirs


# ── Silence detection ───────────────────────────────────


def _detect_silence_intervals(
    video_path: Path,
    noise_db: float = -35,
) -> list[SilenceInterval]:
    """Extract audio, run ffmpeg silencedetect, return silence intervals."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ],
            capture_output=True,
            check=True,
            timeout=600,
        )

        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio_path,
                "-af",
                f"silencedetect=noise={noise_db}dB:d=0.5",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        intervals: list[SilenceInterval] = []
        pending_start: float | None = None

        for line in result.stderr.splitlines():
            m_start = re.search(r"silence_start:\s*([\d.]+)", line)
            if m_start:
                pending_start = float(m_start.group(1))
                continue
            m_end = re.search(
                r"silence_end:\s*([\d.]+)\s*\|?\s*silence_duration:\s*([\d.]+)",
                line,
            )
            if m_end and pending_start is not None:
                end = float(m_end.group(1))
                dur = float(m_end.group(2))
                intervals.append(SilenceInterval(pending_start, end, dur))
                pending_start = None

        return sorted(intervals, key=lambda x: x.start)
    finally:
        Path(audio_path).unlink(missing_ok=True)


def _best_silence(
    target: float,
    intervals: list[SilenceInterval],
    window: float,
) -> float | None:
    """Find the best silence_end near *target* using duration-weighted scoring."""

    def _pick(candidates: list[SilenceInterval], win: float) -> float | None:
        if not candidates:
            return None

        def score(item: SilenceInterval) -> float:
            proximity = 1.0 - min(abs(item.start - target) / win, 1.0)
            dur_norm = min(item.duration, MAX_GAP_SEC) / MAX_GAP_SEC
            return PROXIMITY_WEIGHT * proximity + DURATION_WEIGHT * dur_norm

        return max(candidates, key=score).end

    # Pass 1: long silences in wide window
    wide = [s for s in intervals if abs(s.start - target) <= window and s.duration >= 1.5]
    result = _pick(wide, window)
    if result is not None:
        return result

    # Pass 2: any silence in narrow window
    narrow = min(window, 30.0)
    narrow_candidates = [s for s in intervals if abs(s.start - target) <= narrow]
    return _pick(narrow_candidates, narrow)
