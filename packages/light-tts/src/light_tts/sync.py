from __future__ import annotations

import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import read_wav, write_wav


@dataclass(frozen=True)
class FitResult:
    samples: np.ndarray
    sample_rate: int
    atempo: float
    trimmed: bool
    overflow_s: float = 0.0


def _apply_atempo(samples: np.ndarray, sample_rate: int, atempo: float) -> np.ndarray:
    """Time-stretch via ffmpeg ``atempo`` (chain filters when factor > 2.0)."""
    if abs(atempo - 1.0) < 0.01:
        return samples
    filters: list[str] = []
    remaining = atempo
    while remaining > 1.01:
        step = min(remaining, 2.0)
        filters.append(f"atempo={step:.4f}")
        remaining /= step
    filter_str = ",".join(filters)
    with tempfile.TemporaryDirectory(prefix="light-tts-sync-") as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        write_wav(inp, samples, sample_rate)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(inp),
            "-filter:a",
            filter_str,
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        stretched, _ = read_wav(out)
        return stretched


def _fade_out(samples: np.ndarray, sample_rate: int, fade_ms: float = 30.0) -> np.ndarray:
    fade_samples = min(len(samples), int(sample_rate * fade_ms / 1000.0))
    if fade_samples <= 0:
        return samples
    out = samples.copy()
    ramp = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    out[-fade_samples:] *= ramp
    return out


def compute_turn_placed_start(
    scheduled_start: float,
    prev_end: float | None,
    *,
    speaker_gap_s: float,
    max_inter_speaker_pause_s: float,
) -> float:
    """Place the next speaker turn without preserving long dead air from the source."""
    if prev_end is None:
        return scheduled_start
    earliest = prev_end + speaker_gap_s
    capped = min(scheduled_start, prev_end + max_inter_speaker_pause_s)
    return max(earliest, capped)


def compute_subtitle_aligned_start(
    scheduled_start: float,
    prev_end: float | None,
    *,
    speaker_gap_s: float,
) -> float:
    """Place monologue speech at subtitle cue time; push later only when the prior turn overruns."""
    if prev_end is None:
        return scheduled_start
    return max(scheduled_start, prev_end + speaker_gap_s)


def fit_duration(
    samples: np.ndarray,
    sample_rate: int,
    target_duration: float,
    *,
    max_duration: float | None = None,
    atempo_min: float = 0.88,
    atempo_max: float = 1.28,
    allow_trim: bool = False,
    pad_to_target: bool = True,
    strict_cap: bool = True,
) -> FitResult:
    """Fit *samples* into subtitle timing.

    *target_duration* is the cue window. *max_duration* is the latest the clip may end
    (typically next cue start minus gap). Speech may extend past the subtitle window up
    to *max_duration* when that cap is larger (same-speaker runs).

    When *strict_cap* is False, time-stretch up to *atempo_max* but never trim; callers
    push later segments on the timeline instead.
    """
    del atempo_min  # reserved for future slow-down stretch
    if target_duration <= 0 or len(samples) == 0:
        return FitResult(samples=samples, sample_rate=sample_rate, atempo=1.0, trimmed=False)

    cap = max_duration if max_duration is not None else target_duration
    cap = max(cap, 0.05)
    # Strict trim mode: never extend past the subtitle window.
    if allow_trim:
        cap = min(cap, target_duration)

    actual = len(samples) / sample_rate
    tolerance = target_duration * 0.08

    if actual <= cap and abs(actual - target_duration) <= tolerance:
        return FitResult(samples=samples, sample_rate=sample_rate, atempo=1.0, trimmed=False)

    if actual > target_duration:
        ratio = actual / target_duration
        atempo = min(max(ratio, 1.0), atempo_max)
        stretched = _apply_atempo(samples, sample_rate, atempo)
        stretched_dur = len(stretched) / sample_rate

        if stretched_dur <= cap:
            overflow = max(0.0, stretched_dur - target_duration)
            return FitResult(
                samples=stretched,
                sample_rate=sample_rate,
                atempo=atempo,
                trimmed=False,
                overflow_s=overflow,
            )

        if not strict_cap:
            overflow = max(0.0, stretched_dur - target_duration)
            if ratio > atempo_max:
                warnings.warn(
                    f"Speech {actual:.2f}s > window {target_duration:.2f}s (needs {ratio:.2f}x); "
                    f"atempo {atempo:.2f}x — keeping full audio, pushing later turns",
                    stacklevel=2,
                )
            return FitResult(
                samples=stretched,
                sample_rate=sample_rate,
                atempo=atempo,
                trimmed=False,
                overflow_s=overflow,
            )

        if ratio > atempo_max:
            warnings.warn(
                f"Speech {actual:.2f}s > cap {cap:.2f}s (window {target_duration:.2f}s, needs {ratio:.2f}x); "
                f"atempo {atempo:.2f}x — trimming tail to avoid speaker overlap",
                stacklevel=2,
            )
        max_samples = int(cap * sample_rate)
        trimmed = _fade_out(stretched[:max_samples], sample_rate)
        return FitResult(
            samples=trimmed,
            sample_rate=sample_rate,
            atempo=atempo,
            trimmed=True,
            overflow_s=0.0,
        )

    if not pad_to_target:
        return FitResult(samples=samples, sample_rate=sample_rate, atempo=1.0, trimmed=False)

    # Shorter than window — pad with silence (no slow-down stretch).
    target_samples = int(target_duration * sample_rate)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: len(samples)] = samples
    return FitResult(samples=padded, sample_rate=sample_rate, atempo=1.0, trimmed=False)


def fit_budget(
    cue_start: float,
    cue_duration: float,
    *,
    speech_offset: float,
    next_start: float | None,
    next_speaker: str | None,
    cue_speaker: str,
    speaker_gap_s: float,
    allow_trim: bool,
    atempo_max: float = 1.28,
    atempo_max_cross: float = 1.42,
) -> tuple[float, float, bool, float]:
    """Return ``(target, max_duration, allow_overflow, atempo_max)`` for one cue."""
    target = cue_duration
    placed_start = cue_start + speech_offset

    if allow_trim:
        return target, target, False, atempo_max

    if next_start is None:
        return target, max(target, cue_duration + 1.0), True, atempo_max

    until_next = max(0.05, next_start - speaker_gap_s - placed_start)
    same_speaker = bool(cue_speaker and next_speaker and cue_speaker == next_speaker)

    if same_speaker:
        return target, until_next, True, atempo_max

    # Speaker hand-off: do not bleed into the next voice; allow faster atempo first.
    return target, min(target, until_next), False, atempo_max_cross
