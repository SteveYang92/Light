from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 *samples* as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


def trim_edge_silence(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.012,
    pad_ms: float = 20.0,
) -> np.ndarray:
    """Remove leading/trailing silence so concatenated cues play back-to-back."""
    if len(samples) == 0:
        return samples
    mask = np.abs(samples) > threshold
    if not mask.any():
        # All silence (model failure) — return empty rather than padding the timeline with dead air.
        return samples[:0]
    start = int(np.argmax(mask))
    end = int(len(samples) - np.argmax(mask[::-1]))
    pad = int(sample_rate * pad_ms / 1000.0)
    start = max(0, start - pad)
    end = min(len(samples), end + pad)
    return samples[start:end]


def concat_with_crossfade(
    chunks: list[np.ndarray],
    sample_rate: int,
    *,
    crossfade_ms: float = 30.0,
) -> np.ndarray:
    """Join cue clips with a short crossfade (no timeline gaps)."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    cf = max(0, int(sample_rate * crossfade_ms / 1000.0))
    out = chunks[0]
    for nxt in chunks[1:]:
        if cf <= 0 or len(out) < cf or len(nxt) < cf:
            out = np.concatenate([out, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
        blended = out[-cf:] * fade_out + nxt[:cf] * fade_in
        out = np.concatenate([out[:-cf], blended, nxt[cf:]])
    return out
