"""Audio extraction for ASR — 16kHz mono WAV via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

ASR_WAV_NAME = "audio_asr.wav"


def has_audio_stream(input_path: str) -> bool:
    """Return True if the media file has at least one audio stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() != ""


def extract_audio_16k(input_path: str, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip()
        if stderr_tail:
            # Surface the last meaningful line(s) of ffmpeg output
            lines = stderr_tail.split("\n")
            last_lines = [ln.strip() for ln in lines[-5:] if ln.strip() and "libav" not in ln]
            detail = "; ".join(last_lines) if last_lines else stderr_tail[-300:]
            raise RuntimeError(f"ffmpeg failed (exit {e.returncode}): {detail}") from e
        raise RuntimeError(f"ffmpeg failed with exit status {e.returncode}") from e


def extract_audio(input_path: str, out_dir: str | Path) -> str:
    """Extract 16kHz mono audio for ASR to ``<out_dir>/audio_asr.wav``."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not has_audio_stream(input_path):
        raise RuntimeError(f"Input file has no audio track — cannot generate subtitles: {input_path}")

    asr_wav = str(output / ASR_WAV_NAME)
    extract_audio_16k(input_path, asr_wav)

    return asr_wav
