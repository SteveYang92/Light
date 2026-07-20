import subprocess
from pathlib import Path


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


def probe_duration(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def probe_fps(input_path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    if not fps_str:
        return None
    num, denom = fps_str.split("/")
    return float(num) / float(denom)


def probe_video_size(input_path: str) -> tuple[int, int] | None:
    """Return ``(width, height)`` of the first video stream, or None if unavailable.

    Audio-only inputs and probe failures return None so callers can fall back
    to the default 16:9 PlayRes.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line or "," not in line:
        return None
    w_s, h_s = line.split(",", 1)
    try:
        width, height = int(w_s), int(h_s)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
