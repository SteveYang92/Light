import subprocess


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
