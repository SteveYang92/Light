from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import MixMode

logger = logging.getLogger(__name__)

OUTPUT_SUFFIX = "_dub"
_VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v"}


def find_video(output_dir: Path, video_override: str | None) -> Path:
    if video_override:
        path = Path(video_override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        return path
    candidates: list[Path] = []
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        if OUTPUT_SUFFIX in entry.stem or "_pack" in entry.stem:
            continue
        if entry.stem.startswith("video") or entry.name.startswith("video."):
            candidates.insert(0, entry)
        else:
            candidates.append(entry)
    if not candidates:
        raise FileNotFoundError(f"No video file in {output_dir} (tried {_VIDEO_EXTENSIONS})")
    if len(candidates) > 1:
        names = "\n".join(f"  {f.name}" for f in candidates)
        raise RuntimeError(f"Multiple videos found:\n{names}\nUse --video to specify.")
    return candidates[0]


def mix_dub(
    video_path: Path,
    dub_wav: Path,
    output_path: Path,
    *,
    mode: MixMode,
    duck_db: float = -18.0,
) -> None:
    """Mux dubbed audio with *video_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == MixMode.REPLACE:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dub_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    elif mode == MixMode.DUCK:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dub_wav),
            "-filter_complex",
            f"[0:a]volume={duck_db}dB[bg];[bg][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]",
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    elif mode == MixMode.DUAL:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dub_wav),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    else:
        raise ValueError(f"Unknown mix mode: {mode}")

    logger.info("Mixing: ffmpeg %s ...", " ".join(cmd[1:4]))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg mix failed: {stderr[-500:]}") from exc
