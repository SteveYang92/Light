from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from .audio_io import read_wav, write_wav
from .config import MixMode
from .mix import OUTPUT_SUFFIX, find_video, mix_dub

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v"}


def discover_segments(output_dir: Path) -> list[Path]:
    """Find ``.seg*/`` directories under *output_dir*, sorted by name."""
    seg_dirs = sorted(d for d in output_dir.iterdir() if d.name.startswith(".seg") and d.is_dir())
    if not seg_dirs:
        seg_dirs = sorted(d for d in output_dir.iterdir() if d.name.startswith("chunk_") and d.is_dir())
    return seg_dirs


def load_split_points(output_dir: Path) -> tuple[list[float], float]:
    path = output_dir / "split_points.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    split_points = data.get("split_points")
    if not split_points or len(split_points) < 2:
        raise ValueError(f"Invalid split_points in {path}")
    overlap = float(data.get("overlap", 10.0))
    return [float(x) for x in split_points], overlap


def _find_root_video(output_dir: Path) -> Path | None:
    for name in ("video.webm", "video.mp4", "video.mkv"):
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() in _VIDEO_EXTENSIONS and OUTPUT_SUFFIX not in entry.stem:
            return entry
    return None


def _probe_keyframes(video_path: Path) -> list[float]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,flags",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    keyframes: list[float] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                keyframes.append(float(parts[0]))
            except ValueError:
                continue
    return sorted(keyframes)


def _segment_durations(seg_dirs: list[Path]) -> list[float]:
    durations: list[float] = []
    for seg in seg_dirs:
        video = find_video(seg, None)
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        durations.append(float(result.stdout.strip()))
    return durations


def compute_segment_offsets(
    output_dir: Path,
    seg_dirs: list[Path],
    split_points: list[float],
    overlap: float,
) -> list[float]:
    """Match subtitle merge offsets in ``light_subtitle.merge_outputs``."""
    n = len(seg_dirs)
    if len(split_points) != n + 1:
        raise ValueError(f"Expected {n + 1} split points for {n} segments, got {len(split_points)}")

    durations = _segment_durations(seg_dirs)
    original_kfs: list[float] | None = None
    root_video = _find_root_video(output_dir)
    if root_video is not None:
        original_kfs = _probe_keyframes(root_video)

    offsets = [0.0]
    if n == 1:
        return offsets

    for k in range(1, n):
        requested_start = split_points[k] - overlap
        actual_start = requested_start
        if original_kfs:
            for kf in original_kfs:
                if kf <= requested_start + 0.001:
                    actual_start = kf
                else:
                    break
        offsets.append(actual_start)

    if original_kfs:
        deltas = [f"{offsets[k] - (split_points[k] - overlap):+.3f}s" for k in range(1, n)]
        logger.info(
            "Keyframe-corrected offsets: %s (deltas: %s)",
            [f"{o:.3f}" for o in offsets],
            deltas,
        )
    else:
        # Fallback when root video is unavailable.
        cum = [0.0]
        trimmed: list[float] = []
        for k, dur in enumerate(durations):
            if k == 0:
                trimmed.append(dur - overlap if n > 1 else dur)
            elif k == n - 1:
                trimmed.append(dur - overlap)
            else:
                trimmed.append(dur - 2 * overlap)
        for td in trimmed[:-1]:
            cum.append(cum[-1] + td)
        offsets = [0.0]
        for k in range(1, n):
            offsets.append(cum[k] - overlap)

    return offsets


def merge_dub_timeline(output_dir: Path) -> tuple[Path, int]:
    """Merge per-segment ``tts/dub.wav`` into ``tts/dub_full.wav`` with overlap trim."""
    output_dir = output_dir.resolve()
    seg_dirs = discover_segments(output_dir)
    if not seg_dirs:
        raise FileNotFoundError(f"No .seg*/ directories in {output_dir}")

    split_points, overlap = load_split_points(output_dir)
    if len(split_points) != len(seg_dirs) + 1:
        raise ValueError(f"split_points length {len(split_points)} does not match {len(seg_dirs)} segments (+1)")

    offsets = compute_segment_offsets(output_dir, seg_dirs, split_points, overlap)
    total_duration = split_points[-1]
    sample_rate: int | None = None
    master: np.ndarray | None = None

    for k, seg in enumerate(seg_dirs):
        dub_path = seg / "tts" / "dub.wav"
        if not dub_path.is_file():
            raise FileNotFoundError(f"Missing segment dub: {dub_path}")
        samples, sr = read_wav(dub_path)
        if sample_rate is None:
            sample_rate = sr
            master = np.zeros(int(total_duration * sr) + sr, dtype=np.float32)
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch in {dub_path}: {sr} != {sample_rate}")

        assert master is not None and sample_rate is not None
        g_start = split_points[k]
        g_end = split_points[k + 1]
        offset = offsets[k]
        local_start = max(0.0, g_start - offset)
        local_end = min(len(samples) / sample_rate, g_end - offset)
        if local_end <= local_start:
            logger.warning("Segment %s contributes no dub audio after overlap trim", seg.name)
            continue

        start_idx = int(round(local_start * sample_rate))
        end_idx = int(round(local_end * sample_rate))
        chunk = samples[start_idx:end_idx]
        master_start = int(round(g_start * sample_rate))
        master_end = master_start + len(chunk)
        if master_end > len(master):
            chunk = chunk[: len(master) - master_start]
            master_end = master_start + len(chunk)
        master[master_start:master_end] = chunk
        logger.info(
            "Merged %s: global [%.2f, %.2f]s ← local [%.2f, %.2f]s (%d samples)",
            seg.name,
            g_start,
            g_end,
            local_start,
            local_end,
            len(chunk),
        )

    assert master is not None and sample_rate is not None
    target_samples = int(round(total_duration * sample_rate))
    master = master[:target_samples]
    out_dir = output_dir / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dub_full.wav"
    write_wav(out_path, master, sample_rate)
    return out_path, sample_rate


def resolve_episode_slug(output_dir: Path) -> str:
    root_video = _find_root_video(output_dir)
    if root_video is not None and root_video.stem != "video":
        return root_video.stem
    return output_dir.name


def merge_episode_dub(
    output_dir: Path,
    *,
    mix_mode: MixMode = MixMode.DUCK,
    duck_db: float = -18.0,
    video: str | None = None,
) -> Path:
    """Build ``tts/dub_full.wav`` and mux with the episode root video."""
    dub_full, _ = merge_dub_timeline(output_dir)
    root_video = Path(video).expanduser().resolve() if video else _find_root_video(output_dir)
    if root_video is None or not root_video.is_file():
        raise FileNotFoundError(f"No episode video in {output_dir}")
    slug = resolve_episode_slug(output_dir)
    out_mp4 = output_dir / f"{slug}{OUTPUT_SUFFIX}.mp4"
    mix_dub(root_video, dub_full, out_mp4, mode=mix_mode, duck_db=duck_db)
    return out_mp4


def prepare_ref_audio(episode_dir: Path, *, source_seg: str = ".seg1") -> list[Path]:
    """Copy reference audio from *source_seg* into all other segments."""
    episode_dir = episode_dir.resolve()
    src = episode_dir / source_seg / "tts"
    if not (src / "ref.wav").is_file():
        raise FileNotFoundError(f"Missing reference audio: {src / 'ref.wav'}")

    copied: list[Path] = []
    ref_files = ["ref.wav", "ref_text.txt", "ref_meta.json"]
    for seg in discover_segments(episode_dir):
        if seg.name == source_seg:
            continue
        dest = seg / "tts"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ref_files:
            src_file = src / name
            if src_file.is_file():
                dest_file = dest / name
                dest_file.write_bytes(src_file.read_bytes())
        copied.append(dest)
    return copied
