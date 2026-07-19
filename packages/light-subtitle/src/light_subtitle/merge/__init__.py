"""Merge segment outputs into unified video + subtitle + transcript files.

Reads ``.seg1/``, ``.seg2/``, … directories under ``output_dir``, writes merged
files named ``{slug}.mp4`` / ``.zh.srt`` / ``.zh.vtt`` / ``.cues.json`` /
``.transcript.json`` (+ ``.annotations.ass`` / ``.annotations.vtt`` if present).

The original video is reused directly — segments are only processed for ASR.
Subtitle offsets are computed from ``split_points.json`` (saved by the split
step) so that segment-local timestamps map to the original video timeline.

Implementation split by concern: :mod:`.parse` (srt/vtt IO + time utils),
:mod:`.dedup` (overlap/term dedup), :mod:`.tracks` (single-track mergers),
:mod:`.bilingual` and :mod:`.annotations` (bilingual & annotation mergers).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import logger
from ..usage.report import load_usage_from_dir, merge_reports, write_usage_report
from ..usage.tracker import USAGE_REPORT_FILENAME
from .annotations import _merge_annotations_ass, _merge_annotations_vtt
from .bilingual import _merge_bilingual_ass, _merge_bilingual_vtt
from .tracks import _merge_cues_json, _merge_srt, _merge_transcript, _merge_vtt

# ── Segment discovery ───────────────────────────────────

_VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v"}


def _find_video_file(seg_dir: Path) -> Path | None:
    """Return the actual video file in *seg_dir*, or None."""
    for entry in seg_dir.iterdir():
        if entry.suffix in _VIDEO_EXTENSIONS and entry.name.startswith("video"):
            return entry
    return None


def _discover_segments(output_dir: Path) -> list[Path]:
    """Find ``.seg*/`` directories, sorted by name."""
    seg_dirs = sorted(d for d in output_dir.iterdir() if d.name.startswith(".seg") and d.is_dir())
    if not seg_dirs:
        seg_dirs = sorted(d for d in output_dir.iterdir() if d.name.startswith("chunk_") and d.is_dir())
    return seg_dirs


def _get_segment_durations(seg_dirs: list[Path]) -> list[float]:
    """Get video duration for each segment via ffprobe."""
    durations: list[float] = []
    for seg in seg_dirs:
        video_file = _find_video_file(seg)
        if not video_file:
            durations.append(0.0)
            continue
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        durations.append(float(result.stdout.strip()))
    return durations


# ── Public API ──────────────────────────────────────────


def merge_all(output_dir: Path, slug: str, overlap: float = 10) -> None:
    """Merge all segment outputs in *output_dir* into ``{slug}.*`` files.

    Single-segment → copy files directly (no merge needed).
    """
    seg_dirs = _discover_segments(output_dir)
    if not seg_dirs:
        logger.info(f"No .seg/ chunk_/ directories found in {output_dir}")
        return

    logger.info(f"Found {len(seg_dirs)} segments: {[d.name for d in seg_dirs]}")

    if len(seg_dirs) == 1:
        _copy_single_segment(output_dir, seg_dirs[0], slug)
        _merge_usage_reports(output_dir, seg_dirs)
        return

    _merge_multi(output_dir, seg_dirs, slug, overlap)
    _merge_usage_reports(output_dir, seg_dirs)


# ── Multi-segment merge ─────────────────────────────────


def _probe_keyframes(video_path: Path) -> list[float]:
    """Return sorted keyframe PTS values for *video_path* using ffprobe."""
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
        timeout=30,
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


def _merge_multi(output_dir: Path, seg_dirs: list[Path], slug: str, overlap: float) -> None:
    durations = _get_segment_durations(seg_dirs)
    N = len(seg_dirs)

    # ── Read split points (saved by video_split) ──
    split_points: list[float] | None = None
    split_points_path = output_dir / "split_points.json"
    if split_points_path.exists():
        data = json.loads(split_points_path.read_text(encoding="utf-8"))
        split_points = data.get("split_points")

    # ── Probe original video keyframes for accurate offsets ──
    # When video_split uses -c copy, each segment starts at a keyframe
    # BEFORE the requested position.  We find the actual keyframe time
    # so that subtitle offsets map segment-local timestamps to the real
    # original video timeline.
    original_kfs: list[float] | None = None
    for ext in _VIDEO_EXTENSIONS:
        candidate = output_dir / f"video{ext}"
        if candidate.exists():
            original_kfs = _probe_keyframes(candidate)
            break

    # ── Compute subtitle offsets corrected for keyframe alignment ──
    offsets = [0.0]
    if split_points and len(split_points) == N + 1:
        for k in range(1, N):
            requested_start = split_points[k] - overlap
            actual_start = requested_start  # fallback
            if original_kfs:
                # Find the keyframe at or before the requested start.
                for kf in original_kfs:
                    if kf <= requested_start + 0.001:
                        actual_start = kf
                    else:
                        break
            offsets.append(actual_start)
    else:
        # Fallback: estimate from cumulative durations
        cum = [0.0]
        trimmed: list[float] = []
        for k, dur in enumerate(durations):
            if k == 0:
                trimmed.append(dur - overlap if N > 1 else dur)
            elif k == N - 1:
                trimmed.append(dur - overlap)
            else:
                trimmed.append(dur - 2 * overlap)
        for td in trimmed[:-1]:
            cum.append(cum[-1] + td)
        for k in range(1, N):
            offsets.append(cum[k] - overlap)

    if split_points and original_kfs:
        deltas = [f"{offsets[k] - (split_points[k] - overlap):+.3f}s" for k in range(1, N)]
        logger.info(f"  Keyframe-corrected offsets: {[f'{o:.3f}' for o in offsets]} (deltas: {deltas})")

    # ── Copy the original video ──
    _copy_original_video(output_dir, slug)

    # ── Merge subtitle / data files ──
    # zh track (translation target) — always attempted.
    _merge_srt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="zh")
    _merge_vtt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="zh")
    # en track (source) — present in bilingual runs; no-op if en.srt/en.vtt absent.
    _merge_srt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="en")
    _merge_vtt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="en")
    # bilingual.ass / bilingual.vtt (merged ZH+EN display) — present in bilingual runs.
    _merge_bilingual_ass(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_bilingual_vtt(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_cues_json(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_transcript(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_annotations_ass(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_annotations_vtt(output_dir, seg_dirs, offsets, durations, split_points, slug)

    logger.info(f"\nMerge complete → {output_dir / slug}.*")


# ── Single-segment fast path ────────────────────────────


def _copy_single_segment(output_dir: Path, seg_dir: Path, slug: str) -> None:
    """Copy files from a single segment dir to root with semantic names."""
    import shutil

    mapping = {
        "zh.srt": f"{slug}.zh.srt",
        "zh.vtt": f"{slug}.zh.vtt",
        "en.srt": f"{slug}.en.srt",
        "en.vtt": f"{slug}.en.vtt",
        "bilingual.ass": f"{slug}.bilingual.ass",
        "bilingual.vtt": f"{slug}.bilingual.vtt",
        "transcript.json": f"{slug}.transcript.json",
        "cues.json": f"{slug}.cues.json",
        "annotations.ass": f"{slug}.annotations.ass",
        "annotations.vtt": f"{slug}.annotations.vtt",
    }
    for src_name, dst_name in mapping.items():
        src = seg_dir / src_name
        dst = output_dir / dst_name
        if not src.exists():
            continue
        # Overwrite stale root files — e.g. after ``--resume-from subtitle`` in
        # ``.seg1/`` refreshes bare ``zh.srt`` / ``bilingual.ass`` but root
        # ``{slug}.*`` already exists from an earlier merge.
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)

    usage_src = seg_dir / USAGE_REPORT_FILENAME
    if usage_src.exists():
        usage_dst = output_dir / USAGE_REPORT_FILENAME
        if usage_dst.exists():
            usage_dst.unlink()
        shutil.copy2(usage_src, usage_dst)

    # Copy video
    video_file = _find_video_file(seg_dir)
    if video_file:
        dst = output_dir / f"{slug}{video_file.suffix}"
        if not dst.exists():
            shutil.copy2(video_file, dst)

    logger.info(f"  Single segment: copied from {seg_dir.name}/ → {slug}.*")


def _merge_usage_reports(output_dir: Path, seg_dirs: list[Path]) -> None:
    """Aggregate per-segment usage_report.json into the output root."""
    reports = []
    for seg_dir in seg_dirs:
        report = load_usage_from_dir(seg_dir)
        if report is not None:
            reports.append(report)
    if not reports:
        return
    merged = merge_reports(reports)
    write_usage_report(merged, output_dir / USAGE_REPORT_FILENAME)
    logger.info(f"  Merged usage report → {USAGE_REPORT_FILENAME}")


# ── Video: reuse original ───────────────────────────────


def _copy_original_video(output_dir: Path, slug: str) -> None:
    """Copy the original video to ``{slug}.<ext>``.

    The original video (``video.*`` at the work directory root) is reused
    directly — segment videos are only processed for ASR and discarded.
    """
    import shutil

    src: Path | None = None
    for ext in _VIDEO_EXTENSIONS:
        candidate = output_dir / f"video{ext}"
        if candidate.exists():
            src = candidate
            break

    if src is None:
        logger.warning("  ⚠ No original video found to copy")
        return

    dst = output_dir / f"{slug}{src.suffix}"
    if not dst.exists():
        shutil.copy2(src, dst)
        logger.info(f"  Copied video → {dst.name}")
