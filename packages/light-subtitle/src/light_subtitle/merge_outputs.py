"""Merge segment outputs into unified video + subtitle + transcript files.

Reads ``.seg1/``, ``.seg2/``, … directories under ``output_dir``, writes merged
files named ``{slug}.mp4`` / ``.zh.srt`` / ``.zh.vtt`` / ``.cues.json`` /
``.transcript.json`` (+ ``.annotations.ass`` / ``.annotations.vtt`` if present).

Ports the logic from ``x-subtitle/scripts/merge_segments.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# ── Time utilities ──────────────────────────────────────

_SRT_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_ASS_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[.](\d+)")


def _srt_to_seconds(ts: str) -> float:
    m = _SRT_TIME_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid timestamp: {ts}")
    h, mm, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mm * 60 + s + ms / 1000


def _seconds_to_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _seconds_to_vtt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _ass_to_seconds(ts: str) -> float:
    m = _ASS_TIME_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid ASS timestamp: {ts}")
    h, mm, s, cs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mm * 60 + s + cs / 100


def _seconds_to_ass(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


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
        print(f"No .seg/ chunk_/ directories found in {output_dir}", file=sys.stderr)
        return

    print(f"Found {len(seg_dirs)} segments: {[d.name for d in seg_dirs]}", file=sys.stderr)

    if len(seg_dirs) == 1:
        _copy_single_segment(output_dir, seg_dirs[0], slug)
        return

    _merge_multi(output_dir, seg_dirs, slug, overlap)


# ── Multi-segment merge ─────────────────────────────────


def _merge_multi(output_dir: Path, seg_dirs: list[Path], slug: str, overlap: float) -> None:
    durations = _get_segment_durations(seg_dirs)
    N = len(seg_dirs)

    # ── Merge video first (returns keyframe-aligned inpoints) ──
    actual_inpoints = _merge_videos(output_dir, seg_dirs, durations, overlap, slug)

    # ── Compute effective durations and subtitle offsets ──
    # Effective contribution of each segment after trimming:
    #   seg 0:     [0,          dur[0] - overlap]
    #   seg k>0:   [inpoint[k], dur[k] - overlap]   (middle)
    #   seg N-1:   [inpoint[k], dur[k]]             (last, no outpoint)
    effective: list[float] = []
    for k in range(N):
        if k == 0:
            effective.append(durations[k] - overlap if N > 1 else durations[k])
        elif k == N - 1:
            effective.append(durations[k] - actual_inpoints[k])
        else:
            effective.append(durations[k] - actual_inpoints[k] - overlap)

    cum_effective = [0.0]
    for e in effective[:-1]:
        cum_effective.append(cum_effective[-1] + e)

    # Subtitle offset for segment k: global time = offset[k] + local_time
    offsets = [0.0]
    for k in range(1, N):
        offsets.append(cum_effective[k] - actual_inpoints[k])

    # ── Merge subtitle / data files ──
    _merge_srt(output_dir, seg_dirs, offsets, durations, overlap, actual_inpoints, slug)
    _merge_vtt(output_dir, seg_dirs, offsets, durations, overlap, actual_inpoints, slug)
    _merge_cues_json(output_dir, seg_dirs, offsets, durations, overlap, actual_inpoints, slug)
    _merge_transcript(output_dir, seg_dirs, offsets, durations, overlap, actual_inpoints, slug)

    # ── Merge annotations (if present) ──
    _merge_annotations_ass(output_dir, seg_dirs, offsets, durations, overlap, actual_inpoints, slug)
    _merge_annotations_vtt(output_dir, seg_dirs, offsets, durations, overlap, actual_inpoints, slug)

    print(f"\nMerge complete → {output_dir / slug}.*", file=sys.stderr)


# ── Single-segment fast path ────────────────────────────


def _copy_single_segment(output_dir: Path, seg_dir: Path, slug: str) -> None:
    """Copy files from a single segment dir to root with semantic names."""
    import shutil

    mapping = {
        "zh.srt": f"{slug}.zh.srt",
        "zh.vtt": f"{slug}.zh.vtt",
        "transcript.json": f"{slug}.transcript.json",
        "cues.json": f"{slug}.cues.json",
        "annotations.ass": f"{slug}.annotations.ass",
        "annotations.vtt": f"{slug}.annotations.vtt",
    }
    for src_name, dst_name in mapping.items():
        src = seg_dir / src_name
        dst = output_dir / dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Copy video
    video_file = _find_video_file(seg_dir)
    if video_file:
        dst = output_dir / f"{slug}{video_file.suffix}"
        if not dst.exists():
            shutil.copy2(video_file, dst)

    print(f"  Single segment: copied from {seg_dir.name}/ → {slug}.*", file=sys.stderr)


# ── Video merge ─────────────────────────────────────────


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


def _merge_videos(
    output_dir: Path,
    seg_dirs: list[Path],
    durations: list[float],
    overlap: float,
    slug: str,
) -> list[float]:
    """Merge segment videos using keyframe-aligned inpoints.

    Returns the actual inpoints used (keyframe times closest to *overlap*),
    so that subtitle offsets can be adjusted to match the real video timeline.
    """
    N = len(seg_dirs)
    actual_inpoints = [0.0] * N

    # Probe keyframes for middle/last segments to find safe inpoints.
    # A safe inpoint lands on a keyframe, avoiding undecodable P/B-frames
    # that reference a dropped reference frame.  We pick the keyframe
    # closest to *overlap* (either before or after) to minimise deviation.
    for k in range(1, N):
        video_file = _find_video_file(seg_dirs[k])
        if not video_file:
            actual_inpoints[k] = overlap
            continue
        kfs = _probe_keyframes(video_file)
        if not kfs:
            actual_inpoints[k] = overlap
            continue
        # Find the keyframe closest to *overlap*.
        closest = min(kfs, key=lambda x: abs(x - overlap))
        actual_inpoints[k] = closest

    if any(p != overlap for p in actual_inpoints[1:]):
        print(
            f"  Keyframe-aligned inpoints: {[f'{p:.3f}' for p in actual_inpoints]} (vs nominal {overlap:.1f}s overlap)",
            file=sys.stderr,
        )

    concat_list = output_dir / ".concat_list.txt"

    with open(concat_list, "w") as f:
        for k, seg in enumerate(seg_dirs):
            video_file = _find_video_file(seg)
            if not video_file:
                continue
            f.write(f"file '{video_file.absolute()}'\n")
            if k > 0:
                f.write(f"inpoint {actual_inpoints[k]:.3f}\n")
            if k < N - 1:
                f.write(f"outpoint {durations[k] - overlap:.3f}\n")

    out_path = output_dir / f"{slug}.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out_path)],
        capture_output=True,
        timeout=600,
    )
    concat_list.unlink()

    if result.returncode != 0:
        print(f"Video merge failed: {result.stderr.decode()[-500:]}", file=sys.stderr)
    elif out_path.exists():
        print(f"  Merged video → {out_path}", file=sys.stderr)

    return actual_inpoints


# ── SRT merge ───────────────────────────────────────────


def _parse_srt(path: Path) -> list[tuple[float, float, str]]:
    if not path.exists():
        return []
    cues: list[tuple[float, float, str]] = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Find the timestamp line
        ts_idx = 0 if "-->" in lines[0] else 1
        if ts_idx >= len(lines):
            continue
        m = re.match(r"(.+?)\s*-->\s*(.+)", lines[ts_idx])
        if not m:
            continue
        try:
            start = _srt_to_seconds(m.group(1))
            end = _srt_to_seconds(m.group(2))
        except ValueError:
            continue
        text_lines = lines[ts_idx + 1 :]
        text = "\n".join(text_lines).strip()
        if text:
            cues.append((start, end, text))
    return cues


def _write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(cues, 1):
            f.write(f"{i}\n")
            f.write(f"{_seconds_to_srt(start)} --> {_seconds_to_srt(end)}\n")
            f.write(f"{text}\n\n")


def _merge_srt(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    overlap: float,
    actual_inpoints: list[float],
    slug: str,
) -> None:
    margin_end = overlap + 2
    all_cues: list[tuple[float, float, str]] = []
    N = len(seg_dirs)
    skipped_invalid = 0

    for k, seg in enumerate(seg_dirs):
        cues = _parse_srt(seg / "zh.srt")
        offset = offsets[k]
        seg_dur = durations[k]
        margin_start = actual_inpoints[k] + 2 if k > 0 else 0
        for start, end, text in cues:
            if k > 0 and start < margin_start:
                continue
            if k < N - 1 and start > seg_dur - margin_end:
                continue
            global_start = start + offset
            global_end = end + offset
            if global_end < global_start:
                skipped_invalid += 1
                continue
            all_cues.append((global_start, global_end, text))

    if skipped_invalid:
        print(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)", file=sys.stderr)

    all_cues.sort(key=lambda c: c[0])
    out = output_dir / f"{slug}.zh.srt"
    _write_srt(all_cues, out)
    print(f"  Merged SRT: {len(all_cues)} cues → {out.name}", file=sys.stderr)


# ── VTT merge ───────────────────────────────────────────


def _parse_vtt(path: Path) -> list[tuple[float, float, str, str]]:
    """Return (start, end, text, settings)."""
    if not path.exists():
        return []
    cues: list[tuple[float, float, str, str]] = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or lines[0].strip() == "WEBVTT":
            continue
        ts_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_idx = i
                break
        if ts_idx < 0:
            continue
        m = re.match(r"(.+?)\s*-->\s*(\S+)(.*)", lines[ts_idx])
        if not m:
            continue
        settings = m.group(3).strip()
        try:
            start = _srt_to_seconds(m.group(1).strip())
            end = _srt_to_seconds(m.group(2).strip())
        except ValueError:
            continue
        text = "\n".join(lines[ts_idx + 1 :]).strip()
        if text:
            cues.append((start, end, text, settings))
    return cues


def _write_vtt(cues: list[tuple[float, float, str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (start, end, text, settings) in enumerate(cues, 1):
            ts = f"{_seconds_to_vtt(start)} --> {_seconds_to_vtt(end)}"
            if settings:
                ts += " " + settings
            f.write(f"{i}\n{ts}\n{text}\n\n")


def _merge_vtt(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    overlap: float,
    actual_inpoints: list[float],
    slug: str,
) -> None:
    margin_end = overlap + 2
    all_cues: list[tuple[float, float, str, str]] = []
    N = len(seg_dirs)
    skipped_invalid = 0

    for k, seg in enumerate(seg_dirs):
        cues = _parse_vtt(seg / "zh.vtt")
        offset = offsets[k]
        seg_dur = durations[k]
        margin_start = actual_inpoints[k] + 2 if k > 0 else 0
        for start, end, text, settings in cues:
            if k > 0 and start < margin_start:
                continue
            if k < N - 1 and start > seg_dur - margin_end:
                continue
            global_start = start + offset
            global_end = end + offset
            if global_end < global_start:
                skipped_invalid += 1
                continue
            all_cues.append((global_start, global_end, text, settings))

    if skipped_invalid:
        print(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)", file=sys.stderr)

    all_cues.sort(key=lambda c: c[0])
    out = output_dir / f"{slug}.zh.vtt"
    _write_vtt(all_cues, out)
    print(f"  Merged VTT: {len(all_cues)} cues → {out.name}", file=sys.stderr)


# ── cues.json merge ─────────────────────────────────────


def _merge_cues_json(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    overlap: float,
    actual_inpoints: list[float],
    slug: str,
) -> None:
    margin_end = overlap + 2
    all_cues: list[dict] = []
    N = len(seg_dirs)
    cue_counter = 0
    skipped_invalid = 0

    for k, seg in enumerate(seg_dirs):
        cues_path = seg / "cues.json"
        if not cues_path.exists():
            continue
        data = json.loads(cues_path.read_text(encoding="utf-8"))
        cues = data.get("cues", []) if isinstance(data, dict) else data
        if not isinstance(cues, list):
            continue
        offset = offsets[k]
        seg_dur = durations[k]
        margin_start = actual_inpoints[k] + 2 if k > 0 else 0
        for cue in cues:
            start = cue.get("start", 0)
            if k > 0 and start < margin_start:
                continue
            if k < N - 1 and start > seg_dur - margin_end:
                continue
            global_start = start + offset
            global_end = cue.get("end", 0) + offset
            if global_end < global_start:
                skipped_invalid += 1
                continue
            cue_counter += 1
            cue["id"] = cue_counter
            cue["start"] = global_start
            cue["end"] = global_end
            all_cues.append(cue)

    if skipped_invalid:
        print(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)", file=sys.stderr)

    all_cues.sort(key=lambda c: c.get("start", 0))

    # Preserve media/speaker metadata from first segment
    media_info: dict = {}
    speakers: list[dict] = []
    if seg_dirs:
        first = seg_dirs[0] / "cues.json"
        if first.exists():
            data = json.loads(first.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                media_info = data.get("media", {})
                speakers = data.get("speakers", [])

    out = output_dir / f"{slug}.cues.json"
    out.write_text(
        json.dumps({"media": media_info, "speakers": speakers, "cues": all_cues}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Merged cues.json: {len(all_cues)} cues → {out.name}", file=sys.stderr)


# ── transcript.json merge ───────────────────────────────


def _merge_transcript(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    overlap: float,
    actual_inpoints: list[float],
    slug: str,
) -> None:
    margin_end = overlap + 2
    all_words: list[dict] = []
    all_segments: list[dict] = []
    N = len(seg_dirs)
    language = "en"

    for k, seg in enumerate(seg_dirs):
        tx_path = seg / "transcript.json"
        if not tx_path.exists():
            continue
        data = json.loads(tx_path.read_text(encoding="utf-8"))
        if k == 0:
            language = data.get("language", "en")

        words = data.get("words", [])
        segments = data.get("segments", [])
        offset = offsets[k]
        seg_dur = durations[k]
        margin_start = actual_inpoints[k] + 2 if k > 0 else 0

        for w in words:
            w_start = w.get("start", 0)
            if k > 0 and w_start < margin_start:
                continue
            if k < N - 1 and w_start > seg_dur - margin_end:
                continue
            w["start"] = w_start + offset
            w["end"] = w.get("end", 0) + offset
            all_words.append(w)

        for seg_obj in segments:
            seg_start = seg_obj.get("start", 0)
            if k > 0 and seg_start < margin_start:
                continue
            if k < N - 1 and seg_start > seg_dur - margin_end:
                continue
            seg_obj["start"] = seg_start + offset
            seg_obj["end"] = seg_obj.get("end", 0) + offset
            all_segments.append(seg_obj)

    all_words.sort(key=lambda w: w.get("start", 0))
    all_segments.sort(key=lambda s: s.get("start", 0))

    out_data = {
        "format": "light-transcript.v1",
        "source": "",
        "language": language,
        "words": all_words,
        "segments": all_segments,
    }
    out = output_dir / f"{slug}.transcript.json"
    out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Merged transcript: {len(all_words)} words → {out.name}", file=sys.stderr)


# ── Annotation merge ────────────────────────────────────


def _merge_annotations_ass(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    overlap: float,
    actual_inpoints: list[float],
    slug: str,
) -> None:
    margin_end = overlap + 2
    has_any = any((seg / "annotations.ass").exists() for seg in seg_dirs)
    if not has_any:
        return

    N = len(seg_dirs)
    header_lines: list[str] = []
    all_events: list[str] = []
    in_header = True

    for k, seg in enumerate(seg_dirs):
        ass_path = seg / "annotations.ass"
        if not ass_path.exists():
            continue
        offset = offsets[k]
        seg_dur = durations[k]
        margin_start = actual_inpoints[k] + 2 if k > 0 else 0
        for line in ass_path.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.startswith("Dialogue:"):
                if in_header:
                    header_lines.append(line)
                continue
            in_header = False
            fields = line.strip().split(",", 9)
            if len(fields) < 10:
                continue
            try:
                start = _ass_to_seconds(fields[1])
            except ValueError:
                continue
            if k > 0 and start < margin_start:
                continue
            if k < N - 1 and start > seg_dur - margin_end:
                continue
            start += offset
            end = _ass_to_seconds(fields[2]) + offset
            fields[1] = _seconds_to_ass(start)
            fields[2] = _seconds_to_ass(end)
            all_events.append(",".join(fields) + "\n")

    if not all_events:
        return

    out = output_dir / f"{slug}.annotations.ass"
    out.write_text("".join(header_lines + all_events), encoding="utf-8")
    print(f"  Merged annotations.ass: {len(all_events)} entries → {out.name}", file=sys.stderr)


def _merge_annotations_vtt(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    overlap: float,
    actual_inpoints: list[float],
    slug: str,
) -> None:
    margin_end = overlap + 2
    all_cues: list[tuple[float, float, str, str]] = []
    N = len(seg_dirs)

    has_any = any((seg / "annotations.vtt").exists() for seg in seg_dirs)
    if not has_any:
        return

    for k, seg in enumerate(seg_dirs):
        cues = _parse_vtt(seg / "annotations.vtt")
        offset = offsets[k]
        seg_dur = durations[k]
        margin_start = actual_inpoints[k] + 2 if k > 0 else 0
        for start, end, text, settings in cues:
            if k > 0 and start < margin_start:
                continue
            if k < N - 1 and start > seg_dur - margin_end:
                continue
            all_cues.append((start + offset, end + offset, text, settings))

    if not all_cues:
        return

    all_cues.sort(key=lambda c: c[0])
    out = output_dir / f"{slug}.annotations.vtt"
    _write_vtt(all_cues, out)
    print(f"  Merged annotations.vtt: {len(all_cues)} cues → {out.name}", file=sys.stderr)
