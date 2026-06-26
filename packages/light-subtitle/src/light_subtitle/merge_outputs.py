"""Merge segment outputs into unified video + subtitle + transcript files.

Reads ``.seg1/``, ``.seg2/``, … directories under ``output_dir``, writes merged
files named ``{slug}.mp4`` / ``.zh.srt`` / ``.zh.vtt`` / ``.cues.json`` /
``.transcript.json`` (+ ``.annotations.ass`` / ``.annotations.vtt`` if present).

The original video is reused directly — segments are only processed for ASR.
Subtitle offsets are computed from ``split_points.json`` (saved by the split
step) so that segment-local timestamps map to the original video timeline.
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
        print(f"  Keyframe-corrected offsets: {[f'{o:.3f}' for o in offsets]} (deltas: {deltas})", file=sys.stderr)

    # ── Copy the original video ──
    _copy_original_video(output_dir, slug)

    # ── Merge subtitle / data files ──
    # zh track (translation target) — always attempted.
    _merge_srt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="zh")
    _merge_vtt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="zh")
    # en track (source) — present in bilingual runs; no-op if en.srt/en.vtt absent.
    _merge_srt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="en")
    _merge_vtt(output_dir, seg_dirs, offsets, durations, split_points, slug, lang="en")
    # bilingual.ass (merged ZH+EN display) — present in bilingual runs.
    _merge_bilingual_ass(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_cues_json(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_transcript(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_annotations_ass(output_dir, seg_dirs, offsets, durations, split_points, slug)
    _merge_annotations_vtt(output_dir, seg_dirs, offsets, durations, split_points, slug)

    print(f"\nMerge complete → {output_dir / slug}.*", file=sys.stderr)


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
        print("  ⚠ No original video found to copy", file=sys.stderr)
        return

    dst = output_dir / f"{slug}{src.suffix}"
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Copied video → {dst.name}", file=sys.stderr)


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


_EPS = 0.5  # tolerance for split-point boundary filtering (seconds)


def _dedup_srt_overlaps(cues: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Remove adjacent overlapping SRT cues, keeping the later one in each pair."""
    if len(cues) < 2:
        return cues
    deduped: list[tuple[float, float, str]] = [cues[0]]
    for cue in cues[1:]:
        if deduped[-1][1] > cue[0] + 0.001:
            deduped[-1] = cue
        else:
            deduped.append(cue)
    return deduped


def _dedup_vtt_overlaps(cues: list[tuple[float, float, str, str]]) -> list[tuple[float, float, str, str]]:
    """Remove adjacent overlapping VTT cues, keeping the later one in each pair."""
    if len(cues) < 2:
        return cues
    deduped: list[tuple[float, float, str, str]] = [cues[0]]
    for cue in cues[1:]:
        if deduped[-1][1] > cue[0] + 0.001:
            deduped[-1] = cue
        else:
            deduped.append(cue)
    return deduped


def _dedup_json_overlaps(cues: list[dict]) -> list[dict]:
    """Remove adjacent overlapping JSON cues, keeping the later one in each pair."""
    if len(cues) < 2:
        return cues
    deduped: list[dict] = [cues[0]]
    for cue in cues[1:]:
        if deduped[-1].get("end", 0) > cue.get("start", 0) + 0.001:
            deduped[-1] = cue
        else:
            deduped.append(cue)
    return deduped


def _merge_srt(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
    lang: str = "zh",
) -> None:
    all_cues: list[tuple[float, float, str]] = []
    N = len(seg_dirs)
    skipped_invalid = 0

    for k, seg in enumerate(seg_dirs):
        cues = _parse_srt(seg / f"{lang}.srt")
        offset = offsets[k]
        seg_dur = durations[k]
        for start, end, text in cues:
            global_start = start + offset
            global_end = end + offset
            # Filter by split-point boundaries (precise, no margin gap).
            if split_points and N == len(split_points) - 1:
                if k > 0 and global_start < split_points[k] - _EPS:
                    continue
                if k < N - 1 and global_start > split_points[k + 1] + _EPS:
                    continue
            else:
                # Fallback: fixed margin around overlap region.
                if k > 0 and start < 12:
                    continue
                if k < N - 1 and start > seg_dur - 12:
                    continue
            if global_end < global_start:
                skipped_invalid += 1
                continue
            all_cues.append((global_start, global_end, text))

    if skipped_invalid:
        print(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)", file=sys.stderr)

    if not all_cues:
        return  # language track absent in all segments (e.g. en.srt only in bilingual runs)

    all_cues.sort(key=lambda c: c[0])
    all_cues = _dedup_srt_overlaps(all_cues)
    out = output_dir / f"{slug}.{lang}.srt"
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
    split_points: list[float] | None,
    slug: str,
    lang: str = "zh",
) -> None:
    all_cues: list[tuple[float, float, str, str]] = []
    N = len(seg_dirs)
    skipped_invalid = 0

    for k, seg in enumerate(seg_dirs):
        cues = _parse_vtt(seg / f"{lang}.vtt")
        offset = offsets[k]
        seg_dur = durations[k]
        for start, end, text, settings in cues:
            global_start = start + offset
            global_end = end + offset
            if split_points and N == len(split_points) - 1:
                if k > 0 and global_start < split_points[k] - _EPS:
                    continue
                if k < N - 1 and global_start > split_points[k + 1] + _EPS:
                    continue
            else:
                if k > 0 and start < 12:
                    continue
                if k < N - 1 and start > seg_dur - 12:
                    continue
            if global_end < global_start:
                skipped_invalid += 1
                continue
            all_cues.append((global_start, global_end, text, settings))

    if skipped_invalid:
        print(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)", file=sys.stderr)

    if not all_cues:
        return  # language track absent in all segments

    all_cues.sort(key=lambda c: c[0])
    all_cues = _dedup_vtt_overlaps(all_cues)
    out = output_dir / f"{slug}.{lang}.vtt"
    _write_vtt(all_cues, out)
    print(f"  Merged VTT: {len(all_cues)} cues → {out.name}", file=sys.stderr)


# ── cues.json merge ─────────────────────────────────────


def _merge_cues_json(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
) -> None:
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
        for cue in cues:
            start = cue.get("start", 0)
            global_start = start + offset
            global_end = cue.get("end", 0) + offset
            if split_points and N == len(split_points) - 1:
                if k > 0 and global_start < split_points[k] - _EPS:
                    continue
                if k < N - 1 and global_start > split_points[k + 1] + _EPS:
                    continue
            else:
                if k > 0 and start < 12:
                    continue
                if k < N - 1 and start > seg_dur - 12:
                    continue
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
    all_cues = _dedup_json_overlaps(all_cues)

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
    split_points: list[float] | None,
    slug: str,
) -> None:
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

        for w in words:
            w_start = w.get("start", 0)
            w_global = w_start + offset
            if split_points and N == len(split_points) - 1:
                if k > 0 and w_global < split_points[k] - _EPS:
                    continue
                if k < N - 1 and w_global > split_points[k + 1] + _EPS:
                    continue
            else:
                if k > 0 and w_start < 12:
                    continue
                if k < N - 1 and w_start > seg_dur - 12:
                    continue
            w["start"] = w_global
            w["end"] = w.get("end", 0) + offset
            all_words.append(w)

        for seg_obj in segments:
            seg_start = seg_obj.get("start", 0)
            s_global = seg_start + offset
            if split_points and N == len(split_points) - 1:
                if k > 0 and s_global < split_points[k] - _EPS:
                    continue
                if k < N - 1 and s_global > split_points[k + 1] + _EPS:
                    continue
            else:
                if k > 0 and seg_start < 12:
                    continue
                if k < N - 1 and seg_start > seg_dur - 12:
                    continue
            seg_obj["start"] = s_global
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

_ANNOTATION_MARKER_RE = re.compile(r"^\s*(?:※\s*)+")


def _strip_annotation_marker(text: str) -> str:
    """Remove leading ※ markers from annotation body text."""
    return _ANNOTATION_MARKER_RE.sub("", text).strip()


def _extract_annotation_term(text: str) -> str:
    """Extract normalized term from formatted annotation text.

    "※ RL训练：强化学习的方法" → "rl训练"
    """
    body = _strip_annotation_marker(text)
    if "：" in body:
        return body.split("：")[0].strip().lower()
    if ":" in body:
        return body.split(":")[0].strip().lower()
    return body.strip().lower()


def _dedup_annotation_terms(cues: list[tuple]) -> list[tuple]:
    """Remove duplicate annotations by normalized term, keeping first occurrence."""
    seen: set[str] = set()
    removed = 0
    deduped: list = []
    for cue in cues:
        key = _extract_annotation_term(cue[2])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(cue)
    if removed:
        print(f"    Deduplicated {removed} annotation(s) by term", file=sys.stderr)
    return deduped


def _dedup_bilingual_ass_overlaps(
    events: list[tuple[float, float, str, list[str]]],
) -> list[tuple[float, float, str, list[str]]]:
    """Remove adjacent overlapping ASS Dialogue events (main subtitle stream).

    Mirrors ``_dedup_srt_overlaps``: when two adjacent events overlap (prev.end
    > cur.start + tol), keep the later one — bilingual main cues from
    overlapping segments should not double-display.
    """
    if len(events) < 2:
        return events
    deduped: list[tuple[float, float, str, list[str]]] = [events[0]]
    for event in events[1:]:
        if deduped[-1][1] > event[0] + 0.001:
            deduped[-1] = event
        else:
            deduped.append(event)
    return deduped


def _merge_bilingual_ass(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
) -> None:
    """Merge per-segment ``bilingual.ass`` into ``{slug}.bilingual.ass``.

    Mirrors ``_merge_annotations_ass`` (ASS Dialogue time-shift via
    ``split(",", 9)``) but applies main-subtitle semantics: split-point
    boundary filtering like the SRT/VTT mergers, and overlap dedup via
    ``_dedup_bilingual_ass_overlaps`` (keep later cue on overlap) rather than
    annotation term dedup.
    """
    has_any = any((seg / "bilingual.ass").exists() for seg in seg_dirs)
    if not has_any:
        return

    N = len(seg_dirs)
    header_lines: list[str] = []
    all_events: list[tuple[float, float, str, list[str]]] = []
    in_header = True

    for k, seg in enumerate(seg_dirs):
        ass_path = seg / "bilingual.ass"
        if not ass_path.exists():
            continue
        offset = offsets[k]
        seg_dur = durations[k]
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
            global_start = start + offset
            if split_points and N == len(split_points) - 1:
                if k > 0 and global_start < split_points[k] - _EPS:
                    continue
                if k < N - 1 and global_start > split_points[k + 1] + _EPS:
                    continue
            else:
                if k > 0 and start < 12:
                    continue
                if k < N - 1 and start > seg_dur - 12:
                    continue
            end = _ass_to_seconds(fields[2]) + offset
            fields[1] = _seconds_to_ass(global_start)
            fields[2] = _seconds_to_ass(end)
            all_events.append((global_start, end, fields[9], fields))

    if not all_events:
        return

    all_events.sort(key=lambda e: e[0])
    all_events = _dedup_bilingual_ass_overlaps(all_events)
    event_lines = [",".join(fields) + "\n" for _, _, _, fields in all_events]

    out = output_dir / f"{slug}.bilingual.ass"
    out.write_text("".join(header_lines + event_lines), encoding="utf-8")
    print(f"  Merged bilingual.ass: {len(event_lines)} cues → {out.name}", file=sys.stderr)


def _merge_annotations_ass(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
) -> None:
    has_any = any((seg / "annotations.ass").exists() for seg in seg_dirs)
    if not has_any:
        return

    N = len(seg_dirs)
    header_lines: list[str] = []
    all_events: list[tuple[float, float, str, list[str]]] = []
    in_header = True

    for k, seg in enumerate(seg_dirs):
        ass_path = seg / "annotations.ass"
        if not ass_path.exists():
            continue
        offset = offsets[k]
        seg_dur = durations[k]
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
            global_start = start + offset
            if split_points and N == len(split_points) - 1:
                if k > 0 and global_start < split_points[k] - _EPS:
                    continue
                if k < N - 1 and global_start > split_points[k + 1] + _EPS:
                    continue
            else:
                if k > 0 and start < 12:
                    continue
                if k < N - 1 and start > seg_dur - 12:
                    continue
            end = _ass_to_seconds(fields[2]) + offset
            fields[1] = _seconds_to_ass(global_start)
            fields[2] = _seconds_to_ass(end)
            all_events.append((global_start, end, fields[9], fields))

    if not all_events:
        return

    all_events.sort(key=lambda e: e[0])
    all_events = _dedup_annotation_terms(all_events)
    event_lines = [",".join(fields) + "\n" for _, _, _, fields in all_events]

    out = output_dir / f"{slug}.annotations.ass"
    out.write_text("".join(header_lines + event_lines), encoding="utf-8")
    print(f"  Merged annotations.ass: {len(event_lines)} entries → {out.name}", file=sys.stderr)


def _merge_annotations_vtt(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
) -> None:
    all_cues: list[tuple[float, float, str, str]] = []
    N = len(seg_dirs)

    has_any = any((seg / "annotations.vtt").exists() for seg in seg_dirs)
    if not has_any:
        return

    for k, seg in enumerate(seg_dirs):
        cues = _parse_vtt(seg / "annotations.vtt")
        offset = offsets[k]
        seg_dur = durations[k]
        for start, end, text, settings in cues:
            global_start = start + offset
            if split_points and N == len(split_points) - 1:
                if k > 0 and global_start < split_points[k] - _EPS:
                    continue
                if k < N - 1 and global_start > split_points[k + 1] + _EPS:
                    continue
            else:
                if k > 0 and start < 12:
                    continue
                if k < N - 1 and start > seg_dur - 12:
                    continue
            all_cues.append((global_start, end + offset, text, settings))

    if not all_cues:
        return

    all_cues.sort(key=lambda c: c[0])
    all_cues = _dedup_vtt_overlaps(all_cues)
    all_cues = _dedup_annotation_terms(all_cues)
    out = output_dir / f"{slug}.annotations.vtt"
    _write_vtt(all_cues, out)
    print(f"  Merged annotations.vtt: {len(all_cues)} cues → {out.name}", file=sys.stderr)
