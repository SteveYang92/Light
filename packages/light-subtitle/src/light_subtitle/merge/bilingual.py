"""Bilingual track mergers — bilingual.ass (cue-group aware) and bilingual.vtt."""

from __future__ import annotations

from pathlib import Path

from light_models import seconds_to_ass

from .. import artifacts, logger
from .dedup import _BilingualGroup, _dedup_bilingual_ass_overlaps, _dedup_vtt_overlaps
from .parse import _EPS, _ass_to_seconds, _parse_vtt, _write_vtt


def _merge_bilingual_ass(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
) -> None:
    """Merge per-segment bilingual ASS into root ``video.bilingual.ass``.

    Mirrors ``_merge_annotations_ass`` (ASS Dialogue time-shift via
    ``split(",", 9)``) but applies main-subtitle semantics: split-point
    boundary filtering like the SRT/VTT mergers, and overlap dedup via
    ``_dedup_bilingual_ass_overlaps`` (keep later cue on overlap) rather than
    annotation term dedup.

    One display cue spans several Dialogue events with identical start/end
    (rounded-box drawing + text per language), so consecutive events sharing
    one time range within a segment are grouped and survive dedup together.
    """
    has_any = any(artifacts.find_sidecar(seg, "bilingual.ass") for seg in seg_dirs)
    if not has_any:
        return

    N = len(seg_dirs)
    header_lines: list[str] = []
    all_groups: list[_BilingualGroup] = []
    in_header = True

    for k, seg in enumerate(seg_dirs):
        ass_path = artifacts.find_sidecar(seg, "bilingual.ass")
        if ass_path is None:
            continue
        offset = offsets[k]
        seg_dur = durations[k]
        current_key: tuple[float, float] | None = None
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
                end = _ass_to_seconds(fields[2])
            except ValueError:
                continue
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
            fields[1] = seconds_to_ass(global_start)
            fields[2] = seconds_to_ass(global_end)
            key = (global_start, global_end)
            if key == current_key and all_groups:
                all_groups[-1][2].append(fields)
            else:
                all_groups.append((global_start, global_end, [fields]))
                current_key = key

    if not all_groups:
        return

    all_groups.sort(key=lambda g: (g[0], g[1]))
    all_groups = _dedup_bilingual_ass_overlaps(all_groups)
    event_lines = [",".join(fields) + "\n" for _, _, group in all_groups for fields in group]

    out = artifacts.sidecar_path(output_dir, "bilingual.ass")
    out.write_text("".join(header_lines + event_lines), encoding="utf-8")
    logger.info(f"  Merged bilingual.ass: {len(all_groups)} cues → {out.name}")
    _ = slug


def _merge_bilingual_vtt(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
) -> None:
    """Merge per-segment bilingual VTT into root ``video.bilingual.vtt``."""
    has_any = any(artifacts.find_sidecar(seg, "bilingual.vtt") for seg in seg_dirs)
    if not has_any:
        return

    all_cues: list[tuple[float, float, str, str]] = []
    N = len(seg_dirs)
    skipped_invalid = 0

    for k, seg in enumerate(seg_dirs):
        src = artifacts.find_sidecar(seg, "bilingual.vtt") or (seg / "bilingual.vtt")
        cues = _parse_vtt(src)
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
        logger.warning(f"  ⚠ Skipped {skipped_invalid} bilingual VTT cues with backwards timestamps")

    if not all_cues:
        return

    all_cues.sort(key=lambda c: c[0])
    all_cues = _dedup_vtt_overlaps(all_cues)
    out = artifacts.sidecar_path(output_dir, "bilingual.vtt")
    _write_vtt(all_cues, out)
    logger.info(f"  Merged bilingual.vtt: {len(all_cues)} cues → {out.name}")
    _ = slug
