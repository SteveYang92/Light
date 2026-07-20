"""Annotation track mergers — annotations.ass (term dedup) and annotations.vtt."""

from __future__ import annotations

from pathlib import Path

from light_models import seconds_to_ass

from .. import artifacts, logger
from .dedup import _dedup_annotation_terms, _dedup_vtt_overlaps
from .parse import _EPS, _ass_to_seconds, _parse_vtt, _write_vtt


def _merge_annotations_ass(
    output_dir: Path,
    seg_dirs: list[Path],
    offsets: list[float],
    durations: list[float],
    split_points: list[float] | None,
    slug: str,
) -> None:
    has_any = any(artifacts.find_sidecar(seg, "annotations.ass") for seg in seg_dirs)
    if not has_any:
        return

    N = len(seg_dirs)
    header_lines: list[str] = []
    all_events: list[tuple[float, float, str, list[str]]] = []
    in_header = True

    for k, seg in enumerate(seg_dirs):
        ass_path = artifacts.find_sidecar(seg, "annotations.ass")
        if ass_path is None:
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
            fields[1] = seconds_to_ass(global_start)
            fields[2] = seconds_to_ass(end)
            all_events.append((global_start, end, fields[9], fields))

    if not all_events:
        return

    all_events.sort(key=lambda e: e[0])
    all_events = _dedup_annotation_terms(all_events)
    event_lines = [",".join(fields) + "\n" for _, _, _, fields in all_events]

    out = artifacts.sidecar_path(output_dir, "annotations.ass")
    out.write_text("".join(header_lines + event_lines), encoding="utf-8")
    logger.info(f"  Merged annotations.ass: {len(event_lines)} entries → {out.name}")
    _ = slug


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

    has_any = any(artifacts.find_sidecar(seg, "annotations.vtt") for seg in seg_dirs)
    if not has_any:
        return

    for k, seg in enumerate(seg_dirs):
        src = artifacts.find_sidecar(seg, "annotations.vtt") or (seg / "annotations.vtt")
        cues = _parse_vtt(src)
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
    out = artifacts.sidecar_path(output_dir, "annotations.vtt")
    _write_vtt(all_cues, out)
    _ = slug
