"""Single-track mergers — per-language SRT/VTT, cues.json, transcript.json."""

from __future__ import annotations

import json
from pathlib import Path

from light_core import logger
from light_subtitle import artifacts as sub_artifacts

from .dedup import _dedup_json_overlaps, _dedup_srt_overlaps, _dedup_vtt_overlaps
from .parse import _EPS, _parse_srt, _parse_vtt, _write_srt, _write_vtt


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
        src = sub_artifacts.find_sidecar(seg, f"{lang}.srt") or (seg / f"{lang}.srt")
        cues = _parse_srt(src)
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
        logger.warning(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)")

    if not all_cues:
        return  # language track absent in all segments (e.g. en.srt only in bilingual runs)

    all_cues.sort(key=lambda c: c[0])
    all_cues = _dedup_srt_overlaps(all_cues)
    out = sub_artifacts.sidecar_path(output_dir, f"{lang}.srt")
    _write_srt(all_cues, out)
    logger.info(f"  Merged SRT: {len(all_cues)} cues → {out.name}")
    _ = slug  # API compat


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
        src = sub_artifacts.find_sidecar(seg, f"{lang}.vtt") or (seg / f"{lang}.vtt")
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
        logger.warning(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)")

    if not all_cues:
        return  # language track absent in all segments

    all_cues.sort(key=lambda c: c[0])
    all_cues = _dedup_vtt_overlaps(all_cues)
    out = sub_artifacts.sidecar_path(output_dir, f"{lang}.vtt")
    _write_vtt(all_cues, out)
    logger.info(f"  Merged VTT: {len(all_cues)} cues → {out.name}")
    _ = slug  # API compat


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
        logger.warning(f"  ⚠ Skipped {skipped_invalid} cues with backwards timestamps (end < start)")

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

    out = output_dir / "cues.json"
    out.write_text(
        json.dumps({"media": media_info, "speakers": speakers, "cues": all_cues}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"  Merged cues.json: {len(all_cues)} cues → {out.name}")


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
    out = output_dir / "transcript.json"
    out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"  Merged transcript: {len(all_words)} words → {out.name}")
