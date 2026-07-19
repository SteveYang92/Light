"""Merge segment outputs — backward-compatible re-export shell.

The implementation moved to the :mod:`light_subtitle.merge` package; this
module only keeps historical import paths working (runner, tests).
"""

from .merge import (
    _copy_original_video,
    _copy_single_segment,
    _discover_segments,
    _find_video_file,
    _get_segment_durations,
    _merge_multi,
    _merge_usage_reports,
    _probe_keyframes,
    merge_all,
)
from .merge.annotations import _merge_annotations_ass, _merge_annotations_vtt
from .merge.bilingual import _merge_bilingual_ass, _merge_bilingual_vtt
from .merge.dedup import (
    _BilingualGroup,
    _dedup_annotation_terms,
    _dedup_bilingual_ass_overlaps,
    _dedup_json_overlaps,
    _dedup_srt_overlaps,
    _dedup_vtt_overlaps,
    _extract_annotation_term,
    _strip_annotation_marker,
)
from .merge.parse import (
    _EPS,
    _ass_to_seconds,
    _parse_srt,
    _parse_vtt,
    _seconds_to_vtt,
    _srt_to_seconds,
    _write_srt,
    _write_vtt,
)
from .merge.tracks import _merge_cues_json, _merge_srt, _merge_transcript, _merge_vtt

__all__ = [
    "_BilingualGroup",
    "_EPS",
    "_ass_to_seconds",
    "_copy_original_video",
    "_copy_single_segment",
    "_dedup_annotation_terms",
    "_dedup_bilingual_ass_overlaps",
    "_dedup_json_overlaps",
    "_dedup_srt_overlaps",
    "_dedup_vtt_overlaps",
    "_discover_segments",
    "_extract_annotation_term",
    "_find_video_file",
    "_get_segment_durations",
    "_merge_annotations_ass",
    "_merge_annotations_vtt",
    "_merge_bilingual_ass",
    "_merge_bilingual_vtt",
    "_merge_cues_json",
    "_merge_multi",
    "_merge_srt",
    "_merge_transcript",
    "_merge_usage_reports",
    "_merge_vtt",
    "_parse_srt",
    "_parse_vtt",
    "_probe_keyframes",
    "_seconds_to_vtt",
    "_srt_to_seconds",
    "_strip_annotation_marker",
    "_write_srt",
    "_write_vtt",
    "merge_all",
]
