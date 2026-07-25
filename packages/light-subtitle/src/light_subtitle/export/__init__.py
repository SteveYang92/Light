"""Subtitle/transcript artifact writers.

Public API is defined in sibling modules and re-exported here so existing
``from light_subtitle.export import ...`` callers keep working:
:mod:`.formats` (SRT/VTT/ASS/JSON), :mod:`.bilingual` (bilingual ASS/VTT),
:mod:`.annotations` (annotation ASS/VTT), :mod:`.transcript` (transcript/
segment/raw-cue JSON).
"""

from .annotations import (
    export_annotation_ass,
    export_annotation_vtt,
    format_annotation_display,
    strip_annotation_marker,
)
from .bilingual import (
    BILINGUAL_VTT_MARKER,
    BilingualGroup,
    export_bilingual_ass,
    export_bilingual_vtt,
)
from .formats import export_ass, export_json, export_json_file, export_srt, export_vtt
from .transcript import export_raw_cues, export_segments, export_transcript

__all__ = [
    "BILINGUAL_VTT_MARKER",
    "BilingualGroup",
    "export_annotation_ass",
    "export_annotation_vtt",
    "export_ass",
    "export_bilingual_ass",
    "export_bilingual_vtt",
    "export_json",
    "export_json_file",
    "export_raw_cues",
    "export_segments",
    "export_srt",
    "export_transcript",
    "export_vtt",
    "format_annotation_display",
    "strip_annotation_marker",
]
