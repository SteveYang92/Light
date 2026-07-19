"""Transcript/segment JSON export — transcript.json, segment.json, raw cue review files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from light_models import Segment, Word

from ... import artifacts
from ...language.base import detect_source_lang


def export_transcript(
    words: list[Word], segments: list[Segment], output_path: str, source: str = "whisper.cpp"
) -> None:
    """Export a standardized transcript.json with word-level timestamps.

    This is the canonical transcription format consumed by light-qc's
    ``--transcript`` parameter.  It is ASR-agnostic: the pipeline
    normalises whisper.cpp output into a flat word list, so replacing
    the ASR backend does not require any changes in light-qc.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build word-index lookup for segment word_range.
    word_index: dict[int, int] = {}
    for i, w in enumerate(words):
        word_index[id(w)] = i

    data = {
        "format": "light-transcript.v1",
        "source": source,
        "language": detect_source_lang(words),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "words": [artifacts.word_to_dict(w) for w in words],
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker,
                "text": s.source_text,
                "word_range": [
                    word_index[id(s.words[0])] if s.words else 0,
                    word_index[id(s.words[-1])] if s.words else 0,
                ],
            }
            for s in segments
        ],
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_raw_cues(cues: list, output_path: str) -> None:
    """Export raw translated cues as JSON for LLM output review."""
    artifacts.write_raw_cues(output_path, cues)


def export_segments(words: list[Word], segments: list[Segment], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_words": len(words),
        "total_units": len(segments),
        "words": [artifacts.word_to_dict(w) for w in words],
        "units": [
            {
                "unit_id": s.unit_id,
                "start": s.start,
                "end": s.end,
                "duration": round(s.end - s.start, 3),
                "speaker": s.speaker,
                "word_count": len(s.words),
                "source_text": s.source_text,
                "word_range": {
                    "from": s.words[0].text if s.words else "",
                    "to": s.words[-1].text if s.words else "",
                },
            }
            for s in segments
        ],
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
