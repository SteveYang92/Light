"""Build SubtitleCue objects from segments."""

from __future__ import annotations

from light_models import Segment, SubtitleCue


def build_source_cues(segments: list[Segment], lang: str) -> list[SubtitleCue]:
    return [
        SubtitleCue(
            cue_id=f"src_{i:04d}",
            unit_id=s.unit_id,
            start=s.start,
            end=s.end,
            text=s.source_text,
            lang=lang,
            speaker=s.speaker,
            words=list(s.words),
        )
        for i, s in enumerate(segments)
    ]
