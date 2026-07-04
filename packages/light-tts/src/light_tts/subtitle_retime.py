from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assemble import PlacedSegment
from .cues_loader import Cue
from .merge_turns import SpeakerTurn


@dataclass(frozen=True)
class TimedCue:
    cue_id: str
    start: float
    end: float
    text: str


def retime_turn_cues(
    turn: SpeakerTurn,
    cues_by_id: dict[str, Cue],
    audio_start: float,
    audio_duration: float,
) -> list[TimedCue]:
    """Partition *audio_duration* across display cues in *turn* by text length."""
    cue_list = [cues_by_id[cid] for cid in turn.cue_ids if cid in cues_by_id]
    if not cue_list:
        return []
    if audio_duration <= 0:
        return [
            TimedCue(cue_id=cue.cue_id, start=audio_start, end=audio_start, text=cue.text.strip())
            for cue in cue_list
        ]

    weights = [max(1, len(c.text.strip())) for c in cue_list]
    total_w = sum(weights)
    audio_end = audio_start + audio_duration
    retimed: list[TimedCue] = []
    cursor = audio_start
    for index, cue in enumerate(cue_list):
        if index == len(cue_list) - 1:
            end = audio_end
        else:
            end = audio_start + audio_duration * sum(weights[: index + 1]) / total_w
        retimed.append(
            TimedCue(
                cue_id=cue.cue_id,
                start=cursor,
                end=end,
                text=cue.text.strip(),
            )
        )
        cursor = end
    return retimed


def build_dub_subtitles(
    turns: list[SpeakerTurn],
    cues: list[Cue],
    placed: list[PlacedSegment],
) -> list[TimedCue]:
    """Build retimed subtitle cues aligned to natural turn playback."""
    cues_by_id = {cue.cue_id: cue for cue in cues}
    timed: list[TimedCue] = []
    if len(placed) != len(turns):
        raise ValueError(f"Expected {len(turns)} placed segments, got {len(placed)}")
    for turn, segment in zip(turns, placed, strict=True):
        duration = len(segment.samples) / segment.sample_rate
        timed.extend(retime_turn_cues(turn, cues_by_id, segment.start, duration))
    return timed


def build_turn_placement_meta(
    turns: list[SpeakerTurn],
    placed: list[PlacedSegment],
) -> list[dict[str, object]]:
    """Return per-turn audio placement metadata for debugging."""
    rows: list[dict[str, object]] = []
    for turn, segment in zip(turns, placed, strict=True):
        duration = len(segment.samples) / segment.sample_rate
        rows.append(
            {
                "turn_id": turn.turn_id,
                "cue_ids": list(turn.cue_ids),
                "scheduled_start_s": round(turn.start, 3),
                "placed_start_s": round(segment.start, 3),
                "audio_duration_s": round(duration, 3),
                "placed_end_s": round(segment.start + duration, 3),
            }
        )
    return rows


def _seconds_to_srt(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def _seconds_to_vtt(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def write_srt(cues: list[TimedCue], path: Path) -> None:
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(f"{_seconds_to_srt(cue.start)} --> {_seconds_to_srt(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(cues: list[TimedCue], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.append(f"{_seconds_to_vtt(cue.start)} --> {_seconds_to_vtt(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_dub_subtitles(
    output_dir: Path,
    turns: list[SpeakerTurn],
    placed: list[PlacedSegment],
    cues: list[Cue],
    *,
    tts_dir: Path,
    lang: str = "zh",
) -> Path:
    """Write ``{lang}_dub.srt``, ``{lang}_dub.vtt``, and ``tts/dub_subtitle_meta.json``."""
    timed = build_dub_subtitles(turns, cues, placed)
    srt_path = output_dir / f"{lang}_dub.srt"
    vtt_path = output_dir / f"{lang}_dub.vtt"
    write_srt(timed, srt_path)
    write_vtt(timed, vtt_path)

    import json

    meta = {
        "align_mode": "turn_retime",
        "cue_count": len(timed),
        "turn_count": len(turns),
        "turns": build_turn_placement_meta(turns, placed),
    }
    meta_path = tts_dir / "dub_subtitle_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return srt_path
