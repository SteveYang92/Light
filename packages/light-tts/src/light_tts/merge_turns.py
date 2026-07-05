from __future__ import annotations

from dataclasses import dataclass

from .cues_loader import Cue

_TERMINAL_PUNCT = "。！？；.!?;"


@dataclass(frozen=True)
class SpeakerTurn:
    """One dubbing unit: consecutive same-speaker subtitle cues merged for natural TTS."""

    turn_id: str
    speaker: str
    start: float
    slot_end: float
    last_cue_end: float
    text: str
    lang: str
    cue_ids: tuple[str, ...]

    @property
    def slot_duration(self) -> float:
        return max(0.0, self.slot_end - self.start)


def _join_turn_text(parts: list[str], lang: str) -> str:
    cleaned = [p.strip() for p in parts if p.strip()]
    if not cleaned:
        return ""
    text = "".join(cleaned) if lang == "zh" else " ".join(cleaned)
    if text and text[-1] not in _TERMINAL_PUNCT:
        text += "。" if lang == "zh" else "."
    return text


def _ends_sentence(cue: Cue) -> bool:
    text = cue.text.strip()
    return bool(text and text[-1] in _TERMINAL_PUNCT)


def merge_speaker_turns(
    cues: list[Cue],
    *,
    speaker_gap_s: float = 0.08,
    max_turn_duration_s: float | None = 120.0,
    max_turn_chars: int | None = 120,
    min_turn_chars: int = 40,
    max_inter_cue_gap_s: float = 2.0,
) -> list[SpeakerTurn]:
    """Merge consecutive same-speaker cues into speaker turns for dubbing.

    Display subtitle rules (max lines, CPS, min duration) do not apply. Turns are
    still capped by text length and span so each Qwen request stays short enough
    to avoid runaway generation.
    """
    if not cues:
        return []

    ordered = sorted(cues, key=lambda c: c.start)
    raw_groups: list[list[Cue]] = []
    current: list[Cue] = [ordered[0]]

    for cue in ordered[1:]:
        prev_sp = current[-1].speaker.strip() or "__default__"
        curr_sp = cue.speaker.strip() or "__default__"
        if curr_sp == prev_sp:
            current.append(cue)
        else:
            raw_groups.append(current)
            current = [cue]
    raw_groups.append(current)

    groups: list[list[Cue]] = []
    for group in raw_groups:
        if max_turn_duration_s is None and max_turn_chars is None:
            groups.append(group)
            continue
        chunk: list[Cue] = []
        chunk_chars = 0
        for cue in group:
            if not chunk:
                chunk = [cue]
                chunk_chars = len(cue.text.strip())
                continue
            span = cue.end - chunk[0].start
            inter_gap = max(0.0, cue.start - chunk[-1].end)
            next_chars = chunk_chars + len(cue.text.strip())
            duration_exceeded = max_turn_duration_s is not None and span > max_turn_duration_s
            chars_exceeded = (
                max_turn_chars is not None and chunk_chars >= min_turn_chars and next_chars > max_turn_chars
            )
            gap_exceeded = inter_gap > max_inter_cue_gap_s and chunk_chars >= min_turn_chars
            should_split = False
            if duration_exceeded or gap_exceeded:
                should_split = True
            elif chars_exceeded:
                # Prefer sentence boundaries, but force split when the chunk is
                # already very long so Qwen never sees a runaway-sized request.
                should_split = _ends_sentence(chunk[-1]) or (max_turn_chars is not None and next_chars > max_turn_chars)

            if should_split:
                groups.append(chunk)
                chunk = [cue]
                chunk_chars = len(cue.text.strip())
            else:
                chunk.append(cue)
                chunk_chars = next_chars
        if chunk:
            groups.append(chunk)

    turns: list[SpeakerTurn] = []
    for idx, group in enumerate(groups):
        next_group = groups[idx + 1] if idx + 1 < len(groups) else None
        start = group[0].start
        if next_group is not None:
            slot_end = max(start + 0.1, next_group[0].start - speaker_gap_s)
        else:
            slot_end = max(group[-1].end, start + 0.5)

        speaker = group[0].speaker.strip() or "__default__"
        turns.append(
            SpeakerTurn(
                turn_id=f"turn_{idx:04d}",
                speaker=speaker,
                start=start,
                slot_end=slot_end,
                last_cue_end=group[-1].end,
                text=_join_turn_text([c.text for c in group], group[0].lang),
                lang=group[0].lang,
                cue_ids=tuple(c.cue_id for c in group),
            )
        )

    return turns
