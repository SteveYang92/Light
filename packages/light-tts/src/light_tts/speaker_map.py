from __future__ import annotations

from typing import Protocol

from .config import TtsConfig
from .cues_loader import Cue

# Qwen3-TTS preset speakers (Chinese + English); used for auto_assign rotation.
VOICE_POOL: tuple[str, ...] = ("Vivian", "Uncle_Fu", "Serena", "Eric", "Ryan", "Aiden")


class HasSpeaker(Protocol):
    speaker: str


def build_indextts_speaker_map(items: list[HasSpeaker]) -> dict[str, str]:
    """Identity map for IndexTTS2 — synthesis resolves ref audio per speaker label."""
    labels = sorted({item.speaker.strip() or "__default__" for item in items})
    return {label: label for label in labels}


def build_speaker_voice_map(items: list[HasSpeaker], config: TtsConfig) -> dict[str, str]:
    """Map diarization labels (``SPEAKER_00``) to Qwen3-TTS voice names."""
    explicit = dict(config.speakers)
    mapping: dict[str, str] = {}
    used_voices: set[str] = set()
    pool_idx = 0

    def _next_pool_voice() -> str:
        nonlocal pool_idx
        for _ in range(len(VOICE_POOL)):
            voice = VOICE_POOL[pool_idx % len(VOICE_POOL)]
            pool_idx += 1
            if voice not in used_voices:
                return voice
        return VOICE_POOL[pool_idx % len(VOICE_POOL)]

    for item in items:
        label = item.speaker.strip() or "__default__"
        if label in mapping:
            continue
        if label in explicit:
            voice = explicit[label]
            mapping[label] = voice
            used_voices.add(voice)
            continue
        if label == "__default__":
            mapping[label] = config.default_voice
            used_voices.add(config.default_voice)
            continue
        if config.auto_assign:
            voice = _next_pool_voice()
            mapping[label] = voice
            used_voices.add(voice)
        else:
            mapping[label] = config.default_voice
            used_voices.add(config.default_voice)

    return mapping


def voice_for_speaker(speaker: str, speaker_map: dict[str, str], default_voice: str) -> str:
    label = speaker.strip() or "__default__"
    return speaker_map.get(label, default_voice)


def voice_for_cue(cue: Cue, speaker_map: dict[str, str], default_voice: str) -> str:
    return voice_for_speaker(cue.speaker, speaker_map, default_voice)


def language_for_voice(voice: str, config: TtsConfig) -> str:
    if voice in config.voices:
        return config.voices[voice].language
    if voice in {"Ryan", "Aiden"}:
        return "English"
    return "Chinese"
