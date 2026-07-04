from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from .assemble import PlacedSegment
from .audio_io import read_wav, trim_edge_silence, write_wav
from .config import AlignMode, TtsConfig
from .cues_loader import Cue
from .engine import TtsEngine, create_engine
from .indextts_runtime import resolve_ref_audio_path
from .merge_turns import SpeakerTurn
from .speaker_map import (
    build_indextts_speaker_map,
    build_speaker_voice_map,
    language_for_voice,
    voice_for_cue,
    voice_for_speaker,
)
from .sync import compute_subtitle_aligned_start, compute_turn_placed_start, fit_budget, fit_duration

logger = logging.getLogger(__name__)
console = Console()

_ZH_SEC_PER_CHAR = 0.22
# Qwen3-TTS-12Hz produces 12 audio frames per second; capping tokens prevents
# the model from running away to ~327s when it misses the end-of-speech token.
_TTS_FRAMES_PER_SEC = 12
_MAX_TOKENS_HEADROOM = 1.5


def _max_tokens_for_duration(
    target_s: float,
    *,
    min_tokens: int = 512,
    max_tokens: int = 2048,
    headroom: float = _MAX_TOKENS_HEADROOM,
) -> int:
    """Cap generation length so a missed stop token can't blow up to 327s.

    Values are bounded: too low can produce silence, too high can run away to the
    model's default 4096-token ceiling.
    """
    if target_s <= 0:
        return min_tokens
    return max(min_tokens, min(max_tokens, int(target_s * _TTS_FRAMES_PER_SEC * headroom)))


def _synthesis_speed(text: str, target_duration: float, *, max_speed: float = 1.25) -> float:
    if target_duration <= 0 or not text.strip():
        return 1.0
    est = len(text.strip()) * _ZH_SEC_PER_CHAR
    if est <= target_duration * 1.05:
        return 1.0
    return min(max_speed, est / target_duration)


def _is_monologue(cues: list[Cue]) -> bool:
    labels = {c.speaker.strip() or "__default__" for c in cues}
    return len(labels) <= 1


def _is_model_cap_outlier(duration_s: float, sample_rate: int, sample_count: int) -> bool:
    """Qwen3-mlx often emits ~327.68s of padding on long inputs."""
    if duration_s > 300.0:
        return True
    expected_max_samples = int(327.7 * sample_rate)
    return sample_count >= expected_max_samples - sample_rate


def _natural_target_duration(text: str) -> float:
    """Expected natural Chinese narration duration for Qwen token caps."""
    return max(0.5, len(text.strip()) * _ZH_SEC_PER_CHAR)


def _pipeline_sample_rate(placed: list[PlacedSegment], default: int) -> int:
    if not placed:
        return default
    rates = {seg.sample_rate for seg in placed}
    if len(rates) > 1:
        raise ValueError(f"Mixed sample rates in placed segments: {sorted(rates)}")
    return placed[0].sample_rate


def _place_turn_by_cues(
    turn: SpeakerTurn,
    cues_by_id: dict[str, Cue],
    samples: np.ndarray,
    sample_rate: int,
    config: TtsConfig,
    *,
    atempo_max: float,
    prev_end: float | None = None,
    next_turn_start: float | None = None,
) -> tuple[list[PlacedSegment], float | None]:
    """Split one synthesized turn across display cues without overlapping neighbours."""
    cue_list = [cues_by_id[cid] for cid in turn.cue_ids if cid in cues_by_id]
    if not cue_list:
        return [], prev_end

    def _fit_chunk(chunk: np.ndarray, cue: Cue, *, next_start: float | None) -> tuple[PlacedSegment, float]:
        scheduled = cue.start + config.speech_offset
        start = scheduled if prev_end is None else max(scheduled, prev_end + config.speaker_gap_s)
        if next_start is not None:
            until_next = next_start - config.speaker_gap_s - start
        else:
            until_next = max(cue.duration, cue.end - start + 0.25)
        max_dur = max(0.05, min(cue.duration, until_next))
        fitted = fit_duration(
            chunk,
            sample_rate,
            max_dur,
            max_duration=max_dur,
            atempo_min=config.atempo_min,
            atempo_max=atempo_max,
            allow_trim=True,
            pad_to_target=False,
            strict_cap=True,
        )
        end = start + len(fitted.samples) / sample_rate
        return PlacedSegment(start=start, samples=fitted.samples, sample_rate=sample_rate), end

    if len(cue_list) == 1:
        next_start = next_turn_start
        segment, end = _fit_chunk(samples, cue_list[0], next_start=next_start)
        return [segment], end

    weights = [max(1, len(c.text)) for c in cue_list]
    total_w = sum(weights)
    placed: list[PlacedSegment] = []
    idx = 0
    end = prev_end
    for i, cue in enumerate(cue_list):
        if i == len(cue_list) - 1:
            chunk = samples[idx:]
            next_start = next_turn_start
        else:
            n = int(round(len(samples) * weights[i] / total_w))
            chunk = samples[idx : idx + n]
            idx += n
            next_start = cue_list[i + 1].start
        segment, end = _fit_chunk(chunk, cue, next_start=next_start)
        placed.append(segment)
        prev_end = end
    return placed, end


def _place_turn_natural(
    turn: SpeakerTurn,
    samples: np.ndarray,
    sample_rate: int,
    config: TtsConfig,
    prev_end: float | None,
) -> tuple[PlacedSegment, float]:
    """Place a full turn without time-stretch or per-cue splitting."""
    scheduled = turn.start + config.speech_offset
    start = scheduled if prev_end is None else max(scheduled, prev_end + config.speaker_gap_s)
    segment = PlacedSegment(start=start, samples=samples.astype(np.float32), sample_rate=sample_rate)
    end = start + len(samples) / sample_rate
    return segment, end


def _synthesize_turn(
    turn: SpeakerTurn,
    *,
    tts_engine: TtsEngine,
    voice: str,
    language: str,
    instruct: str | None,
    config: TtsConfig,
    monologue: bool,
    out_path: Path,
    natural_target_s: float,
    max_tokens: int | None,
) -> tuple[np.ndarray, int, dict[str, object]]:
    """One Qwen call per pre-split turn; failures are recorded, not retried."""
    slot = turn.slot_duration
    if config.resume and out_path.is_file():
        samples, sample_rate = read_wav(out_path)
        if sample_rate != tts_engine.sample_rate:
            logger.warning(
                "%s resume skipped: sample rate %s != engine %s — regenerating",
                turn.turn_id,
                sample_rate,
                tts_engine.sample_rate,
            )
        else:
            stat = {
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "voice": voice,
                "cue_ids": list(turn.cue_ids),
                "text_chars": len(turn.text),
                "target_duration_s": round(slot, 3),
                "natural_target_s": round(natural_target_s, 3),
                "max_tokens": max_tokens,
                "raw_duration_s": round(len(samples) / sample_rate, 3),
                "status": "resumed",
                "retried": False,
            }
            return samples, sample_rate, stat

    speed_target = natural_target_s if monologue else slot
    speed = _synthesis_speed(turn.text, speed_target, max_speed=config.tts_speed_max if monologue else 1.35)
    style = None if monologue else instruct

    result = tts_engine.synthesize(
        turn.text,
        voice,
        language=language,
        instruct=style,
        speed=speed,
        max_tokens=max_tokens,
        seed=config.qwen_seed,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )
    samples = result.samples
    sample_rate = result.sample_rate
    actual = len(samples) / sample_rate if sample_rate > 0 else 0.0
    speech_samples = trim_edge_silence(samples, sample_rate)
    speech_actual = len(speech_samples) / sample_rate if sample_rate > 0 else 0.0
    too_long = speech_actual > max(slot, natural_target_s) * config.tts_outlier_ratio
    invalid = not config.is_official_indextts and (
        len(speech_samples) == 0 or _is_model_cap_outlier(actual, sample_rate, len(samples)) or (monologue and too_long)
    )
    if len(speech_samples) == 0 and config.is_official_indextts:
        invalid = True

    status = "ok"
    if invalid:
        logger.warning("%s invalid Qwen output %.1fs — skipping chunk", turn.turn_id, actual)
        samples = np.array([], dtype=np.float32)
        status = "failed"
    else:
        samples = speech_samples

    if len(samples) > 0:
        write_wav(out_path, samples, sample_rate)

    stat = {
        "turn_id": turn.turn_id,
        "speaker": turn.speaker,
        "voice": voice,
        "cue_ids": list(turn.cue_ids),
        "text_chars": len(turn.text),
        "target_duration_s": round(slot, 3),
        "natural_target_s": round(natural_target_s, 3),
        "max_tokens": max_tokens,
        "raw_duration_s": round(actual, 3),
        "speech_duration_s": round(speech_actual, 3),
        "status": status,
    }
    return samples, sample_rate, stat


def synthesize_turns(
    turns: list[SpeakerTurn],
    cues: list[Cue],
    config: TtsConfig,
    *,
    segments_dir: Path,
    engine: TtsEngine | None = None,
    stats_path: Path | None = None,
) -> tuple[dict[str, str], list[PlacedSegment], int, list[SpeakerTurn]]:
    """Synthesize merged speaker turns (default dub path)."""
    tts_engine = engine or create_engine(config)
    if config.is_official_indextts:
        speaker_map = build_indextts_speaker_map(turns)
    else:
        speaker_map = build_speaker_voice_map(turns, config)
    segments_dir.mkdir(parents=True, exist_ok=True)
    monologue = _is_monologue(cues)
    cues_by_id = {c.cue_id: c for c in cues}

    placed: list[PlacedSegment] = []
    placed_turns: list[SpeakerTurn] = []
    stats: list[dict[str, object]] = []
    sample_rate = tts_engine.sample_rate
    prev_end: float | None = None

    for turn_index, turn in enumerate(turns):
        if config.is_official_indextts:
            voice = turn.speaker.strip() or "__default__"
            language = "Chinese"
            instruct = None
        else:
            voice = voice_for_speaker(turn.speaker, speaker_map, config.default_voice)
            language = language_for_voice(voice, config)
            voice_cfg = config.voices.get(voice)
            instruct = voice_cfg.instruct if voice_cfg and voice_cfg.instruct else None
        out_path = segments_dir / f"{turn.turn_id}.wav"

        slot = turn.slot_duration
        atempo_max = config.atempo_max_monologue if monologue else config.atempo_max_cross
        natural_target = _natural_target_duration(turn.text) if monologue else slot
        use_token_cap = not config.is_official_indextts and (monologue or len(turn.text) >= config.chunk_min_chars())
        max_tokens = (
            _max_tokens_for_duration(
                natural_target,
                min_tokens=config.qwen_max_tokens_min,
                max_tokens=config.qwen_max_tokens_max,
                headroom=config.qwen_max_tokens_headroom,
            )
            if use_token_cap
            else None
        )

        samples, sample_rate, stat = _synthesize_turn(
            turn,
            tts_engine=tts_engine,
            voice=voice,
            language=language,
            instruct=instruct,
            config=config,
            monologue=monologue,
            out_path=out_path,
            natural_target_s=natural_target,
            max_tokens=max_tokens,
        )
        if len(samples) == 0:
            stat["final_duration_s"] = 0.0
            stat["placed_start_s"] = None
            stats.append(stat)
            continue

        align_mode = config.effective_align_mode
        if align_mode == AlignMode.TURN_RETIME and monologue:
            segment, prev_end = _place_turn_natural(turn, samples, sample_rate, config, prev_end)
            placed.append(segment)
            placed_turns.append(turn)
            stat["final_duration_s"] = round(len(samples) / sample_rate, 3)
            stat["placed_start_s"] = round(segment.start, 3)
            stat["trimmed"] = False
            stat["align_mode"] = align_mode.value
            stats.append(stat)
            continue

        if monologue and align_mode == AlignMode.SUBTITLE_ALIGNED:
            next_turn_start = turns[turn_index + 1].start if turn_index + 1 < len(turns) else None
            sub_segments, prev_end = _place_turn_by_cues(
                turn,
                cues_by_id,
                samples,
                sample_rate,
                config,
                atempo_max=atempo_max,
                prev_end=prev_end,
                next_turn_start=next_turn_start,
            )
            placed.extend(sub_segments)
            placed_turns.append(turn)
            if sub_segments:
                stat["final_duration_s"] = round(sum(len(s.samples) for s in sub_segments) / sample_rate, 3)
                stat["placed_start_s"] = round(sub_segments[0].start, 3)
            stat["trimmed"] = False
            stats.append(stat)
            continue

        if monologue:
            fitted = fit_duration(
                samples,
                sample_rate,
                slot,
                max_duration=slot,
                atempo_min=config.atempo_min,
                atempo_max=atempo_max,
                allow_trim=True,
                pad_to_target=False,
                strict_cap=True,
            )
        else:
            fitted = fit_duration(
                samples,
                sample_rate,
                slot,
                max_duration=slot,
                atempo_min=config.atempo_min,
                atempo_max=atempo_max,
                allow_trim=False,
                pad_to_target=False,
                strict_cap=False,
            )

        scheduled = turn.start + config.speech_offset
        if monologue:
            start = compute_subtitle_aligned_start(
                scheduled,
                prev_end,
                speaker_gap_s=config.speaker_gap_s,
            )
        else:
            start = compute_turn_placed_start(
                scheduled,
                prev_end,
                speaker_gap_s=config.speaker_gap_s,
                max_inter_speaker_pause_s=config.max_inter_speaker_pause_s,
            )
        prev_end = start + len(fitted.samples) / sample_rate
        placed.append(PlacedSegment(start=start, samples=fitted.samples, sample_rate=sample_rate))
        placed_turns.append(turn)
        stat["final_duration_s"] = round(len(fitted.samples) / sample_rate, 3)
        stat["placed_start_s"] = round(start, 3)
        stat["trimmed"] = fitted.trimmed
        stats.append(stat)

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return speaker_map, placed, _pipeline_sample_rate(placed, tts_engine.sample_rate), placed_turns


def synthesize_cues(
    cues: list[Cue],
    config: TtsConfig,
    *,
    segments_dir: Path,
    engine: TtsEngine | None = None,
) -> tuple[dict[str, str], list[PlacedSegment], int]:
    """Synthesize each display cue separately (debug / legacy)."""
    tts_engine = engine or create_engine(config)
    if config.is_official_indextts:
        speaker_map = build_indextts_speaker_map(cues)
    else:
        speaker_map = build_speaker_voice_map(cues, config)
    segments_dir.mkdir(parents=True, exist_ok=True)

    placed: list[PlacedSegment] = []
    sample_rate = tts_engine.sample_rate

    for i, cue in enumerate(cues):
        next_cue = cues[i + 1] if i + 1 < len(cues) else None
        if config.is_official_indextts:
            voice = cue.speaker.strip() or "__default__"
            language = "Chinese"
            instruct = None
        else:
            voice = voice_for_cue(cue, speaker_map, config.default_voice)
            language = language_for_voice(voice, config)
            voice_cfg = config.voices.get(voice)
            instruct = voice_cfg.instruct if voice_cfg and voice_cfg.instruct else None
        out_path = segments_dir / f"{cue.cue_id}.wav"

        target, max_dur, _, atempo_max = fit_budget(
            cue.start,
            cue.duration,
            speech_offset=config.speech_offset,
            next_start=next_cue.start if next_cue else None,
            next_speaker=next_cue.speaker if next_cue else None,
            cue_speaker=cue.speaker,
            speaker_gap_s=config.speaker_gap_s,
            allow_trim=config.allow_trim,
            atempo_max=config.atempo_max,
            atempo_max_cross=config.atempo_max_cross,
        )
        same_next = bool(next_cue and cue.speaker and cue.speaker == next_cue.speaker)

        if config.resume and out_path.is_file():
            samples, sr = read_wav(out_path)
            if sr != tts_engine.sample_rate:
                logger.warning(
                    "%s resume skipped: sample rate %s != engine %s — regenerating",
                    cue.cue_id,
                    sr,
                    tts_engine.sample_rate,
                )
            else:
                sample_rate = sr
                fitted = fit_duration(
                    samples,
                    sample_rate,
                    target,
                    max_duration=max_dur,
                    atempo_min=config.atempo_min,
                    atempo_max=atempo_max,
                    allow_trim=config.allow_trim,
                )
                start = cue.start + config.speech_offset
                placed.append(PlacedSegment(start=start, samples=fitted.samples, sample_rate=sample_rate))
                continue

        max_sp = 1.35 if (not same_next and not config.allow_trim) else 1.25
        speed = _synthesis_speed(cue.text, target, max_speed=max_sp)
        result = tts_engine.synthesize(
            cue.text,
            voice,
            language=language,
            instruct=instruct,
            speed=speed,
            max_tokens=(
                None
                if config.is_official_indextts
                else _max_tokens_for_duration(
                    target,
                    min_tokens=config.qwen_max_tokens_min,
                    max_tokens=config.qwen_max_tokens_max,
                    headroom=config.qwen_max_tokens_headroom,
                )
            ),
            seed=config.qwen_seed,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        samples = result.samples
        sample_rate = result.sample_rate
        write_wav(out_path, samples, sample_rate)

        fitted = fit_duration(
            samples,
            sample_rate,
            target,
            max_duration=max_dur,
            atempo_min=config.atempo_min,
            atempo_max=atempo_max,
            allow_trim=config.allow_trim,
        )
        start = cue.start + config.speech_offset
        placed.append(PlacedSegment(start=start, samples=fitted.samples, sample_rate=sample_rate))

    return speaker_map, placed, _pipeline_sample_rate(placed, tts_engine.sample_rate)


def reassemble_turns_from_segments(
    turns: list[SpeakerTurn],
    cues: list[Cue],
    config: TtsConfig,
    *,
    segments_dir: Path,
    stats_path: Path | None = None,
) -> tuple[list[PlacedSegment], int, list[SpeakerTurn]]:
    """Rebuild timeline from existing segment WAVs (no TTS engine)."""
    monologue = _is_monologue(cues)
    cues_by_id = {c.cue_id: c for c in cues}
    placed: list[PlacedSegment] = []
    placed_turns: list[SpeakerTurn] = []
    stats: list[dict[str, object]] = []
    sample_rate: int | None = None
    prev_end: float | None = None
    align_mode = config.effective_align_mode

    for turn_index, turn in enumerate(turns):
        out_path = segments_dir / f"{turn.turn_id}.wav"
        if not out_path.is_file():
            raise FileNotFoundError(f"Missing segment WAV for reassemble: {out_path}")
        samples, sr = read_wav(out_path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch in {out_path}: {sr} != {sample_rate}")

        slot = turn.slot_duration
        atempo_max = config.atempo_max_monologue if monologue else config.atempo_max_cross

        if align_mode == AlignMode.TURN_RETIME and monologue:
            segment, prev_end = _place_turn_natural(turn, samples, sr, config, prev_end)
            placed.append(segment)
            placed_turns.append(turn)
            stats.append(
                {
                    "turn_id": turn.turn_id,
                    "speaker": turn.speaker,
                    "cue_ids": list(turn.cue_ids),
                    "target_duration_s": round(slot, 3),
                    "raw_duration_s": round(len(samples) / sr, 3),
                    "final_duration_s": round(len(samples) / sr, 3),
                    "placed_start_s": round(segment.start, 3),
                    "trimmed": False,
                    "status": "reassembled",
                    "align_mode": align_mode.value,
                }
            )
            continue

        if monologue and align_mode == AlignMode.SUBTITLE_ALIGNED:
            next_turn_start = turns[turn_index + 1].start if turn_index + 1 < len(turns) else None
            sub_segments, prev_end = _place_turn_by_cues(
                turn,
                cues_by_id,
                samples,
                sr,
                config,
                atempo_max=atempo_max,
                prev_end=prev_end,
                next_turn_start=next_turn_start,
            )
            placed.extend(sub_segments)
            placed_turns.append(turn)
            stats.append(
                {
                    "turn_id": turn.turn_id,
                    "speaker": turn.speaker,
                    "cue_ids": list(turn.cue_ids),
                    "target_duration_s": round(slot, 3),
                    "raw_duration_s": round(len(samples) / sr, 3),
                    "final_duration_s": (
                        round(sum(len(s.samples) for s in sub_segments) / sr, 3) if sub_segments else 0.0
                    ),
                    "placed_start_s": round(sub_segments[0].start, 3) if sub_segments else None,
                    "trimmed": False,
                    "status": "reassembled",
                    "subtitle_aligned": True,
                }
            )
            continue

        if monologue:
            fitted = fit_duration(
                samples,
                sr,
                slot,
                max_duration=slot,
                atempo_min=config.atempo_min,
                atempo_max=atempo_max,
                allow_trim=True,
                pad_to_target=False,
                strict_cap=True,
            )
        else:
            fitted = fit_duration(
                samples,
                sr,
                slot,
                max_duration=slot,
                atempo_min=config.atempo_min,
                atempo_max=atempo_max,
                allow_trim=False,
                pad_to_target=False,
                strict_cap=False,
            )

        scheduled = turn.start + config.speech_offset
        if monologue:
            start = compute_subtitle_aligned_start(
                scheduled,
                prev_end,
                speaker_gap_s=config.speaker_gap_s,
            )
        else:
            start = compute_turn_placed_start(
                scheduled,
                prev_end,
                speaker_gap_s=config.speaker_gap_s,
                max_inter_speaker_pause_s=config.max_inter_speaker_pause_s,
            )
        prev_end = start + len(fitted.samples) / sr
        placed.append(PlacedSegment(start=start, samples=fitted.samples, sample_rate=sr))
        placed_turns.append(turn)
        stats.append(
            {
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "cue_ids": list(turn.cue_ids),
                "target_duration_s": round(slot, 3),
                "raw_duration_s": round(len(samples) / sr, 3),
                "final_duration_s": round(len(fitted.samples) / sr, 3),
                "placed_start_s": round(start, 3),
                "trimmed": fitted.trimmed,
                "status": "reassembled",
            }
        )

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    assert sample_rate is not None
    return placed, _pipeline_sample_rate(placed, sample_rate), placed_turns


def print_speaker_map(speaker_map: dict[str, str], *, config: TtsConfig | None = None) -> None:
    if config and config.is_official_indextts:
        table = Table(title="Speaker → Reference audio")
        table.add_column("Speaker")
        table.add_column("Ref WAV")
        for speaker in sorted(speaker_map):
            try:
                ref = resolve_ref_audio_path(config, speaker)
                table.add_row(speaker, str(ref))
            except FileNotFoundError as exc:
                table.add_row(speaker, f"(missing: {exc})")
        console.print(table)
        return

    table = Table(title="Speaker → Voice")
    table.add_column("Speaker")
    table.add_column("Qwen3 Voice")
    for speaker, voice in sorted(speaker_map.items()):
        table.add_row(speaker, voice)
    console.print(table)


def save_voice_map(path: Path, speaker_map: dict[str, str], config: TtsConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "engine": config.engine_mode.value,
        "speakers": speaker_map,
    }
    if config.is_official_indextts:
        payload["ref_audio"] = {speaker: str(resolve_ref_audio_path(config, speaker)) for speaker in speaker_map}
        if config.indextts_supports_emotion:
            payload["emotion"] = config.indextts_emotion
            payload["emotion_weight"] = config.indextts_emotion_weight
        payload["indextts_version"] = config.indextts_resolved_version
    else:
        payload["model"] = config.model
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_turns_manifest(path: Path, turns: list[SpeakerTurn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "turn_id": t.turn_id,
            "speaker": t.speaker,
            "start": t.start,
            "slot_end": t.slot_end,
            "slot_duration": round(t.slot_duration, 3),
            "cue_ids": list(t.cue_ids),
            "text": t.text,
        }
        for t in turns
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
