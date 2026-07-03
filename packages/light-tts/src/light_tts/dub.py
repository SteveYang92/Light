from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .assemble import assemble_timeline
from .audio_io import write_wav
from .config import DEFAULT_MODEL, EngineMode, TtsConfig
from .cues_loader import load_cues, resolve_dub_cues_path
from .indextts2_runtime import maybe_reexec_in_official_venv, resolve_ref_audio_path
from .merge_turns import merge_speaker_turns
from .mix import OUTPUT_SUFFIX, find_video, mix_dub
from .synthesize import print_speaker_map, save_turns_manifest, save_voice_map, synthesize_cues, synthesize_turns

logger = logging.getLogger(__name__)
console = Console()


def _probe_duration(path: Path) -> float:
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def _apply_indextts2_yaml(config: TtsConfig) -> None:
    """Merge bundled or run-local ``indextts2.yaml`` into *config*."""
    path = config.resolve_indextts2_path()
    if not path:
        return
    merged = TtsConfig.from_yaml(path, output_dir=config.output_dir)
    merged.engine_mode = EngineMode.INDEXTTS2
    for key in (
        "indextts_official_root",
        "indextts_checkpoints",
        "indextts_ref_audio",
        "indextts_speaker_refs",
        "indextts_emotion",
        "indextts_emotion_weight",
        "indextts_num_beams",
        "indextts_use_fp16",
        "indextts_use_random",
        "indextts_verbose",
        "indextts_torch_compile",
        "indextts_chunk_chars",
        "indextts_chunk_min_chars",
        "max_turn_duration_s",
        "crossfade_ms",
        "atempo_max_monologue",
        "atempo_max",
        "resume",
        "speech_offset",
        "speaker_gap_s",
        "max_inter_speaker_pause_s",
    ):
        value = getattr(merged, key)
        if key in {"indextts_ref_audio", "indextts_checkpoints"} and not value:
            continue
        if key == "indextts_speaker_refs" and not value:
            continue
        setattr(config, key, value)


def _apply_voices_yaml(config: TtsConfig) -> None:
    """Merge bundled or run-local ``voices.yaml`` into *config*."""
    voices_yaml = config.resolve_voices_path()
    if not voices_yaml:
        return
    merged = TtsConfig.from_yaml(voices_yaml, output_dir=config.output_dir)
    if not config.speakers:
        config.speakers = merged.speakers
    if not config.voices:
        config.voices = merged.voices
    config.default_voice = merged.default_voice
    config.auto_assign = merged.auto_assign
    config.temperature = merged.temperature
    for key in (
        "top_k",
        "top_p",
        "repetition_penalty",
        "tts_speed_max",
        "atempo_max_monologue",
        "tts_outlier_ratio",
        "qwen_chunk_chars",
        "qwen_chunk_min_chars",
        "qwen_max_tokens_headroom",
        "qwen_max_tokens_min",
        "qwen_max_tokens_max",
    ):
        setattr(config, key, getattr(merged, key))
    if merged.qwen_seed is not None:
        config.qwen_seed = merged.qwen_seed
    if config.model == DEFAULT_MODEL:
        config.model = merged.model


def _limit_preview_cues(cues: list, duration_s: float) -> list:
    """Keep a short prefix for audible preview generation."""
    if not cues:
        return cues
    start = cues[0].start
    end = start + max(0.1, duration_s)
    limited = [c for c in cues if c.start < end]
    return limited or cues[:1]


def _preview_timeline_duration(placed: list, *, fallback_duration: float) -> float:
    """Preview WAVs should end after generated speech, not after sparse source cues."""
    if not placed:
        return fallback_duration
    last_end = max(seg.start + (len(seg.samples) / seg.sample_rate) for seg in placed)
    return max(0.1, last_end + 1.0)


def run_dub(config: TtsConfig, *, cues_path: Path | None = None, skip_mix: bool = False) -> Path:
    """Full dub pipeline: synthesize → align → assemble → mix."""
    if config.engine_mode == EngineMode.INDEXTTS2:
        _apply_indextts2_yaml(config)
        maybe_reexec_in_official_venv(
            official_root=Path(config.indextts_official_root),
            enabled=True,
        )
    else:
        _apply_voices_yaml(config)

    output_dir = Path(config.output_dir).resolve()
    cues_file = cues_path or resolve_dub_cues_path(output_dir, lang=config.lang)
    cues = load_cues(cues_file, lang=config.lang, max_cues=config.max_cues if config.per_cue else None)

    if config.preview:
        cues = _limit_preview_cues(cues, config.preview_duration_s)

    if config.engine_mode == EngineMode.INDEXTTS2:
        labels = {cue.speaker.strip() or "__default__" for cue in cues}
        for label in sorted(labels):
            resolve_ref_audio_path(config, label)

    tts_dir = output_dir / "tts" / "preview" if config.preview else output_dir / "tts"
    segments_dir = tts_dir / "segments"
    dub_wav_path = tts_dir / "dub.wav"

    if config.per_cue:
        if config.max_cues is not None:
            cues = cues[: config.max_cues]
        console.print(f"[bold]Dubbing[/bold] {len(cues)} display cues from {cues_file.name} (--per-cue)")
        speaker_map, placed, sample_rate = synthesize_cues(cues, config, segments_dir=segments_dir)
        unit_count = len(cues)
        mode = "per_cue"
    else:
        turns = merge_speaker_turns(
            cues,
            speaker_gap_s=config.speaker_gap_s,
            max_turn_duration_s=config.max_turn_duration_s,
            max_turn_chars=config.chunk_chars(),
            min_turn_chars=config.chunk_min_chars(),
        )
        if config.max_cues is not None:
            turns = turns[: config.max_cues]
        save_turns_manifest(tts_dir / "turns.json", turns)
        console.print(
            f"[bold]Dubbing[/bold] {len(turns)} speaker turns (from {len(cues)} display cues) — {cues_file.name}"
        )
        speaker_map, placed, sample_rate = synthesize_turns(
            turns,
            cues,
            config,
            segments_dir=segments_dir,
            stats_path=tts_dir / "chunks.json",
        )
        unit_count = len(turns)
        mode = "speaker_turns"

    print_speaker_map(speaker_map, config=config)
    save_voice_map(tts_dir / "voice_map.json", speaker_map, config)

    if config.preview:
        total_duration = _preview_timeline_duration(placed, fallback_duration=max(c.end for c in cues) + 1.0)
        video = None
    else:
        try:
            video = find_video(output_dir, config.video)
            total_duration = _probe_duration(video)
        except FileNotFoundError:
            total_duration = max(c.end for c in cues) + 1.0
            video = None
            logger.warning("No video found — dub.wav length from last cue end")

    timeline = assemble_timeline(placed, total_duration, sample_rate, crossfade_ms=config.crossfade_ms)
    write_wav(dub_wav_path, timeline, sample_rate)
    console.print(f"  dub.wav → {dub_wav_path} ({total_duration:.1f}s)")

    run_state = {
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),  # noqa: UP017
        "cues_path": str(cues_file),
        "cue_count": len(cues),
        "unit_count": unit_count,
        "mode": mode,
        "config": {
            "lang": config.lang,
            "model": config.model,
            "engine": config.engine_mode.value,
            "mix_mode": config.mix_mode.value,
            "preview": config.preview,
            "preview_duration_s": config.preview_duration_s if config.preview else None,
            "chunk_chars": config.chunk_chars(),
            "chunk_min_chars": config.chunk_min_chars(),
            "qwen_chunk_chars": config.qwen_chunk_chars,
            "qwen_max_tokens_min": config.qwen_max_tokens_min,
            "qwen_max_tokens_max": config.qwen_max_tokens_max,
            "indextts_emotion": config.indextts_emotion if config.engine_mode == EngineMode.INDEXTTS2 else None,
            "indextts_ref_audio": config.indextts_ref_audio,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "qwen_seed": config.qwen_seed,
        },
    }
    (tts_dir / "tts_run.json").write_text(json.dumps(run_state, indent=2), encoding="utf-8")

    if config.preview or skip_mix or video is None:
        return dub_wav_path

    out_mp4 = output_dir / f"{video.stem}{OUTPUT_SUFFIX}.mp4"
    mix_dub(video, dub_wav_path, out_mp4, mode=config.mix_mode, duck_db=config.duck_db)
    console.print(f"  [green]✓[/green] {out_mp4}")
    return out_mp4


def run_poc(
    cues_path: Path,
    out_dir: Path,
    *,
    model: str,
    max_cues: int = 3,
    engine_mode: EngineMode = EngineMode.MLX,
) -> None:
    """Phase 0 POC: synthesize up to *max_cues* zh cues to WAV files only."""
    config = TtsConfig(
        output_dir=str(out_dir.parent),
        lang="zh",
        model=model,
        max_cues=max_cues,
        engine_mode=engine_mode,
    )
    _apply_voices_yaml(config)
    cues = load_cues(cues_path, lang="zh", max_cues=max_cues)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = out_dir
    config.resume = False

    speaker_map, _, sample_rate = synthesize_cues(cues, config, segments_dir=segments_dir)
    print_speaker_map(speaker_map, config=config)
    console.print(f"[green]Wrote {len(cues)} WAV files to {out_dir}[/green] (sr={sample_rate})")
