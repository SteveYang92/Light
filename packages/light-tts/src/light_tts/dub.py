from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .assemble import assemble_timeline
from .audio_io import write_wav
from .config import DEFAULT_MODEL, AlignMode, EngineMode, TtsConfig
from .cues_loader import load_cues, resolve_dub_cues_path
from .indextts_runtime import maybe_reexec_in_official_venv, resolve_ref_audio_path
from .merge_turns import merge_speaker_turns
from .mix import OUTPUT_SUFFIX, find_video, mix_dub
from .subtitle_retime import export_dub_subtitles
from .synthesize import (
    print_speaker_map,
    reassemble_turns_from_segments,
    save_turns_manifest,
    save_voice_map,
    synthesize_cues,
    synthesize_turns,
)

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


def _apply_indextts_yaml(config: TtsConfig) -> None:
    """Merge bundled or run-local ``indextts.yaml`` (or legacy ``indextts2.yaml``) into *config*."""
    path = config.resolve_indextts_yaml_path()
    if not path:
        return
    merged = TtsConfig.from_yaml(path, output_dir=config.output_dir)
    cli_engine = config.engine_mode
    for key in (
        "indextts_official_root",
        "indextts_version",
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
        "indextts_use_fast",
        "indextts_max_text_tokens_per_segment",
        "indextts_segments_bucket_max_size",
        "indextts_chunk_chars",
        "indextts_chunk_min_chars",
        "indextts_metal_root",
        "indextts_metal_url",
        "indextts_metal_host",
        "indextts_metal_port",
        "indextts_metal_cfm_steps",
        "indextts_metal_manage_server",
        "indextts_normalize_rate",
        "max_turn_duration_s",
        "crossfade_ms",
        "atempo_max_monologue",
        "atempo_max",
        "speech_offset",
        "speaker_gap_s",
        "max_inter_cue_gap_s",
        "max_inter_speaker_pause_s",
        "subtitle_aligned",
        "align_mode",
    ):
        value = getattr(merged, key)
        if key in {"indextts_ref_audio", "indextts_checkpoints"} and not value:
            continue
        if key == "indextts_speaker_refs" and not value:
            continue
        setattr(config, key, value)
    if cli_engine in (EngineMode.INDEXTTS2, EngineMode.INDEXTTS15, EngineMode.INDEXTTS2_METAL):
        config.engine_mode = cli_engine
    else:
        config.engine_mode = merged.engine_mode
    if config.engine_mode == EngineMode.INDEXTTS15:
        config.indextts_version = "1.5"


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


def _resolve_total_duration(output_dir: Path, config: TtsConfig, cues: list) -> tuple[float, Path | None]:
    try:
        video = find_video(output_dir, config.video)
        return _probe_duration(video), video
    except FileNotFoundError:
        logger.warning("No video found — dub.wav length from last cue end")
        return max(c.end for c in cues) + 1.0, None


def _finalize_dub(
    config: TtsConfig,
    output_dir: Path,
    *,
    cues: list,
    placed: list,
    placed_turns: list | None,
    sample_rate: int,
    tts_dir: Path,
    dub_wav_path: Path,
    preview: bool = False,
) -> float:
    """Assemble timeline, write dub.wav, and export retimed subtitles when applicable."""
    if preview:
        total_duration = _preview_timeline_duration(placed, fallback_duration=max(c.end for c in cues) + 1.0)
    else:
        total_duration, _ = _resolve_total_duration(output_dir, config, cues)

    timeline = assemble_timeline(
        placed,
        total_duration,
        sample_rate,
        crossfade_ms=config.assembly_crossfade_ms,
        replace_on_overlap=config.assembly_replace_on_overlap,
    )
    write_wav(dub_wav_path, timeline, sample_rate)

    if config.effective_align_mode == AlignMode.TURN_RETIME and placed_turns and len(placed) == len(placed_turns):
        srt_path = export_dub_subtitles(
            output_dir,
            placed_turns,
            placed,
            cues,
            tts_dir=tts_dir,
            lang=config.lang,
        )
        console.print(f"  {srt_path.name} → {srt_path} (retimed to dub audio)")

    return total_duration


def run_reassemble(config: TtsConfig, *, cues_path: Path | None = None) -> Path:
    """Rebuild ``tts/dub.wav`` from existing segment WAVs."""
    if config.resolve_indextts_yaml_path():
        _apply_indextts_yaml(config)

    output_dir = Path(config.output_dir).resolve()
    cues_file = cues_path or resolve_dub_cues_path(output_dir, lang=config.lang)
    cues = load_cues(cues_file, lang=config.lang)
    turns = merge_speaker_turns(
        cues,
        speaker_gap_s=config.speaker_gap_s,
        max_turn_duration_s=config.max_turn_duration_s,
        max_turn_chars=config.chunk_chars(),
        min_turn_chars=config.chunk_min_chars(),
        max_inter_cue_gap_s=config.max_inter_cue_gap_s,
    )
    tts_dir = output_dir / "tts"
    segments_dir = tts_dir / "segments"
    dub_wav_path = tts_dir / "dub.wav"

    placed, sample_rate, placed_turns = reassemble_turns_from_segments(
        turns,
        cues,
        config,
        segments_dir=segments_dir,
        stats_path=tts_dir / "chunks.json",
    )
    total_duration = _finalize_dub(
        config,
        output_dir,
        cues=cues,
        placed=placed,
        placed_turns=placed_turns,
        sample_rate=sample_rate,
        tts_dir=tts_dir,
        dub_wav_path=dub_wav_path,
    )
    console.print(f"  dub.wav → {dub_wav_path} ({total_duration:.1f}s, reassembled)")
    return dub_wav_path


def run_mix_only(config: TtsConfig) -> Path:
    """Mux existing ``tts/dub.wav`` with the segment video (no TTS engine load)."""
    output_dir = Path(config.output_dir).resolve()
    dub_wav_path = output_dir / "tts" / "dub.wav"
    if not dub_wav_path.is_file():
        raise FileNotFoundError(f"Missing dubbed audio: {dub_wav_path}")
    video = find_video(output_dir, config.video)
    out_mp4 = output_dir / f"{video.stem}{OUTPUT_SUFFIX}.mp4"
    mix_dub(video, dub_wav_path, out_mp4, mode=config.mix_mode, duck_db=config.duck_db)
    console.print(f"  [green]✓[/green] {out_mp4}")
    return out_mp4


def run_dub(config: TtsConfig, *, cues_path: Path | None = None, skip_mix: bool = False) -> Path:
    """Full dub pipeline: synthesize → align → assemble → mix."""
    if config.mix_only:
        return run_mix_only(config)
    if config.reassemble:
        return run_reassemble(config)
    if config.is_indextts_dub:
        _apply_indextts_yaml(config)
        if config.is_official_indextts:
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

    if config.is_indextts_dub:
        labels = {cue.speaker.strip() or "__default__" for cue in cues}
        for label in sorted(labels):
            resolve_ref_audio_path(config, label)

    tts_dir = output_dir / "tts" / "preview" if config.preview else output_dir / "tts"
    segments_dir = tts_dir / "segments"
    dub_wav_path = tts_dir / "dub.wav"
    placed_turns: list | None = None

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
            max_inter_cue_gap_s=config.max_inter_cue_gap_s,
        )
        if config.max_cues is not None:
            turns = turns[: config.max_cues]
        save_turns_manifest(tts_dir / "turns.json", turns)
        console.print(
            f"[bold]Dubbing[/bold] {len(turns)} speaker turns (from {len(cues)} display cues) — {cues_file.name}"
        )
        speaker_map, placed, sample_rate, placed_turns = synthesize_turns(
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
        video = None
    else:
        _, video = _resolve_total_duration(output_dir, config, cues)

    total_duration = _finalize_dub(
        config,
        output_dir,
        cues=cues,
        placed=placed,
        placed_turns=placed_turns,
        sample_rate=sample_rate,
        tts_dir=tts_dir,
        dub_wav_path=dub_wav_path,
        preview=config.preview,
    )
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
            "indextts_version": config.indextts_resolved_version if config.is_indextts_dub else None,
            "indextts_emotion": config.indextts_emotion if config.indextts_supports_emotion else None,
            "indextts_ref_audio": config.indextts_ref_audio,
            "indextts_metal_url": config.indextts_metal_url if config.is_indextts_metal else None,
            "indextts_metal_cfm_steps": config.indextts_metal_cfm_steps if config.is_indextts_metal else None,
            "align_mode": config.effective_align_mode.value,
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
