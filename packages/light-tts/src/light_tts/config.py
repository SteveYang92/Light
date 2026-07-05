from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

try:
    from enum import StrEnum
except ImportError:

    class StrEnum(str, Enum):  # noqa: UP042 — Python 3.10 compat (official IndexTTS2 venv)
        pass


from pathlib import Path

import yaml

IndexTTSVersion = Literal["1.5", "2.0"]

# CustomVoice supports preset speakers (Vivian, Uncle_Fu, …). Base is for voice cloning only.
# 1.7B is slower than 0.6B, but is the better default for quality/stability previews.
DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"


class MixMode(StrEnum):
    REPLACE = "replace"
    DUCK = "duck"
    DUAL = "dual"


class EngineMode(StrEnum):
    MLX = "mlx"
    HTTP = "http"
    MOCK = "mock"
    INDEXTTS2 = "indextts2"
    INDEXTTS15 = "indextts15"
    INDEXTTS2_METAL = "indextts2_metal"


class AlignMode(StrEnum):
    """How synthesized turns are placed on the timeline."""

    TURN_COMPACT = "turn_compact"
    TURN_RETIME = "turn_retime"
    SUBTITLE_ALIGNED = "subtitle_aligned"


@dataclass
class VoiceConfig:
    language: str = "Chinese"
    instruct: str = ""


@dataclass
class TtsConfig:
    output_dir: str
    lang: str = "zh"
    model: str = DEFAULT_MODEL
    voices_path: str | None = None
    default_voice: str = "Vivian"
    auto_assign: bool = True
    speakers: dict[str, str] = field(default_factory=dict)
    voices: dict[str, VoiceConfig] = field(default_factory=dict)
    mix_mode: MixMode = MixMode.DUCK
    engine_mode: EngineMode = EngineMode.MLX
    mlx_server_url: str = field(default_factory=lambda: os.environ.get("MLX_AUDIO_URL", "http://127.0.0.1:8000"))
    max_cues: int | None = None
    resume: bool = False
    mix_only: bool = False
    reassemble: bool = False
    align_mode: AlignMode = AlignMode.TURN_COMPACT
    subtitle_aligned: bool = False
    speech_offset: float = 0.05
    crossfade_ms: float = 50.0
    atempo_min: float = 0.88
    atempo_max: float = 1.28
    atempo_max_cross: float = 1.42
    speaker_gap_s: float = 0.08
    max_turn_duration_s: float = 120.0
    max_inter_cue_gap_s: float = 2.0
    max_inter_speaker_pause_s: float = 0.75
    allow_trim: bool = False
    per_cue: bool = False
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    tts_speed_max: float = 2.0
    atempo_max_monologue: float = 2.2
    tts_outlier_ratio: float = 2.5
    qwen_chunk_chars: int = 180
    qwen_chunk_min_chars: int = 40
    qwen_max_tokens_headroom: float = 1.5
    qwen_max_tokens_min: int = 512
    qwen_max_tokens_max: int = 2048
    qwen_seed: int | None = None
    preview: bool = False
    preview_duration_s: float = 180.0
    duck_db: float = -18.0
    video: str | None = None
    indextts_official_root: str = "vendor/index-tts"
    indextts_version: IndexTTSVersion = "2.0"
    indextts_checkpoints: str | None = None
    indextts_ref_audio: str | None = None
    indextts_speaker_refs: dict[str, str] = field(default_factory=dict)
    indextts_emotion: str = "calm"
    indextts_emotion_weight: float = 0.6
    indextts_num_beams: int = 3
    indextts_use_fp16: bool = False
    indextts_use_random: bool = False
    indextts_verbose: bool = False
    indextts_torch_compile: bool = False
    indextts_chunk_chars: int = 160
    indextts_chunk_min_chars: int = 45
    # IndexTTS 1.5 only: official infer_fast() token batching (not the same as chunk_chars).
    indextts_use_fast: bool = True
    indextts_max_text_tokens_per_segment: int = 100
    indextts_segments_bucket_max_size: int = 4
    indextts_metal_root: str = "vendor/index-tts2-metal"
    indextts_metal_url: str = field(default_factory=lambda: os.environ.get("MIT2_SERVER_URL", "http://127.0.0.1:3456"))
    indextts_metal_host: str = "127.0.0.1"
    indextts_metal_port: int = 3456
    indextts_metal_cfm_steps: int = 16
    indextts_metal_manage_server: bool = False
    indextts_normalize_rate: bool = True

    @property
    def is_official_indextts(self) -> bool:
        return self.engine_mode in (EngineMode.INDEXTTS2, EngineMode.INDEXTTS15)

    @property
    def is_indextts_dub(self) -> bool:
        return self.engine_mode in (EngineMode.INDEXTTS2, EngineMode.INDEXTTS15, EngineMode.INDEXTTS2_METAL)

    @property
    def is_indextts_metal(self) -> bool:
        return self.engine_mode == EngineMode.INDEXTTS2_METAL

    @property
    def indextts_resolved_version(self) -> IndexTTSVersion:
        if self.engine_mode == EngineMode.INDEXTTS15:
            return "1.5"
        return self.indextts_version

    @property
    def indextts_supports_emotion(self) -> bool:
        return self.is_official_indextts and self.indextts_resolved_version == "2.0"

    @property
    def effective_align_mode(self) -> AlignMode:
        if self.subtitle_aligned and self.align_mode == AlignMode.TURN_COMPACT:
            return AlignMode.SUBTITLE_ALIGNED
        return self.align_mode

    @property
    def assembly_crossfade_ms(self) -> float:
        if self.effective_align_mode in (AlignMode.TURN_RETIME, AlignMode.SUBTITLE_ALIGNED):
            return 0.0
        return self.crossfade_ms

    @property
    def assembly_replace_on_overlap(self) -> bool:
        return self.effective_align_mode in (AlignMode.TURN_RETIME, AlignMode.SUBTITLE_ALIGNED)

    def chunk_chars(self) -> int:
        if self.is_indextts_dub:
            return self.indextts_chunk_chars
        return self.qwen_chunk_chars

    def chunk_min_chars(self) -> int:
        if self.is_indextts_dub:
            return self.indextts_chunk_min_chars
        return self.qwen_chunk_min_chars

    @classmethod
    def from_yaml(cls, path: str | Path, *, output_dir: str, **overrides: object) -> TtsConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        engine_raw = str(data.get("engine", "")).lower()
        if engine_raw == "indextts15":
            engine_mode = EngineMode.INDEXTTS15
        elif engine_raw == "indextts2_metal":
            engine_mode = EngineMode.INDEXTTS2_METAL
        elif engine_raw in ("indextts2", "indextts"):
            engine_mode = EngineMode.INDEXTTS2
        else:
            engine_mode = EngineMode.MLX
        indextts_version: IndexTTSVersion = "2.0"
        if engine_mode == EngineMode.INDEXTTS15:
            indextts_version = "1.5"
        version_raw = data.get("indextts_version")
        if version_raw is not None:
            version_str = str(version_raw)
            if version_str not in ("1.5", "2.0"):
                raise ValueError(f"Unsupported indextts_version: {version_str!r} (expected '1.5' or '2.0')")
            indextts_version = version_str  # type: ignore[assignment]
        speakers = {str(k): str(v) for k, v in (data.get("speakers") or {}).items()}
        speaker_refs = {str(k): str(v) for k, v in (data.get("speaker_refs") or {}).items()}
        voices_raw = data.get("voices") or {}
        voices = {
            str(name): VoiceConfig(
                language=str(v.get("language", "Chinese")),
                instruct=str(v.get("instruct", "")),
            )
            for name, v in voices_raw.items()
        }
        cfg = cls(
            output_dir=output_dir,
            engine_mode=engine_mode,
            indextts_version=indextts_version,
            model=str(data.get("model", DEFAULT_MODEL)),
            default_voice=str(data.get("default_voice", "Vivian")),
            auto_assign=bool(data.get("auto_assign", True)),
            speakers=speakers,
            voices=voices,
            indextts_speaker_refs=speaker_refs,
            temperature=float(data.get("temperature", 0.6)),
            qwen_seed=int(data["qwen_seed"]) if data.get("qwen_seed") is not None else None,
        )
        skip_keys = {
            "engine",
            "indextts_version",
            "model",
            "default_voice",
            "auto_assign",
            "speakers",
            "speaker_refs",
            "voices",
            "temperature",
            "qwen_seed",
        }
        for key, value in data.items():
            if key in skip_keys:
                continue
            if key == "official_root":
                cfg.indextts_official_root = str(value)
                continue
            if key == "metal_root":
                cfg.indextts_metal_root = str(value)
                continue
            if key == "metal_url":
                cfg.indextts_metal_url = str(value)
                continue
            if key == "metal_cfm_steps":
                cfg.indextts_metal_cfm_steps = int(value)
                continue
            if key == "metal_manage_server":
                cfg.indextts_metal_manage_server = bool(value)
                continue
            if key == "checkpoints":
                cfg.indextts_checkpoints = str(value) if value else None
                continue
            if key == "ref_audio":
                cfg.indextts_ref_audio = str(value) if value else None
                continue
            if key == "emotion":
                cfg.indextts_emotion = str(value)
                continue
            if key == "emotion_weight":
                cfg.indextts_emotion_weight = float(value)
                continue
            if key == "num_beams":
                cfg.indextts_num_beams = int(value)
                continue
            if key == "align_mode":
                cfg.align_mode = AlignMode(str(value))
                continue
            if key == "subtitle_aligned" and value:
                cfg.subtitle_aligned = bool(value)
                continue
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
        for key, value in overrides.items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
        return cfg

    def resolve_voices_path(self) -> Path | None:
        if self.voices_path:
            p = Path(self.voices_path)
            if p.is_file():
                return p
        for candidate in (
            Path(self.output_dir) / "voices.yaml",
            Path(self.output_dir).parent / "voices.yaml",
        ):
            if candidate.is_file():
                return candidate
        bundled = Path(__file__).parent / "assets" / "voices.yaml"
        return bundled if bundled.is_file() else None

    def resolve_indextts_yaml_path(self) -> Path | None:
        for candidate in (
            Path(self.output_dir) / "indextts.yaml",
            Path(self.output_dir).parent / "indextts.yaml",
            Path(self.output_dir) / "indextts2.yaml",
            Path(self.output_dir).parent / "indextts2.yaml",
        ):
            if candidate.is_file():
                return candidate
        bundled = Path(__file__).parent / "assets" / "indextts.yaml"
        if bundled.is_file():
            return bundled
        legacy = Path(__file__).parent / "assets" / "indextts2.yaml"
        return legacy if legacy.is_file() else None

    def resolve_indextts2_path(self) -> Path | None:
        """Backward-compatible alias for :meth:`resolve_indextts_yaml_path`."""
        return self.resolve_indextts_yaml_path()
