import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml


class AsrEngine(StrEnum):
    """ASR engine type."""

    WHISPERX = "whisperx"
    WHISPER_CPP = "whisper-cpp"


@dataclass
class SubtitleConfig:
    input_path: str
    output_dir: str = "./output"
    url: str | None = None  # URL input (mutually exclusive with input_path)
    slug: str | None = None  # Semantic name derived from title or filename
    bilingual: bool = False

    whisper_model: str = "ggml-large-v3-turbo.bin"
    whisper_path: str = "whisper-cli"
    language: str = "auto"

    target_lang: str | None = None

    cps_limit: int = 9
    cps_limit_en: int = 25
    max_lines: int = 2
    max_lines_zh: int = 1
    max_chars_per_line_zh: int = 40
    max_chars_per_line_en: int = 42
    min_duration: float = 0.8
    max_duration: float = 7.0
    reading_padding: float = 0.3

    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    llm_temperature: float = 0.4

    asr: AsrEngine = AsrEngine.WHISPERX
    resume: bool = False
    resume_from: str | None = None
    diarize: bool = False
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    # ── Translation quality evaluation (opt-in) ──
    evaluate_enabled: bool = False  # Run LLM quality evaluation after translation (adds ~2x cost)
    quality_threshold: float = 0.7  # Overall score below this triggers refinement
    max_refine_rounds: int = 1  # Max rounds of refine (per segment)

    annotate: bool = False  # Generate secondary subtitle annotations
    annotation_width: int = 30  # Annotation box width (% of screen, 1–100)
    optimize_entry_points: bool = False  # Auto-fix low-confidence entry points in pace
    transcript_words: list | None = None  # Runtime: word list for entry optimization

    # ── Transcript correction + translation context ──
    correct_enabled: bool = True  # LLM-based ASR error correction after align
    context_prep_enabled: bool = True  # Extract glossary + summary before translation
    content_summary: dict | None = None  # Injected into translation prompts

    glossary: dict[str, str] = field(default_factory=dict)
    speaker_names: dict[str, str] = field(default_factory=dict)
    shot_changes: list[float] = field(default_factory=list)

    @property
    def max_chars_per_line(self) -> int:
        if self.target_lang == "zh":
            return self.max_chars_per_line_zh
        return self.max_chars_per_line_en

    @classmethod
    def from_yaml(cls, path: str) -> "SubtitleConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def clone_for_segment(self, *, input_path: str, output_dir: str) -> "SubtitleConfig":
        """Create a per-segment copy with overridden input_path and output_dir.

        Preserves all pipeline parameters (ASR, LLM, formatting, etc.) so each
        segment in a long-video split receives the same configuration.
        """
        import copy

        cloned = copy.copy(self)
        cloned.input_path = input_path
        cloned.output_dir = output_dir
        # Per-segment state must not leak across segments.
        cloned.glossary = dict(self.glossary)
        cloned.speaker_names = dict(self.speaker_names)
        cloned.shot_changes = list(self.shot_changes)
        cloned.transcript_words = None
        cloned.content_summary = None
        # Auto-resume: if a previous run left pipeline_run.json, pick up where it left off.
        cloned.resume = (Path(output_dir) / "pipeline_run.json").exists()
        return cloned

    @classmethod
    def from_cli(cls, **kwargs) -> "SubtitleConfig":
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return cls(**filtered)
