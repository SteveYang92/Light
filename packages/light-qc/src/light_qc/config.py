from dataclasses import dataclass, field


@dataclass
class QCConfig:
    bilingual: bool = False
    source_lang: str = "auto"
    target_lang: str | None = None

    llm_enabled: bool = False
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_temperature: float = 0.3

    max_lines: int = 2
    max_lines_zh: int = 1
    max_chars_per_line_zh: int = 30
    max_chars_per_line_en: int = 42
    cps_limit: int = 9
    cps_limit_en: int = 25
    min_duration: float = 0.8
    max_duration: float = 7.0
    min_gap: float = 0.1

    # Timeline-sync
    fps: float = 25.0
    entry_tolerance_frames: int = 3

    glossary: dict[str, str] = field(default_factory=dict)
    shot_changes: list[float] = field(default_factory=list)

    # ── Transcript alignment ──
    transcript_path: str | None = None
    alignment_tolerance: float = 0.12
    word_coverage_min: float = 0.95

    @property
    def max_chars_per_line(self) -> str:
        if self.target_lang == "zh" or self.source_lang == "zh":
            return self.max_chars_per_line_zh
        return self.max_chars_per_line_en
