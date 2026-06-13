from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    name: str
    description: str
    input_audio: Path
    input_asr: Path | None
    subtitle_config: dict
    qc_config: dict
    thresholds: dict

    @classmethod
    def from_yaml(cls, path: Path) -> "TestCase":
        import yaml

        data = yaml.safe_load(path.read_text())
        case_dir = path.parent
        input_data = data.get("input", {})
        audio_path = (case_dir / input_data["audio"]).resolve()
        asr_path = None
        if input_data.get("asr_result"):
            asr_path = (case_dir / input_data["asr_result"]).resolve()
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_audio=audio_path,
            input_asr=asr_path,
            subtitle_config=data.get("light_subtitle", {}),
            qc_config=data.get("qc", {}),
            thresholds=data.get("thresholds", {}),
        )


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    case_name: str
    report: dict
    duration_sec: float
    git_commit: str | None


@dataclass
class DiffReport:
    baseline_run_id: str
    current_run_id: str
    errors_delta: int
    warnings_delta: int
    suggestions_delta: int
    rule_changes: list[dict]
    new_issues: list[dict]
    fixed_issues: list[dict]
    degraded: bool
    reasons: list[str]
