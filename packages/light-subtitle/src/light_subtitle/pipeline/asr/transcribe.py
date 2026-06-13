from pathlib import Path

from light_models import Word

from ...config import SubtitleConfig
from ...utils.whisper import run_whisper


def run(config: SubtitleConfig, audio_path: str) -> list[Word]:
    output_dir = str(Path(config.output_dir) / "asr")
    return run_whisper(
        audio_path=audio_path,
        model_path=config.whisper_model,
        whisper_path=config.whisper_path,
        output_dir=output_dir,
        language=config.language,
    )
