from pathlib import Path

from ...config import SubtitleConfig
from ...utils.ffmpeg import extract_audio_16k, has_audio_stream
from .artifacts import audio_wav_path


def run(config: SubtitleConfig) -> str:
    """Extract 16kHz mono audio for ASR."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not has_audio_stream(config.input_path):
        raise RuntimeError(f"Input file has no audio track — cannot generate subtitles: {config.input_path}")

    asr_wav = str(audio_wav_path(config.output_dir))
    extract_audio_16k(config.input_path, asr_wav)

    return asr_wav
