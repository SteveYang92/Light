"""One-call ASR entry point for standalone integrations.

Orchestrates engine dispatch → (whisper.cpp only) forced alignment →
optional diarization.  No notion of run directories: raw outputs land in
the caller-provided *work_dir*.
"""

from __future__ import annotations

from pathlib import Path

from light_core.progress import ProgressCallback
from light_models import Word

from . import checkpoints
from .config import AsrConfig, AsrEngine


def transcribe(
    audio_path: str,
    config: AsrConfig,
    work_dir: str | Path,
    progress: ProgressCallback | None = None,
) -> list[Word]:
    """Transcribe *audio_path* into a word list according to *config*.

    whisper.cpp runs additionally get wav2vec2 forced alignment; when
    ``config.diarize`` is set, speaker labels are assigned last.

    *progress* is reserved for future incremental progress reporting;
    the current providers do not emit callbacks.
    """
    from . import align as _align
    from . import diarize, whisper_cpp, whisperx

    del progress  # reserved — no incremental progress yet
    work = Path(work_dir)
    language = config.language if config.language != "auto" else "en"

    if config.engine == AsrEngine.WHISPERX:
        words = whisperx.run(audio_path, language=language)
    else:
        words = whisper_cpp.transcribe(audio_path, config, work)
        raw_src = work / "whisper_output.json"
        if raw_src.exists():
            raw_name = f"asr_{AsrEngine.WHISPER_CPP.value}{checkpoints.WHISPER_CPP_RAW_SUFFIX}"
            checkpoints.save_whisper_cpp_raw(raw_src, work / raw_name)
        words = _align.align_words(words, audio_path, language=language)

    if config.diarize:
        words = diarize.run(words, audio_path, hf_token=config.hf_token, model_name=config.diarize_model)

    return words
