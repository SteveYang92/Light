"""light-asr — standalone ASR capability package.

Public API: config (:class:`AsrConfig`, :class:`AsrEngine`), audio
extraction, engine providers (whisperX / whisper.cpp), forced alignment,
diarization, checkpoint serialization, and the one-call
:func:`transcribe` entry point.
"""

import warnings

from .align import align_words
from .api import transcribe
from .audio import extract_audio, extract_audio_16k, has_audio_stream
from .checkpoints import load_words_checkpoint, save_whisper_cpp_raw, save_words_checkpoint
from .config import AsrConfig, AsrEngine
from .whisper_cpp import find_model, find_whisper

# Suppress pyannote's harmless torchcodec warning on Apple Silicon.
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

__all__ = [
    "AsrConfig",
    "AsrEngine",
    "align_words",
    "extract_audio",
    "extract_audio_16k",
    "find_model",
    "find_whisper",
    "has_audio_stream",
    "load_words_checkpoint",
    "save_whisper_cpp_raw",
    "save_words_checkpoint",
    "transcribe",
]
