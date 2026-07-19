"""ASR pipeline — extract_audio → transcribe → align → diarize."""

import warnings

# Suppress pyannote's harmless torchcodec warning on Apple Silicon.
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
