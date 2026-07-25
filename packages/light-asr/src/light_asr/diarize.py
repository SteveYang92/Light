"""Speaker diarization — assign speaker labels to ASR words via time overlap.

Uses whisperX's ``DiarizationPipeline`` (wraps pyannote.audio) to detect
"who speaks when", then assigns a speaker label to each word based on
where the word's temporal midpoint falls.

Usage::

    from light_asr import diarize
    words = diarize.run(words, audio_path, hf_token="<your-hf-token>")
"""

from __future__ import annotations

import os

from light_core import logger
from light_models import Word


def _detect_device() -> str:
    """Auto-detect the best available torch device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(
    words: list[Word],
    audio_path: str,
    hf_token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-community-1",
    device: str | None = None,
) -> list[Word]:
    """Assign speaker labels to *words* based on diarization of *audio_path*.

    Args:
        words:      ASR word list from previous pipeline phase.
        audio_path: Audio file (16 kHz mono WAV).
        hf_token:   HuggingFace access token for pyannote models.
                    If ``None``, reads ``HF_TOKEN`` env var.
        model_name: Pyannote diarization model name.
        device:     Torch device (``"cpu"``, ``"mps"``, ``"cuda"``).
                    Auto-detected when ``None``.

    Returns:
        The same word list with ``speaker`` populated where assigned.
        Words whose midpoint falls in no speaker segment retain their
        original speaker value (usually ``None``).
    """
    if device is None:
        device = _detect_device()
    logger.info(f"Using device: {device}")
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Speaker diarization requires a HuggingFace token.\n"
            "  Set HF_TOKEN env var or pass --hf-token.\n"
            "  Sign up at https://hf.co/pyannote/speaker-diarization-3.1"
        )

    import whisperx.diarize

    logger.info("Loading diarization pipeline...")
    with logger.capture_external_output():
        diarize_model = whisperx.diarize.DiarizationPipeline(model_name=model_name, token=token, device=device)

    logger.info("Running diarization on audio...")
    with logger.capture_external_output():
        diarize_result = diarize_model(audio_path, min_speakers=0, max_speakers=10)

    # Newer whisperx returns a pandas DataFrame with columns: start, end, speaker
    import pandas as pd

    if isinstance(diarize_result, tuple):
        df: pd.DataFrame = diarize_result[0]
    else:
        df = diarize_result

    if df is None or df.empty:
        logger.warning("Diarization produced no speaker segments — leaving speaker fields empty.")
        return words

    speaker_segments: list[tuple[float, float, str]] = [
        (row["start"], row["end"], row["speaker"]) for _, row in df.iterrows()
    ]
    logger.info(f"Diarization found {len(speaker_segments)} speaker turns.")

    for w in words:
        midpoint = (w.start + w.end) / 2.0
        best_label: str | None = None
        best_overlap = 0.0

        for seg_start, seg_end, label in speaker_segments:
            if seg_start <= midpoint <= seg_end:
                # Midpoint fully inside — unambiguous assignment.
                w.speaker = label
                best_label = None  # signal that we already assigned
                break
            # Partial overlap — compute overlap length between word span and speaker span.
            overlap_start = max(w.start, seg_start)
            overlap_end = min(w.end, seg_end)
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label

        if best_label is not None:
            w.speaker = best_label

    assigned = sum(1 for w in words if w.speaker)
    logger.info(f"Assigned speaker labels to {assigned}/{len(words)} words.")
    return words
