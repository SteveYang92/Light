"""whisperX transcription with VAD + forced alignment.

Alternative to whisper.cpp.  Uses VAD pre-segmentation to place
segments in their correct time windows, then transcribes with
faster-whisper and aligns with wav2vec2.
"""

from __future__ import annotations

import time

import whisperx
from light_models import Word

from ... import logger


def run(audio_path: str, language: str = "en", model_name: str = "turbo") -> list[Word]:
    """Transcribe *audio_path* with whisperX full pipeline.

    Returns the same ``list[Word]`` format as whisper.cpp, so
    downstream processing (segment, subtitle, export, QC) is
    unchanged.
    """
    t0 = time.time()

    # ── 1. Load model ──
    device = "cpu"
    model = whisperx.load_model(model_name, device, compute_type="int8", vad_method="silero")

    # ── 2. Load audio ──
    audio = whisperx.load_audio(audio_path)

    # ── 3. Transcribe (VAD enabled by default) ──
    # batch_size > 1 merges VAD segments for faster encoder inference.
    result = model.transcribe(audio, batch_size=8)
    logger.info(f"  ASR (whisperX): {len(result['segments'])} segments ({time.time() - t0:.0f}s)")

    # ── 4. Align ──
    align_model, align_meta = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        device,
        return_char_alignments=False,
    )

    # ── 5. Extract words ──
    words: list[Word] = []
    for seg in result.get("segments", []):
        for aw in seg.get("words", []):
            words.append(
                Word(
                    text=" " + aw.get("word", ""),
                    start=aw.get("start", 0.0),
                    end=aw.get("end", 0.0),
                    confidence=aw.get("score", 0.0),
                )
            )

    logger.info(f"  ASR (whisperX): {len(words)} words in {time.time() - t0:.0f}s total")
    return words
