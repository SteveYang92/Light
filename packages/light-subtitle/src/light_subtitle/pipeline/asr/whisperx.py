"""whisperX transcription with VAD + forced alignment.

Alternative to whisper.cpp.  Uses VAD pre-segmentation to place
segments in their correct time windows, then transcribes with
faster-whisper and aligns with wav2vec2.

The ASR pipeline and alignment model are cached at module level so
that repeated calls (e.g. across video segments) share the same
loaded models.  CTranslate2 thread count is set to ``os.cpu_count()``
to fully utilise multi-core CPUs.
"""

from __future__ import annotations

import os
import threading
import time

import whisperx
from light_models import Word

from ... import logger

# ── Module-level model cache ──────────────────────────

_cache: dict | None = None
_cache_lock = threading.Lock()


def _get_or_load_cache(model_name: str, language: str, cpu_threads: int) -> dict:
    """Return cached pipeline + alignment model for *model_name*/*language*.

    Thread-safe: only one thread loads while others wait, then all share.
    """
    global _cache
    key = (model_name, language, cpu_threads)

    with _cache_lock:
        if _cache is None or _cache.get("key") != key:
            device = "cpu"
            pipeline = whisperx.load_model(
                model_name,
                device,
                compute_type="int8",
                vad_method="silero",
                threads=cpu_threads,
            )
            align_model, align_meta = whisperx.load_align_model(
                language_code=language,
                device=device,
            )
            _cache = {
                "key": key,
                "pipeline": pipeline,
                "align_model": align_model,
                "align_meta": align_meta,
            }

    return _cache


def run(audio_path: str, language: str = "en", model_name: str = "turbo") -> list[Word]:
    """Transcribe *audio_path* with whisperX full pipeline.

    Returns the same ``list[Word]`` format as whisper.cpp, so
    downstream processing (segment, subtitle, export, QC) is
    unchanged.
    """
    t0 = time.time()

    cpu_threads = os.cpu_count() or 4
    cache = _get_or_load_cache(model_name, language, cpu_threads)
    pipeline = cache["pipeline"]
    align_model = cache["align_model"]
    align_meta = cache["align_meta"]

    # ── 1. Load audio ──
    audio = whisperx.load_audio(audio_path)

    # ── 2. Transcribe (VAD enabled by default) ──
    # batch_size > 1 merges VAD segments for faster encoder inference.
    result = pipeline.transcribe(audio, batch_size=8)
    logger.info(f"  ASR (whisperX): {len(result['segments'])} segments ({time.time() - t0:.0f}s)")

    # ── 3. Align ──
    device = "cpu"
    result = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        device,
        return_char_alignments=False,
    )

    # ── 4. Extract words ──
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
