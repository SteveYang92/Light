from __future__ import annotations

import logging
from io import BytesIO
from wave import open as wave_open

import numpy as np

from ..audio_io import compress_internal_silence, trim_edge_silence
from ..config import TtsConfig
from ..indextts_metal_runtime import (
    create_metal_client,
    ensure_metal_server,
    load_voice_cache,
    metal_voice_cache_path,
    resolve_metal_paths,
    resolve_metal_voice_id,
    synthesize_wav,
)
from ..indextts_runtime import INDEXTTS2_SAMPLE_RATE
from .base import SynthesisResult, TtsEngine

logger = logging.getLogger(__name__)


def _read_wav_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    with wave_open(BytesIO(payload), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


class IndexTTS2MetalEngine(TtsEngine):
    """IndexTTS2 via native mtts HTTP server (Apple Silicon Metal runtime)."""

    sample_rate = INDEXTTS2_SAMPLE_RATE

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        paths = resolve_metal_paths(config)
        if not paths.bin.is_file():
            raise FileNotFoundError(f"mtts binary not found: {paths.bin}\n  Run ./scripts/setup_indextts2_metal.sh")
        if not paths.model_bundle.is_dir():
            raise FileNotFoundError(
                f"MIT2 model bundle not found: {paths.model_bundle}\n  Run ./scripts/setup_indextts2_metal.sh"
            )
        ensure_metal_server(config, paths)
        self._client = create_metal_client(config)
        cache_path = metal_voice_cache_path(config)
        self._voice_cache = load_voice_cache(cache_path)
        logger.info(
            "IndexTTS2 Metal ready at %s (cfm_steps=%s)",
            config.indextts_metal_url,
            config.indextts_metal_cfm_steps,
        )

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        language: str = "Chinese",
        speed: float = 1.0,
        instruct: str | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> SynthesisResult:
        del language, speed, instruct, max_tokens, seed, top_k, top_p, repetition_penalty
        voice_id = resolve_metal_voice_id(self._config, self._client, voice, cache=self._voice_cache)
        wav_bytes = synthesize_wav(self._client, voice_id=voice_id, text=text)
        samples, sample_rate = _read_wav_bytes(wav_bytes)
        trimmed = trim_edge_silence(samples, sample_rate)
        compressed = compress_internal_silence(trimmed, sample_rate)
        return SynthesisResult(samples=compressed, sample_rate=sample_rate)
