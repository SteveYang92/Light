from __future__ import annotations

import hashlib

import numpy as np

from .base import SynthesisResult, TtsEngine


class MockEngine(TtsEngine):
    """Deterministic synthetic audio for unit tests (no mlx-audio required)."""

    sample_rate = 24000

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
        del language, instruct, max_tokens, top_k, top_p, repetition_penalty
        # ~0.12 s per character, min 0.5 s — rough stand-in for speech duration.
        duration = max(0.5, len(text) * 0.12)
        n = int(duration * self.sample_rate)
        digest_seed = int(hashlib.md5(f"{voice}:{text}:{seed}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(digest_seed)
        freq = 220.0 + (digest_seed % 200)
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        samples = (0.2 * np.sin(2 * np.pi * freq * t) * rng.uniform(0.8, 1.0, n)).astype(np.float32)
        return SynthesisResult(samples=samples, sample_rate=self.sample_rate)
