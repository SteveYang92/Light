from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynthesisResult:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.samples.shape[0]) / float(self.sample_rate)


class TtsEngine:
    """Protocol for TTS backends."""

    sample_rate: int = 24000

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
        raise NotImplementedError
