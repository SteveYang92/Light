from __future__ import annotations

import logging

import httpx
import numpy as np

from ..audio_io import read_wav_bytes
from .base import SynthesisResult, TtsEngine

logger = logging.getLogger(__name__)

_LANG_CODES = {
    "chinese": "chinese",
    "english": "english",
    "auto": "auto",
}


def _lang_code(language: str) -> str:
    return _LANG_CODES.get(language.lower(), language.lower())


class Qwen3HttpEngine(TtsEngine):
    """Call mlx-audio OpenAI-compatible ``/v1/audio/speech`` server."""

    sample_rate = 24000

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

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
        del language
        payload: dict[str, object] = {
            "model": self.model,
            "input": text,
            "voice": voice,
        }
        if instruct:
            payload["instruct"] = instruct
        if speed != 1.0:
            payload["speed"] = speed
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if seed is not None:
            payload["seed"] = int(seed)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if repetition_penalty is not None:
            payload["repetition_penalty"] = float(repetition_penalty)
        resp = httpx.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        samples, sample_rate = read_wav_bytes(resp.content)
        self.sample_rate = sample_rate
        return SynthesisResult(samples=samples, sample_rate=sample_rate)


class Qwen3MlxEngine(TtsEngine):
    """Direct Qwen3-TTS inference via mlx-audio (Apple Silicon)."""

    sample_rate = 24000

    def __init__(self, model: str, *, temperature: float = 0.7, instruct: str = "") -> None:
        self._temperature = temperature
        self._instruct = instruct
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed in the active Python environment.\n"
                "  MLX TTS uses a separate venv (not `uv run` from the Light workspace):\n"
                "    ./scripts/setup/setup_mlx_venv.sh\n"
                "    source .venv-mlx/bin/activate\n"
                "    python scripts/tts/tts_dub.py output/<run> --lang zh --skip-mix\n"
                "  Or mock without mlx: uv run python scripts/tts/tts_dub.py ... --engine mock\n"
                "  Or HTTP engine: --engine http with mlx_audio.server running."
            ) from exc
        logger.info("Loading Qwen3-TTS model: %s", model)
        try:
            self._model = load_model(model)
        except Exception as exc:
            msg = str(exc)
            if "HF_HUB_OFFLINE" in msg or "outgoing traffic has been disabled" in msg:
                raise RuntimeError(
                    f"Qwen3-TTS model not cached locally and HuggingFace offline mode is on.\n"
                    f"  One-time download:\n"
                    f'    HF_HUB_OFFLINE=0 python -c "from mlx_audio.tts.utils import load_model; '
                    f"load_model('{model}')\"\n"
                    f"  Or run dubbing with:\n"
                    f"    HF_HUB_OFFLINE=0 uv run light-tts dub <output_dir> --lang zh\n"
                    f"  (Your shell may set HF_HUB_OFFLINE=1 in ~/.zshrc — unset for TTS only.)"
                ) from exc
            raise
        if hasattr(self._model, "sample_rate"):
            self.sample_rate = int(self._model.sample_rate)

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
        kwargs: dict[str, object] = {
            "text": text,
            "voice": voice,
            "lang_code": _lang_code(language),
            "temperature": self._temperature,
        }
        style = instruct if instruct is not None else self._instruct
        if style:
            kwargs["instruct"] = style
        if speed != 1.0:
            kwargs["speed"] = speed
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if seed is not None:
            kwargs["seed"] = int(seed)
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        results = self._generate_with_supported_kwargs(kwargs)
        if not results:
            raise RuntimeError("Qwen3-TTS returned no audio")
        audio = results[0].audio
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        sr = getattr(results[0], "sample_rate", None) or self.sample_rate
        self.sample_rate = int(sr)
        return SynthesisResult(samples=samples, sample_rate=self.sample_rate)

    def _generate_with_supported_kwargs(self, kwargs: dict[str, object]) -> list[object]:
        """Call mlx-audio while tolerating version-specific optional kwargs."""
        unsupported: list[str] = []
        while True:
            try:
                return list(self._model.generate(**kwargs))
            except TypeError as exc:
                msg = str(exc)
                removed = False
                for key in ("seed", "max_tokens", "top_k", "top_p", "repetition_penalty"):
                    if key in kwargs and key in msg:
                        kwargs.pop(key, None)
                        unsupported.append(key)
                        removed = True
                        break
                if not removed:
                    raise
                logger.warning("mlx-audio rejected %s; retrying without it", unsupported[-1])
