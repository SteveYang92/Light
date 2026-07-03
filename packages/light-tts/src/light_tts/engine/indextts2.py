from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..audio_io import read_wav, trim_edge_silence
from ..config import TtsConfig
from ..indextts2_runtime import (
    INDEXTTS2_SAMPLE_RATE,
    emotion_vector,
    load_official_tts,
    resolve_ref_audio_path,
    resolve_torch_compile,
)
from .base import SynthesisResult, TtsEngine

logger = logging.getLogger(__name__)


class IndexTTS2Engine(TtsEngine):
    """Official IndexTTS2 voice cloning via the cloned index-tts repository."""

    sample_rate = INDEXTTS2_SAMPLE_RATE

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        official_root = Path(config.indextts_official_root).expanduser().resolve()
        checkpoints = (
            Path(config.indextts_checkpoints).expanduser().resolve()
            if config.indextts_checkpoints
            else official_root / "checkpoints"
        )
        if not official_root.is_dir():
            raise FileNotFoundError(f"Official index-tts repo not found: {official_root}")
        torch_compile = resolve_torch_compile(config.indextts_torch_compile)
        if config.indextts_torch_compile and not torch_compile:
            logger.info("IndexTTS2 torch.compile disabled on this device (MPS/CPU)")
        logger.info("Loading official IndexTTS2 from %s", checkpoints)
        self._tts = load_official_tts(
            official_root,
            checkpoints,
            use_fp16=config.indextts_use_fp16,
            use_torch_compile=torch_compile,
        )
        self._emo_vector = emotion_vector(config.indextts_emotion, config.indextts_emotion_weight)

    def resolve_ref_audio(self, speaker: str) -> Path:
        return resolve_ref_audio_path(self._config, speaker)

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
        ref_audio = self.resolve_ref_audio(voice)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)
        try:
            infer_kwargs: dict[str, object] = {
                "spk_audio_prompt": str(ref_audio),
                "text": text,
                "output_path": str(output_path),
                "use_random": self._config.indextts_use_random,
                "verbose": self._config.indextts_verbose,
                "num_beams": self._config.indextts_num_beams,
            }
            if self._emo_vector is not None:
                infer_kwargs["emo_vector"] = self._emo_vector
            self._tts.infer(**infer_kwargs)
            samples, sample_rate = read_wav(output_path)
            trimmed = trim_edge_silence(samples, sample_rate)
            return SynthesisResult(samples=trimmed, sample_rate=sample_rate)
        finally:
            output_path.unlink(missing_ok=True)
