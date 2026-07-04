from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from ..audio_io import read_wav, trim_edge_silence
from ..config import IndexTTSVersion, TtsConfig
from ..indextts_runtime import (
    INDEXTTS2_SAMPLE_RATE,
    default_checkpoints,
    emotion_vector,
    load_official_tts,
    resolve_ref_audio_path,
    resolve_torch_compile,
    variant_spec,
)
from .base import SynthesisResult, TtsEngine

logger = logging.getLogger(__name__)


class OfficialIndexTTSEngine(TtsEngine):
    """Official IndexTTS voice cloning (1.5 or 2.0) via vendor/index-tts."""

    sample_rate = INDEXTTS2_SAMPLE_RATE

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._version: IndexTTSVersion = config.indextts_resolved_version
        spec = variant_spec(self._version)
        self.sample_rate = spec.sample_rate

        official_root = Path(config.indextts_official_root).expanduser().resolve()
        checkpoints = (
            Path(config.indextts_checkpoints).expanduser().resolve()
            if config.indextts_checkpoints
            else default_checkpoints(official_root, self._version)
        )
        if not official_root.is_dir():
            raise FileNotFoundError(f"Official index-tts repo not found: {official_root}")

        torch_compile = resolve_torch_compile(config.indextts_torch_compile)
        if config.indextts_torch_compile and not torch_compile:
            logger.info("IndexTTS torch.compile disabled on this device (MPS/CPU)")
        logger.info("Loading official IndexTTS %s from %s", self._version, checkpoints)
        self._tts = load_official_tts(
            self._version,
            official_root,
            checkpoints,
            use_fp16=config.indextts_use_fp16,
            use_torch_compile=torch_compile,
        )
        self._emo_vector = (
            emotion_vector(config.indextts_emotion, config.indextts_emotion_weight)
            if config.indextts_supports_emotion
            else None
        )

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
            if self._version == "1.5":
                self._infer_v15(ref_audio, text, output_path)
            else:
                self._infer_v2(ref_audio, text, output_path)
            samples, sample_rate = read_wav(output_path)
            trimmed = trim_edge_silence(samples, sample_rate)
            return SynthesisResult(samples=trimmed, sample_rate=sample_rate)
        finally:
            output_path.unlink(missing_ok=True)

    def _infer_v15(self, ref_audio: Path, text: str, output_path: Path) -> None:
        infer_kwargs: dict[str, Any] = {
            "audio_prompt": str(ref_audio),
            "text": text,
            "output_path": str(output_path),
            "verbose": self._config.indextts_verbose,
            "num_beams": self._config.indextts_num_beams,
            "max_text_tokens_per_segment": self._config.indextts_max_text_tokens_per_segment,
        }
        if self._config.indextts_use_fast:
            self._tts.infer_fast(
                **infer_kwargs,
                segments_bucket_max_size=self._config.indextts_segments_bucket_max_size,
            )
        else:
            self._tts.infer(**infer_kwargs)

    def _infer_v2(self, ref_audio: Path, text: str, output_path: Path) -> None:
        infer_kwargs: dict[str, Any] = {
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


IndexTTS2Engine = OfficialIndexTTSEngine
