from __future__ import annotations

from ..config import EngineMode, TtsConfig
from .base import TtsEngine
from .indextts2 import IndexTTS2Engine
from .mock import MockEngine
from .qwen3 import Qwen3HttpEngine, Qwen3MlxEngine

__all__ = ["IndexTTS2Engine", "MockEngine", "Qwen3HttpEngine", "Qwen3MlxEngine", "create_engine"]


def create_engine(config: TtsConfig) -> TtsEngine:
    if config.engine_mode == EngineMode.MOCK:
        return MockEngine()
    if config.engine_mode == EngineMode.INDEXTTS2:
        return IndexTTS2Engine(config)
    if config.engine_mode == EngineMode.HTTP:
        return Qwen3HttpEngine(config.mlx_server_url, config.model)
    return Qwen3MlxEngine(config.model, temperature=config.temperature)
