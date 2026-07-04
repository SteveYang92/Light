from __future__ import annotations

from ..config import EngineMode, TtsConfig
from .base import TtsEngine

__all__ = ["IndexTTS2Engine", "MockEngine", "Qwen3HttpEngine", "Qwen3MlxEngine", "create_engine"]


def create_engine(config: TtsConfig) -> TtsEngine:
    if config.engine_mode == EngineMode.MOCK:
        from .mock import MockEngine

        return MockEngine()
    if config.engine_mode == EngineMode.INDEXTTS2:
        from .indextts2 import IndexTTS2Engine

        return IndexTTS2Engine(config)
    if config.engine_mode == EngineMode.HTTP:
        from .qwen3 import Qwen3HttpEngine

        return Qwen3HttpEngine(config.mlx_server_url, config.model)
    from .qwen3 import Qwen3MlxEngine

    return Qwen3MlxEngine(config.model, temperature=config.temperature)
