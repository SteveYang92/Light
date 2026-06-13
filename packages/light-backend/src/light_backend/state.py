from __future__ import annotations

from .config import BackendConfig

_config: BackendConfig | None = None


def get_config() -> BackendConfig:
    assert _config is not None
    return _config


def set_config(cfg: BackendConfig) -> None:
    global _config
    _config = cfg
