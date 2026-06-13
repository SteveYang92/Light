"""Whisper binary and model auto-detection utilities."""

import os
import shutil
from pathlib import Path


def find_whisper(current: str = "whisper-cli") -> str:
    """Auto-detect the whisper-cli binary path."""
    # 1. If already a valid path, use it
    if current and current != "whisper-cli" and Path(current).exists():
        return current
    # 2. Environment variable
    if env := os.environ.get("WHISPER_PATH"):
        return env
    # 3. ~/whisper.cpp/build/bin/
    for base in [Path.home() / "whisper.cpp", Path.cwd()]:
        for name in ["whisper-cli", "main"]:
            p = base / "build" / "bin" / name
            if p.exists():
                return str(p)
    # 4. PATH
    if shutil.which(current or "whisper-cli"):
        return current or "whisper-cli"
    return current or "whisper-cli"


def find_model(model_name: str, whisper_path: str) -> str:
    """Resolve the whisper model file path."""
    # 1. Already a valid path
    if Path(model_name).exists():
        return model_name
    # 2. Same directory as whisper-cli's models/
    whisper_dir = Path(whisper_path).parent.parent / "models"
    p = whisper_dir / model_name
    if p.exists():
        return str(p)
    # 3. ~/whisper.cpp/models/
    p = Path.home() / "whisper.cpp" / "models" / model_name
    if p.exists():
        return str(p)
    # 4. Environment variable
    if env := os.environ.get("WHISPER_MODEL_DIR"):
        p = Path(env) / model_name
        if p.exists():
            return str(p)
    return model_name
