from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .config import TtsConfig

EMOTION_INDEX = {
    "happy": 0,
    "angry": 1,
    "sad": 2,
    "afraid": 3,
    "disgusted": 4,
    "melancholic": 5,
    "surprised": 6,
    "calm": 7,
}


# Official IndexTTS2 infer_v2 writes 22050 Hz audio (see infer_v2.py sampling_rate = 22050).
INDEXTTS2_SAMPLE_RATE = 22050


def emotion_vector(name: str, weight: float) -> list[float] | None:
    if name == "none":
        return None
    vector = [0.0] * len(EMOTION_INDEX)
    vector[EMOTION_INDEX[name]] = weight
    return vector


def official_python(official_root: Path) -> Path | None:
    candidate = official_root / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def official_run_hint(official_root: Path) -> str:
    return (
        "Run IndexTTS2 dub with the official index-tts uv environment:\n"
        f"  cd {official_root} && PYTHONPATH=. uv run light-tts dub <output_dir> --engine indextts2 ..."
    )


def maybe_reexec_in_official_venv(*, official_root: Path, enabled: bool) -> None:
    """Re-exec the current process inside the official repo venv when deps are missing."""
    if not enabled:
        return
    official_py = official_python(official_root.resolve())
    if official_py is None:
        return
    if Path(sys.executable).resolve() == official_py.resolve():
        return
    import light_tts

    tts_src = str(Path(light_tts.__file__).resolve().parent.parent)
    env = os.environ.copy()
    root = str(official_root.resolve())
    parts = [tts_src, root]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    os.execve(str(official_py), [str(official_py), *sys.argv], env)


def load_official_tts(
    official_root: Path,
    checkpoints: Path,
    *,
    use_fp16: bool,
    use_torch_compile: bool,
) -> Any:
    sys.path.insert(0, str(official_root.resolve()))
    from indextts.infer_v2 import IndexTTS2

    cfg_path = checkpoints / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing official checkpoint config: {cfg_path}")
    return IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(checkpoints),
        use_fp16=use_fp16,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_torch_compile=use_torch_compile,
    )


def resolve_ref_audio_path(config: TtsConfig, speaker: str) -> Path:
    speaker_key = speaker.strip() or "__default__"
    if speaker_key in config.indextts_speaker_refs:
        path = Path(config.indextts_speaker_refs[speaker_key]).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Speaker ref audio not found for {speaker_key}: {path}")
    if config.indextts_ref_audio:
        path = Path(config.indextts_ref_audio).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Reference audio not found: {path}")
    tts_ref = Path(config.output_dir) / "tts" / "ref.wav"
    if tts_ref.is_file():
        return tts_ref.resolve()
    legacy = Path(config.output_dir) / "tts_indextts" / "ref.wav"
    if legacy.is_file():
        return legacy.resolve()
    raise FileNotFoundError(
        f"No reference audio configured. Set ref_audio in indextts2.yaml, speaker_refs, or place ref.wav at {tts_ref}"
    )


def resolve_torch_compile(requested: bool) -> bool:
    if not requested:
        return False
    try:
        import torch
    except ImportError:
        return False
    if torch.backends.mps.is_available():
        return False
    return True
