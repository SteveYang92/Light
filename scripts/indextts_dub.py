#!/usr/bin/env python3
# ruff: noqa: B008
"""IndexTTS2 dub pipeline — runs in the official index-tts uv venv (auto re-exec).

Setup (once)::

    ./scripts/setup_indextts_official.sh
    # downloads checkpoints if missing — see vendor/INDEX-TTS.md

Prepare reference audio at ``<run>/tts/ref.wav`` (or set ``ref_audio`` in indextts2.yaml).

Run::

    uv run python scripts/indextts_dub.py output/<run> --lang zh --skip-mix --preview
    uv run python scripts/indextts_dub.py output/<run> --lang zh --skip-mix --resume
    uv run python scripts/indextts_dub.py output/<run> --lang zh --mix duck
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ROOT = _REPO / "vendor/index-tts"


def _parse_official_root(argv: list[str]) -> Path:
    for index, arg in enumerate(argv):
        if arg == "--official-root" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if arg.startswith("--official-root="):
            return Path(arg.split("=", 1)[1])
    return DEFAULT_OFFICIAL_ROOT


def _maybe_reexec_early() -> None:
    """Re-exec in official venv before importing light_tts (root uv may lack httpx)."""
    official_root = _parse_official_root(sys.argv[1:]).expanduser().resolve()
    official_py = official_root / ".venv" / "bin" / "python"
    if not official_py.is_file():
        return
    if Path(sys.executable).resolve() == official_py.resolve():
        return
    tts_src = str(_REPO / "packages" / "light-tts" / "src")
    env = os.environ.copy()
    parts = [tts_src, str(official_root)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    os.execve(str(official_py), [str(official_py), *sys.argv], env)


_maybe_reexec_early()

sys.path.insert(0, str(_REPO / "packages" / "light-tts" / "src"))

from light_tts.config import EngineMode, MixMode, TtsConfig  # noqa: E402
from light_tts.dub import run_dub  # noqa: E402
from light_tts.indextts2_runtime import maybe_reexec_in_official_venv  # noqa: E402


def main() -> None:
    import typer

    app = typer.Typer(no_args_is_help=True)

    @app.command()
    def dub(
        output_dir: str = typer.Argument(..., help="Pipeline output directory"),
        lang: str = typer.Option("zh", "--lang"),
        official_root: Path = typer.Option(DEFAULT_OFFICIAL_ROOT, "--official-root"),
        ref_audio: Path | None = typer.Option(None, "--ref-audio", help="Speaker reference WAV"),
        checkpoints: Path | None = typer.Option(None, "--checkpoints"),
        emotion: str = typer.Option("calm", "--emotion"),
        emotion_weight: float = typer.Option(0.6, "--emotion-weight"),
        num_beams: int = typer.Option(3, "--num-beams"),
        chunk_chars: int = typer.Option(160, "--chunk-chars"),
        chunk_min_chars: int = typer.Option(45, "--chunk-min-chars"),
        mix: MixMode = typer.Option(MixMode.DUCK, "--mix"),
        cues: Path | None = typer.Option(None, "--cues"),
        video: Path | None = typer.Option(None, "--video"),
        max_cues: int | None = typer.Option(None, "--max-cues"),
        resume: bool = typer.Option(False, "--resume", help="Reuse existing segment WAVs (skip synthesis)"),
        skip_mix: bool = typer.Option(False, "--skip-mix"),
        preview: bool = typer.Option(False, "--preview"),
        preview_duration: float = typer.Option(180.0, "--preview-duration"),
        per_cue: bool = typer.Option(False, "--per-cue"),
        verbose: bool = typer.Option(False, "--verbose"),
    ) -> None:
        maybe_reexec_in_official_venv(official_root=official_root, enabled=True)
        cfg = TtsConfig(
            output_dir=output_dir,
            lang=lang,
            engine_mode=EngineMode.INDEXTTS2,
            indextts_official_root=str(official_root),
            indextts_checkpoints=str(checkpoints) if checkpoints else None,
            indextts_ref_audio=str(ref_audio) if ref_audio else None,
            indextts_emotion=emotion,
            indextts_emotion_weight=emotion_weight,
            indextts_num_beams=num_beams,
            indextts_chunk_chars=chunk_chars,
            indextts_chunk_min_chars=chunk_min_chars,
            indextts_verbose=verbose,
            mix_mode=mix,
            max_cues=max_cues,
            resume=resume,
            video=str(video) if video else None,
            preview=preview,
            preview_duration_s=preview_duration,
            per_cue=per_cue,
        )
        run_dub(cfg, cues_path=cues, skip_mix=skip_mix)

    app()


if __name__ == "__main__":
    main()
