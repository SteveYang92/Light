#!/usr/bin/env python3
# ruff: noqa: B008
"""IndexTTS2 dub pipeline — runs in the official index-tts uv venv (auto re-exec).

Setup (once)::

    git clone https://github.com/index-tts/index-tts .cache/indextts-official/index-tts
    cd .cache/indextts-official/index-tts && uv sync
    # download checkpoints to checkpoints/

Prepare reference audio at ``<run>/tts/ref.wav`` (or set ``ref_audio`` in indextts2.yaml).

Run::

    uv run python scripts/indextts_dub.py output/<run> --lang zh --skip-mix --preview
    uv run python scripts/indextts_dub.py output/<run> --lang zh --skip-mix --resume
    uv run python scripts/indextts_dub.py output/<run> --lang zh --mix duck
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "packages" / "light-tts" / "src"))

from light_tts.config import EngineMode, MixMode, TtsConfig  # noqa: E402
from light_tts.dub import run_dub  # noqa: E402
from light_tts.indextts2_runtime import maybe_reexec_in_official_venv  # noqa: E402

DEFAULT_OFFICIAL_ROOT = _REPO / ".cache/indextts-official/index-tts"


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
        resume: bool = typer.Option(True, "--resume/--no-resume"),
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
