#!/usr/bin/env python3
# ruff: noqa: B008
"""Batch IndexTTS2 dubbing for multi-segment episode directories.

Each segment under ``<episode>/.segN/`` is processed independently. Episode-level
``merge`` stitches ``tts/dub.wav`` files using ``split_points.json`` overlap rules.

Examples::

    uv run python scripts/tts/indextts_dub_batch.py output/Dan_Carlins_... --prepare-ref
    uv run python scripts/tts/indextts_dub_batch.py output/Dan_Carlins_... \\
        --engine indextts2_metal --metal-cfm-steps 20 --skip-mix
    uv run python scripts/tts/indextts_dub_batch.py output/Dan_Carlins_... --mix-only --mix duck
    uv run python scripts/tts/indextts_dub_batch.py output/Dan_Carlins_... --merge --mix duck
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


_DEFAULT_ENGINE = "indextts2"
_METAL_ENGINE = "indextts2_metal"


def _parse_engine(argv: list[str]) -> str:
    for index, arg in enumerate(argv):
        if arg == "--engine" and index + 1 < len(argv):
            return argv[index + 1].lower()
        if arg.startswith("--engine="):
            return arg.split("=", 1)[1].lower()
    return _DEFAULT_ENGINE


def _maybe_reexec_early() -> None:
    if "--mix-only" in sys.argv[1:] or "--prepare-ref" in sys.argv[1:]:
        return
    if _parse_engine(sys.argv[1:]) == _METAL_ENGINE:
        return
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
from light_tts.dub import run_dub, run_mix_only  # noqa: E402
from light_tts.episode_merge import (  # noqa: E402
    discover_segments,
    merge_episode_dub,
    prepare_ref_audio,
)
from light_tts.indextts_runtime import maybe_reexec_in_official_venv  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()


def main() -> None:
    import typer

    app = typer.Typer(no_args_is_help=True)

    @app.command()
    def batch(
        episode_dir: str = typer.Argument(..., help="Episode root with .seg*/ subdirectories"),
        lang: str = typer.Option("zh", "--lang"),
        engine: EngineMode = typer.Option(EngineMode.INDEXTTS2, "--engine"),
        official_root: Path = typer.Option(DEFAULT_OFFICIAL_ROOT, "--official-root"),
        metal_cfm_steps: int = typer.Option(16, "--metal-cfm-steps"),
        ref_audio: Path | None = typer.Option(None, "--ref-audio"),
        checkpoints: Path | None = typer.Option(None, "--checkpoints"),
        emotion: str = typer.Option("calm", "--emotion"),
        emotion_weight: float = typer.Option(0.6, "--emotion-weight"),
        num_beams: int = typer.Option(3, "--num-beams"),
        chunk_chars: int = typer.Option(160, "--chunk-chars"),
        chunk_min_chars: int = typer.Option(45, "--chunk-min-chars"),
        mix: MixMode = typer.Option(MixMode.DUCK, "--mix"),
        resume: bool = typer.Option(False, "--resume"),
        skip_mix: bool = typer.Option(False, "--skip-mix", help="Synthesis only; skip per-segment ffmpeg mix"),
        mix_only: bool = typer.Option(False, "--mix-only", help="Only ffmpeg mix existing tts/dub.wav per segment"),
        merge: bool = typer.Option(False, "--merge", help="Merge segment dubs into episode dub_full.wav (+ mix)"),
        prepare_ref: bool = typer.Option(False, "--prepare-ref", help="Copy .seg1/tts/ref.* to other segments"),
        preview: bool = typer.Option(False, "--preview"),
        preview_duration: float = typer.Option(180.0, "--preview-duration"),
        segments: str = typer.Option("", "--segments", help="Comma list, e.g. .seg2,.seg3 (default: all)"),
        verbose: bool = typer.Option(False, "--verbose"),
    ) -> None:
        root = Path(episode_dir).resolve()
        seg_dirs = discover_segments(root)
        if not seg_dirs:
            raise typer.BadParameter(f"No .seg*/ directories under {root}")

        if segments.strip():
            wanted = {part.strip() for part in segments.split(",") if part.strip()}
            seg_dirs = [seg for seg in seg_dirs if seg.name in wanted]
            if not seg_dirs:
                raise typer.BadParameter(f"No segments matched --segments={segments!r}")

        if prepare_ref:
            copied = prepare_ref_audio(root)
            console.print(f"[green]Prepared ref audio in {len(copied)} segment(s)[/green]")
            if not (resume or skip_mix or mix_only or merge or preview):
                return

        if merge:
            if mix_only or skip_mix or resume or preview:
                raise typer.BadParameter("--merge cannot combine with synthesis flags")
            out = merge_episode_dub(root, mix_mode=mix, video=None)
            console.print(f"[green]✓[/green] Episode dub: {out}")
            return

        needs_indextts = not mix_only
        if needs_indextts and engine != EngineMode.INDEXTTS2_METAL:
            maybe_reexec_in_official_venv(official_root=official_root, enabled=True)

        for seg in seg_dirs:
            console.rule(f"[bold]{seg.name}[/bold]")
            cfg = TtsConfig(
                output_dir=str(seg),
                lang=lang,
                engine_mode=engine,
                indextts_official_root=str(official_root),
                indextts_metal_cfm_steps=metal_cfm_steps,
                indextts_checkpoints=str(checkpoints) if checkpoints else None,
                indextts_ref_audio=str(ref_audio) if ref_audio else None,
                indextts_emotion=emotion,
                indextts_emotion_weight=emotion_weight,
                indextts_num_beams=num_beams,
                indextts_chunk_chars=chunk_chars,
                indextts_chunk_min_chars=chunk_min_chars,
                indextts_verbose=verbose,
                mix_mode=mix,
                resume=resume,
                mix_only=mix_only,
                preview=preview,
                preview_duration_s=preview_duration,
            )
            if mix_only:
                run_mix_only(cfg)
            else:
                run_dub(cfg, skip_mix=skip_mix or preview)

    app()


if __name__ == "__main__":
    main()
