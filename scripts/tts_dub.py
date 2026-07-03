#!/usr/bin/env python3
"""Full dub pipeline — use with ``.venv-mlx`` (mlx-audio), not root ``uv run``.

Setup::

    ./scripts/setup_mlx_venv.sh
    source .venv-mlx/bin/activate

Run::

    python scripts/tts_dub.py output/William_Bill_Maher --lang zh --skip-mix
    python scripts/tts_dub.py output/William_Bill_Maher --lang zh --skip-mix --resume

Mock (no mlx, uses root ``uv run``)::

    uv run python scripts/tts_dub.py output/William_Bill_Maher --lang zh --engine mock --skip-mix
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "packages" / "light-tts" / "src"))

from light_tts.cli import dub  # noqa: E402

if __name__ == "__main__":
    import typer

    typer.run(dub)
