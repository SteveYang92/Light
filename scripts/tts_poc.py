#!/usr/bin/env python3
"""Phase 0 TTS POC — read cues.json, synthesize with Qwen3-TTS preset voices.

Requires mlx-audio in an isolated venv::

    ./scripts/setup_mlx_venv.sh
    source .venv-mlx/bin/activate
    uv pip install -e packages/light-tts

Or use mock engine without mlx::

    uv run python scripts/tts_poc.py --cues ... --out ... --engine mock
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root before package install.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "packages" / "light-tts" / "src"))

from light_tts.cli import poc_cmd  # noqa: E402

if __name__ == "__main__":
    import typer

    typer.run(poc_cmd)
