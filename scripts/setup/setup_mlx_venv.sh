#!/usr/bin/env bash
# Create an isolated uv venv for mlx-audio + Qwen3-TTS (Apple Silicon).
# Keeps MLX/torch stack separate from the Light whisperx workspace.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv-mlx ]]; then
  uv venv .venv-mlx --python 3.12
fi

# shellcheck disable=SC1091
source .venv-mlx/bin/activate
uv pip install mlx-audio
uv pip install -e "packages/light-tts"

echo ""
echo "MLX TTS venv ready: source ${ROOT}/.venv-mlx/bin/activate"
echo "Run dub:  light-tts dub output/<run> --lang zh --skip-mix"
echo "Or:       python scripts/tts/tts_dub.py output/<run> --lang zh --skip-mix"
echo "Model (auto-download on first run): mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
