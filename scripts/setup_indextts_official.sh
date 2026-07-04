#!/usr/bin/env bash
# Initialize official IndexTTS submodule and uv venv (vendor/index-tts).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENDOR="${ROOT}/vendor/index-tts"
LEGACY="${ROOT}/.cache/indextts-official/index-tts"
WITH_V15=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-v15)
      WITH_V15=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--with-v15]" >&2
      exit 1
      ;;
  esac
done

git submodule update --init vendor/index-tts

# One-time migration from legacy .cache layout (local only).
if [[ -d "$LEGACY" ]] && [[ ! -d "$VENDOR/checkpoints" || -z "$(ls -A "$VENDOR/checkpoints" 2>/dev/null)" ]]; then
  if [[ -d "$LEGACY/checkpoints" ]] && [[ -n "$(ls -A "$LEGACY/checkpoints" 2>/dev/null)" ]]; then
    echo "Migrating checkpoints from $LEGACY/checkpoints -> $VENDOR/checkpoints"
    mkdir -p "$VENDOR/checkpoints"
    cp -a "$LEGACY/checkpoints/." "$VENDOR/checkpoints/"
  fi
  if [[ -d "$LEGACY/hf_cache" ]] && [[ ! -d "$VENDOR/hf_cache" ]]; then
    echo "Migrating hf_cache from $LEGACY/hf_cache -> $VENDOR/hf_cache"
    cp -a "$LEGACY/hf_cache" "$VENDOR/hf_cache"
  fi
  if [[ -d "$LEGACY/.venv" ]] && [[ ! -d "$VENDOR/.venv" ]]; then
    echo "Migrating official .venv from $LEGACY/.venv -> $VENDOR/.venv"
    cp -a "$LEGACY/.venv" "$VENDOR/.venv"
  fi
  echo "Legacy install at $LEGACY — you may remove it after verifying vendor/index-tts."
fi

echo "Syncing official index-tts uv environment..."
(cd "$VENDOR" && uv sync)

echo "Installing light-tts into official venv (dub CLI deps)..."
uv pip install --python "$VENDOR/.venv/bin/python" -e "$ROOT/packages/light-tts"

if [[ ! -f "$VENDOR/checkpoints/config.yaml" ]]; then
  echo ""
  echo "Checkpoints not found under vendor/index-tts/checkpoints/"
  echo "Download IndexTTS-2 weights (see vendor/INDEX-TTS.md):"
  echo "  cd vendor/index-tts"
  echo "  hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints"
  echo "  # or: modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints"
  exit 1
fi

if [[ "$WITH_V15" == true ]]; then
  if [[ ! -f "$VENDOR/checkpoints-v15/config.yaml" ]]; then
    echo ""
    echo "Downloading IndexTTS-1.5 weights to vendor/index-tts/checkpoints-v15/ ..."
    (cd "$VENDOR" && hf download IndexTeam/IndexTTS-1.5 --local-dir=checkpoints-v15)
  else
    echo "IndexTTS-1.5 checkpoints already present at vendor/index-tts/checkpoints-v15/"
  fi
fi

echo ""
echo "Official IndexTTS ready at vendor/index-tts"
echo "Preview (v2): uv run python scripts/indextts_dub.py output/<run> --lang zh --skip-mix --preview"
echo "Preview (v1.5): uv run python scripts/indextts_dub.py output/<run> --engine indextts15 --lang zh --skip-mix --preview"
