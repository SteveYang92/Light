#!/usr/bin/env bash
# Install index-tts2-metal native runtime (mtts) and MIT2 model bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MIT2_ROOT="${MIT2_ROOT:-$ROOT/vendor/index-tts2-metal}"
MIT2_BIN="$MIT2_ROOT/mtts"
MODEL_BUNDLE="$MIT2_ROOT/bin"
RELEASE_TAG="v0.2.0"
RELEASE_ASSET="index-tts2-metal-${RELEASE_TAG}-macos-arm64.tar.gz"
RELEASE_URL="https://github.com/raoqu/index-tts2-metal/releases/download/${RELEASE_TAG}/${RELEASE_ASSET}"
BUILD_FROM_SOURCE=false
SKIP_MODEL=false

usage() {
  echo "Usage: $0 [--build-from-source] [--skip-model]"
  echo ""
  echo "  --build-from-source  Compile mtts from vendor/index-tts2-metal source instead of prebuilt."
  echo "  --skip-model         Skip HuggingFace MIT2 model download (mtts binary only)."
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-from-source)
      BUILD_FROM_SOURCE=true
      shift
      ;;
    --skip-model)
      SKIP_MODEL=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "index-tts2-metal requires Apple Silicon macOS (arm64)." >&2
  exit 1
fi

mkdir -p "$MIT2_ROOT"

if [[ "$BUILD_FROM_SOURCE" == true ]]; then
  if [[ ! -d "$MIT2_ROOT/.git" ]]; then
    echo "Cloning index-tts2-metal source into $MIT2_ROOT ..."
    rm -rf "$MIT2_ROOT"
    git clone --depth 1 https://github.com/raoqu/index-tts2-metal.git "$MIT2_ROOT"
  fi
  echo "Building mtts from source ..."
  (cd "$MIT2_ROOT" && ./build.sh)
  MIT2_BIN="$MIT2_ROOT/build/mtts"
else
  if [[ ! -x "$MIT2_BIN" ]]; then
    echo "Downloading prebuilt mtts (${RELEASE_TAG}) ..."
    tmpdir="$(mktemp -d)"
    curl -fsSL "$RELEASE_URL" -o "$tmpdir/$RELEASE_ASSET"
    tar -xzf "$tmpdir/$RELEASE_ASSET" -C "$tmpdir"
    pkg_dir="$tmpdir/index-tts2-metal-${RELEASE_TAG}-macos-arm64"
    if [[ ! -x "$pkg_dir/mtts" ]]; then
      echo "Prebuilt package missing mtts binary." >&2
      exit 1
    fi
    cp "$pkg_dir/mtts" "$MIT2_BIN"
    chmod +x "$MIT2_BIN"
    rm -rf "$tmpdir"
  else
    echo "mtts already present at $MIT2_BIN"
  fi
fi

if [[ ! -x "$MIT2_BIN" ]]; then
  echo "mtts binary not found at $MIT2_BIN" >&2
  exit 1
fi

if [[ "$SKIP_MODEL" == false ]]; then
  if [[ ! -f "$MODEL_BUNDLE/manifest.json" && ! -f "$MODEL_BUNDLE/model_manifest.json" && ! -f "$MODEL_BUNDLE/weights.bin" ]]; then
    echo "Downloading MIT2 model bundle to $MODEL_BUNDLE ..."
    mkdir -p "$MODEL_BUNDLE"
    if command -v hf >/dev/null 2>&1; then
      HF_HUB_OFFLINE=0 hf download raoqu/index-tts2-metal --local-dir "$MODEL_BUNDLE"
    else
      echo "huggingface-cli (hf) not found. Install it, then run:" >&2
      echo "  hf download raoqu/index-tts2-metal --local-dir $MODEL_BUNDLE" >&2
      echo "Or ModelScope:" >&2
      echo "  modelscope download --model iwannaido/index-tts2-metal --local_dir $MODEL_BUNDLE" >&2
      exit 1
    fi
  else
    echo "MIT2 model bundle already present at $MODEL_BUNDLE"
  fi
fi

mkdir -p "$MIT2_ROOT/voices" "$MIT2_ROOT/outputs"

echo ""
echo "Verifying mtts ..."
"$MIT2_BIN" --capabilities >/dev/null

echo ""
echo "IndexTTS2 Metal ready:"
echo "  MIT2_ROOT=$MIT2_ROOT"
echo "  MIT2_BIN=$MIT2_BIN"
echo "  MIT2_MODEL_BUNDLE=$MODEL_BUNDLE"
echo ""
echo "Start server:"
echo "  MIT2_CFM_STEPS=16 $MIT2_BIN --server --host 127.0.0.1 --port 3456 \\"
echo "    --model_bundle $MODEL_BUNDLE --voice_store $MIT2_ROOT/voices"
echo ""
echo "Metal RTF POC:"
echo "  uv run python scripts/tts/indextts2_metal_poc.py --run-dir output/<run> --preview-duration 180"
