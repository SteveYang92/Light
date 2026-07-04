#!/usr/bin/env bash
# Bump the vendored official index-tts submodule to an explicit upstream SHA.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NEW_SHA="${1:-}"
if [[ -z "$NEW_SHA" ]]; then
  echo "Usage: $0 <upstream-sha>" >&2
  echo "Current: $(git -C vendor/index-tts rev-parse HEAD)" >&2
  exit 1
fi

git submodule update --init vendor/index-tts
git -C vendor/index-tts fetch origin
git -C vendor/index-tts checkout "$NEW_SHA"

echo "Updated vendor/index-tts to $NEW_SHA"
echo "Update vendor/INDEX-TTS.md pinned SHA and re-run: cd vendor/index-tts && uv sync"
