# Official IndexTTS vendor record

| Field | Value |
|---|---|
| Upstream | https://github.com/index-tts/index-tts |
| Submodule path | `vendor/index-tts` |
| Pinned SHA | `7264ce2a9a0924becb6b8da3f60725f7663de089` |
| Pin note | `fix: fix indextts2 model resource checks (#707)` |

Light **vendors upstream code only** — model weights are downloaded locally and gitignored.

## First-time setup

```bash
./scripts/setup_indextts_official.sh
```

## Checkpoints (local, not in git)

Download IndexTTS-2 weights into `vendor/index-tts/checkpoints/`:

```bash
cd vendor/index-tts
# HuggingFace
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
# or ModelScope
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

Verify: `vendor/index-tts/checkpoints/config.yaml` and `gpt.pth` exist.

## Upgrade upstream

```bash
./scripts/vendor_sync_indextts.sh <new-sha>
```

Then update the pinned SHA in this file, re-run `uv sync` inside `vendor/index-tts`, and smoke-test IndexTTS2 preview.

## Override path

Default: `vendor/index-tts` (see `packages/light-tts/src/light_tts/assets/indextts2.yaml`).

CLI: `--official-root /path/to/index-tts`
