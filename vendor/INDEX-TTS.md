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
# optional IndexTTS 1.5 weights:
./scripts/setup_indextts_official.sh --with-v15
```

## Checkpoints (local, not in git)

Two independent weight directories (do not mix):

| Version | Directory | HF model | Output SR |
|---------|-----------|----------|-----------|
| **2.0** (default) | `vendor/index-tts/checkpoints/` | `IndexTeam/IndexTTS-2` | 22050 Hz |
| **1.5** | `vendor/index-tts/checkpoints-v15/` | `IndexTeam/IndexTTS-1.5` | 24000 Hz |

Download IndexTTS-2 (required for default dub path):

```bash
cd vendor/index-tts
# HuggingFace
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
# or ModelScope
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

Download IndexTTS-1.5 (optional, for `--engine indextts15`):

```bash
cd vendor/index-tts
hf download IndexTeam/IndexTTS-1.5 --local-dir=checkpoints-v15
```

Verify: `config.yaml` and `gpt.pth` exist under each directory.

Local artifacts (`checkpoints/`, `checkpoints-v15/`, `.venv/`, `hf_cache/`) stay inside the submodule working tree only. `.gitmodules` sets `ignore = all` so Light `git status` does not flag them as dirty.

## Upgrade upstream

```bash
./scripts/vendor_sync_indextts.sh <new-sha>
```

Then update the pinned SHA in this file, re-run `uv sync` inside `vendor/index-tts`, and smoke-test IndexTTS preview (v2 and optionally v1.5).

## Override path

Default: `vendor/index-tts` (see `packages/light-tts/src/light_tts/assets/indextts.yaml`).

CLI: `--official-root /path/to/index-tts`

Engine selection:

- `--engine indextts2` — IndexTTS 2.0 (emotion control, 22050 Hz)
- `--engine indextts15` — IndexTTS 1.5 (24000 Hz, no emotion vector)

IndexTTS 1.5 uses official `infer_fast()` by default (`indextts_use_fast: true` in yaml). The old slow path mapped `chunk_chars` → token segments incorrectly; tune `indextts_max_text_tokens_per_segment` (smaller = faster, default 100) and `num_beams` if RTF is still high.
