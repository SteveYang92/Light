# IndexTTS2 Metal vendor record

| Field | Value |
|---|---|
| Upstream | https://github.com/raoqu/index-tts2-metal |
| Local path | `vendor/index-tts2-metal/` |
| Runtime | `mtts` (C++/Metal native binary) |
| Model bundle | `vendor/index-tts2-metal/bin/` (MIT2 format) |

Light uses this as an **optional** Apple Silicon acceleration path for IndexTTS2 RTF experiments. It does **not** replace the official PyTorch vendor at `vendor/index-tts`.

## First-time setup

```bash
./scripts/setup_indextts2_metal.sh
```

Options:

- `--build-from-source` — clone upstream and compile with `./build.sh` instead of the GitHub release binary.
- `--skip-model` — install `mtts` only; download the model bundle separately.

## Model bundle (local, not in git)

Pre-converted MIT2 weights (several GB):

| Source | Command |
|--------|---------|
| HuggingFace | `hf download raoqu/index-tts2-metal --local-dir vendor/index-tts2-metal/bin` |
| ModelScope | `modelscope download --model iwannaido/index-tts2-metal --local_dir vendor/index-tts2-metal/bin` |

Verify: `vendor/index-tts2-metal/bin/` contains manifest/metadata from the upstream bundle.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MIT2_ROOT` | `vendor/index-tts2-metal` | Install root |
| `MIT2_BIN` | `$MIT2_ROOT/mtts` | Native binary |
| `MIT2_MODEL_BUNDLE` | `$MIT2_ROOT/bin` | MIT2 model directory |
| `MIT2_CFM_STEPS` | `16` | CFM synthesis steps (12–25; lower = faster) |

## Differences vs official IndexTTS2 (PyTorch)

| Feature | Official `vendor/index-tts` | Metal `mtts` |
|---------|----------------------------|--------------|
| Platform | macOS MPS / CUDA / CPU | Apple Silicon macOS only |
| Weights | PyTorch checkpoints | MIT2 bundle |
| Voice input | `ref.wav` per infer | Clone once → voice bundle / voice_id |
| Speed knob | `num_beams` | `--cfm_steps` / `MIT2_CFM_STEPS` |
| Emotion vector | Supported (2.0) | **Not supported** |
| Sample rate | 22050 Hz | 22050 Hz |

## Dub pipeline (light-tts)

Start mtts server (recommended — leave running across dub/resume):

```bash
MIT2_CFM_STEPS=16 vendor/index-tts2-metal/mtts --server \
  --host 127.0.0.1 --port 3456 \
  --model_bundle vendor/index-tts2-metal/bin \
  --voice_store vendor/index-tts2-metal/voices
```

Run dub with Metal engine:

```bash
uv run python scripts/indextts_dub.py output/<run> \
  --engine indextts2_metal --lang zh --skip-mix --preview

# or via light-tts CLI
uv run light-tts dub output/<run> --engine indextts2_metal --lang zh --skip-mix --preview
```

Or set in run-local `indextts.yaml`:

```yaml
engine: indextts2_metal
metal_url: http://127.0.0.1:3456
metal_cfm_steps: 16
metal_manage_server: false
```

Voice clones are cached at `output/<run>/tts/metal_voices.json` and re-created when `ref.wav` changes.

Optional: `--metal-manage-server` auto-starts/stops a local mtts subprocess (POC-style).

## RTF POC

```bash
# Official PyTorch baseline
uv run python scripts/indextts2_poc.py --run-dir output/<run> --preview-duration 180

# Metal native (server mode, default)
uv run python scripts/indextts2_metal_poc.py --run-dir output/<run> --preview-duration 180 --cfm-steps 16

# A/B summary
uv run python scripts/indextts2_rtf_compare.py --run-dir output/<run>
```

For fair multi-chunk RTF, use **server mode** (model loaded once). CLI mode reloads the model per invocation and inflates RTF.

## Manual server

```bash
MIT2_CFM_STEPS=16 vendor/index-tts2-metal/mtts --server \
  --host 127.0.0.1 --port 3456 \
  --model_bundle vendor/index-tts2-metal/bin \
  --voice_store vendor/index-tts2-metal/voices
```

HTTP API: see upstream [docs/api.md](https://github.com/raoqu/index-tts2-metal/blob/main/docs/api.md).
