from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import indextts2_poc as poc  # noqa: E402

# Avoid corporate/IDE HTTP proxies breaking localhost mtts health checks.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(_NO_PROXY_OPENER)

RUN_DIR = poc.RUN_DIR
DEFAULT_METAL_ROOT = Path("vendor/index-tts2-metal")
DEFAULT_METAL_URL = "http://127.0.0.1:3456"
Mode = Literal["server", "cli"]


@dataclass(frozen=True)
class MetalPaths:
    root: Path
    bin: Path
    model_bundle: Path
    voice_store: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS2 Metal native RTF POC (mtts).")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Pipeline run directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="POC output directory.")
    parser.add_argument("--metal-root", type=Path, default=DEFAULT_METAL_ROOT, help="index-tts2-metal install root.")
    parser.add_argument("--metal-url", default=DEFAULT_METAL_URL, help="mtts HTTP base URL (server mode).")
    parser.add_argument("--mode", choices=["server", "cli"], default="server", help="server (default) or cli.")
    parser.add_argument("--no-start-server", action="store_true", help="Reuse an existing mtts server.")
    parser.add_argument("--keep-server", action="store_true", help="Do not stop server on exit.")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind host when auto-starting.")
    parser.add_argument("--port", type=int, default=3456, help="Server bind port when auto-starting.")
    parser.add_argument("--ref-audio", type=Path, default=None, help="Speaker reference WAV.")
    parser.add_argument("--preview-duration", type=float, default=180.0, help="Source timeline seconds to include.")
    parser.add_argument("--lang", default="zh", help="Cue language to synthesize.")
    parser.add_argument("--max-chars", type=int, default=160, help="Maximum characters per chunk.")
    parser.add_argument("--min-chars", type=int, default=45, help="Minimum characters before soft splits.")
    parser.add_argument("--crossfade-ms", type=float, default=30.0, help="Crossfade between generated chunks.")
    parser.add_argument("--cfm-steps", type=int, default=16, help="CFM synthesis steps (12-25).")
    parser.add_argument("--skip-clone", action="store_true", help="Skip voice clone; use --voice-id or --voice-bundle.")
    parser.add_argument("--voice-id", default="", help="Existing server voice id.")
    parser.add_argument("--voice-bundle", type=Path, default=None, help="Existing MIT2 voice bundle path.")
    parser.add_argument("--dry-run", action="store_true", help="Only build chunks and manifest.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> MetalPaths:
    root = args.metal_root.expanduser().resolve()
    bin_path = Path(os.environ.get("MIT2_BIN", root / "mtts")).expanduser().resolve()
    model_bundle = Path(os.environ.get("MIT2_MODEL_BUNDLE", root / "bin")).expanduser().resolve()
    voice_store = root / "voices"
    return MetalPaths(root=root, bin=bin_path, model_bundle=model_bundle, voice_store=voice_store)


def metal_url(args: argparse.Namespace) -> str:
    return str(args.metal_url).rstrip("/")


def http_get_json(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], *, timeout: float = 600.0) -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "audio/wav, application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return body, headers


def http_post_multipart(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    boundary = f"----light-mit2-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(value.encode("utf-8"))
        parts.append(b"\r\n")
    for name, (filename, content, content_type) in files.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_health(base_url: str, *, timeout_s: float = 300.0) -> None:
    """Poll until mtts responds with status=ok (model load can take several minutes)."""
    deadline = time.monotonic() + timeout_s
    last_error = ""
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(f"{base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status != 200:
                    last_error = f"HTTP {resp.status}"
                else:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        if attempt > 1:
                            print(f">> mtts server healthy after {attempt} attempts")
                        return
                    last_error = f"unexpected body: {data!r}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
        if attempt == 1 or attempt % 20 == 0:
            print(f">> waiting for mtts health ({attempt}) ... {last_error}")
        time.sleep(1.0)
    raise TimeoutError(f"mtts server not healthy at {base_url}/health: {last_error}")


def start_server(args: argparse.Namespace, paths: MetalPaths) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["MIT2_CFM_STEPS"] = str(args.cfm_steps)
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    cmd = [
        str(paths.bin),
        "--server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model_bundle",
        str(paths.model_bundle),
        "--voice_store",
        str(paths.voice_store),
    ]
    paths.voice_store.mkdir(parents=True, exist_ok=True)
    print(f">> starting mtts server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    wait_for_health(metal_url(args))
    print(f">> mtts server ready at {metal_url(args)}")
    return proc


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def clone_voice_server(args: argparse.Namespace, ref_audio: Path) -> tuple[str, float]:
    started = time.perf_counter()
    payload = http_post_multipart(
        f"{metal_url(args)}/api/voices",
        {"name": "light-poc", "description": "Light IndexTTS2 Metal POC"},
        {"audio_sample": (ref_audio.name, ref_audio.read_bytes(), "audio/wav")},
    )
    voice_id = str(payload.get("id", ""))
    if not voice_id:
        raise RuntimeError(f"Voice clone failed: {payload}")
    elapsed = round(time.perf_counter() - started, 3)
    print(f">> cloned voice id={voice_id} ({elapsed}s)")
    return voice_id, elapsed


def clone_voice_cli(args: argparse.Namespace, paths: MetalPaths, ref_audio: Path) -> tuple[Path, float]:
    started = time.perf_counter()
    bundle_dir = paths.root / "voices" / "bundles" / "light-poc"
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(paths.bin),
        "--clone",
        str(paths.model_bundle),
        str(ref_audio),
        str(bundle_dir),
    ]
    env = os.environ.copy()
    env["MIT2_CFM_STEPS"] = str(args.cfm_steps)
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    subprocess.run(cmd, check=True, env=env)
    elapsed = round(time.perf_counter() - started, 3)
    print(f">> cloned voice bundle at {bundle_dir} ({elapsed}s)")
    return bundle_dir, elapsed


def synthesize_server(args: argparse.Namespace, *, voice: str, text: str, output_path: Path) -> float:
    started = time.perf_counter()
    payload = {
        "model": "mtts",
        "input": text,
        "voice": {"id": voice},
        "response_format": "wav",
    }
    wav_bytes, headers = http_post_json(f"{metal_url(args)}/v1/audio/speech", payload)
    content_type = headers.get("content-type", "")
    if "json" in content_type:
        raise RuntimeError(wav_bytes.decode("utf-8", errors="replace"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)
    return round(time.perf_counter() - started, 3)


def synthesize_cli(
    args: argparse.Namespace,
    paths: MetalPaths,
    *,
    voice_bundle: Path,
    text: str,
    output_path: Path,
) -> float:
    started = time.perf_counter()
    cmd = [
        str(paths.bin),
        f"--cfm_steps={args.cfm_steps}",
        "--tts",
        str(paths.model_bundle),
        str(voice_bundle),
        text,
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return round(time.perf_counter() - started, 3)


def resolve_voice_target(
    args: argparse.Namespace,
    paths: MetalPaths,
    ref_audio: Path,
    manifest: dict[str, Any],
) -> tuple[str | None, Path | None, float]:
    if args.voice_id:
        manifest["voice"] = {"mode": args.mode, "voice_id": args.voice_id, "clone_elapsed_s": 0.0}
        return args.voice_id, None, 0.0
    if args.voice_bundle is not None:
        bundle = args.voice_bundle.expanduser().resolve()
        manifest["voice"] = {"mode": args.mode, "voice_bundle": str(bundle), "clone_elapsed_s": 0.0}
        return None, bundle, 0.0
    if args.skip_clone:
        raise ValueError("With --skip-clone, pass --voice-id or --voice-bundle.")
    if args.mode == "server":
        voice_id, elapsed = clone_voice_server(args, ref_audio)
        manifest["voice"] = {"mode": "server", "voice_id": voice_id, "clone_elapsed_s": elapsed}
        return voice_id, None, elapsed
    bundle, elapsed = clone_voice_cli(args, paths, ref_audio)
    manifest["voice"] = {"mode": "cli", "voice_bundle": str(bundle), "clone_elapsed_s": elapsed}
    return None, bundle, elapsed


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "poc_indextts2_metal").resolve()
    ref_audio = (args.ref_audio or run_dir / "tts" / "ref.wav").resolve()
    if not ref_audio.is_file():
        ref_audio = (run_dir / "tts_indextts" / "ref.wav").resolve()
    chunks_dir = output_dir / "chunks"
    preview_path = output_dir / "preview.wav"
    manifest_path = output_dir / "manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    cues, source_path = poc.load_text_cues(run_dir, lang=args.lang, preview_duration=args.preview_duration)
    chunks = poc.build_chunks(cues, max_chars=args.max_chars, min_chars=args.min_chars)
    manifest: dict[str, Any] = {
        "status": "prepared",
        "engine": "indextts2_metal",
        "mode": args.mode,
        "run_dir": str(run_dir),
        "source": str(source_path),
        "metal_root": str(paths.root),
        "metal_bin": str(paths.bin),
        "model_bundle": str(paths.model_bundle),
        "metal_url": metal_url(args),
        "ref_audio": str(ref_audio),
        "preview_path": str(preview_path),
        "preview_duration_s": args.preview_duration,
        "lang": args.lang,
        "emotion": "unsupported",
        "cfm_steps": args.cfm_steps,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "cue_ids": chunk.cue_ids,
                "source_start_s": round(chunk.start, 3),
                "source_end_s": round(chunk.end, 3),
                "text_chars": len(chunk.text),
                "text": chunk.text,
                "output_path": str(chunks_dir / f"{chunk.chunk_id}.wav"),
                "status": "pending",
            }
            for chunk in chunks
        ],
    }
    poc.write_manifest(manifest_path, manifest)

    if args.dry_run:
        manifest["status"] = "dry_run"
        poc.write_manifest(manifest_path, manifest)
        print(f"Prepared {len(chunks)} chunks; manifest: {manifest_path}")
        return 0

    if not paths.bin.is_file():
        manifest["status"] = "blocked"
        manifest["error"] = f"mtts binary not found: {paths.bin}"
        manifest["hint"] = "Run ./scripts/setup/setup_indextts2_metal.sh"
        poc.write_manifest(manifest_path, manifest)
        print(manifest["error"], file=sys.stderr)
        print(manifest["hint"], file=sys.stderr)
        return 2
    if not paths.model_bundle.is_dir():
        manifest["status"] = "blocked"
        manifest["error"] = f"MIT2 model bundle not found: {paths.model_bundle}"
        manifest["hint"] = "Run ./scripts/setup/setup_indextts2_metal.sh"
        poc.write_manifest(manifest_path, manifest)
        print(manifest["error"], file=sys.stderr)
        return 2
    if not ref_audio.is_file():
        manifest["status"] = "blocked"
        manifest["error"] = f"Reference audio not found: {ref_audio}"
        poc.write_manifest(manifest_path, manifest)
        print(manifest["error"], file=sys.stderr)
        return 2

    server_proc: subprocess.Popen[bytes] | None = None
    started_server = False
    exit_code = 1
    try:
        if args.mode == "server":
            if args.no_start_server:
                wait_for_health(metal_url(args))
            else:
                server_proc = start_server(args, paths)
                started_server = True

        voice_id, voice_bundle, clone_elapsed = resolve_voice_target(args, paths, ref_audio, manifest)
        manifest["clone_elapsed_s"] = clone_elapsed
        poc.write_manifest(manifest_path, manifest)

        audio_chunks: list[np.ndarray] = []
        sample_rate: int | None = None

        for index, chunk in enumerate(chunks):
            chunk_path = chunks_dir / f"{chunk.chunk_id}.wav"
            entry = manifest["chunks"][index]
            try:
                if args.mode == "server":
                    assert voice_id is not None
                    elapsed = synthesize_server(args, voice=voice_id, text=chunk.text, output_path=chunk_path)
                else:
                    assert voice_bundle is not None
                    elapsed = synthesize_cli(
                        args,
                        paths,
                        voice_bundle=voice_bundle,
                        text=chunk.text,
                        output_path=chunk_path,
                    )
                samples, sr = poc.read_wav(chunk_path)
                trimmed = poc.trim_edge_silence(samples, sr)
                poc.write_wav(chunk_path, trimmed, sr)
                if len(trimmed) > 0:
                    audio_chunks.append(trimmed)
                    sample_rate = sample_rate or sr
                entry["status"] = "ok" if len(trimmed) > 0 else "empty"
                entry["duration_s"] = round(len(trimmed) / sr, 3) if sr else 0.0
                entry["elapsed_s"] = elapsed
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["elapsed_s"] = 0.0
            poc.write_manifest(manifest_path, manifest)

        if audio_chunks and sample_rate is not None:
            preview = poc.concat_with_crossfade(audio_chunks, sample_rate, crossfade_ms=args.crossfade_ms)
            poc.write_wav(preview_path, preview, sample_rate)
            manifest["status"] = "completed"
            manifest["preview_duration_actual_s"] = round(len(preview) / sample_rate, 3)
            chunk_elapsed = sum(float(c.get("elapsed_s", 0.0)) for c in manifest["chunks"] if c.get("status") == "ok")
            manifest["total_chunk_elapsed_s"] = round(chunk_elapsed, 3)
            if manifest["preview_duration_actual_s"] > 0:
                manifest["chunk_rtf"] = round(chunk_elapsed / float(manifest["preview_duration_actual_s"]), 4)
            exit_code = 0
        else:
            manifest["status"] = "failed"
            manifest["error"] = "No non-empty chunks were generated."
            exit_code = 1
        poc.write_manifest(manifest_path, manifest)
        print(f"Manifest: {manifest_path}")
        print(f"Preview: {preview_path}")
        if manifest.get("chunk_rtf") is not None:
            print(f"chunk_rtf: {manifest['chunk_rtf']}")
    finally:
        if started_server and not args.keep_server:
            stop_server(server_proc)
        elif args.keep_server and started_server:
            print(f">> mtts server left running at {metal_url(args)}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
