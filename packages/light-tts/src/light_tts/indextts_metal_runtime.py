from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import TtsConfig
from .indextts_runtime import resolve_ref_audio_path

logger = logging.getLogger(__name__)

_server_proc: subprocess.Popen[bytes] | None = None
_server_registered = False


@dataclass(frozen=True)
class MetalPaths:
    root: Path
    bin: Path
    model_bundle: Path
    voice_store: Path


def resolve_metal_paths(config: TtsConfig) -> MetalPaths:
    root = Path(config.indextts_metal_root).expanduser().resolve()
    bin_path = Path(os.environ.get("MIT2_BIN", root / "mtts")).expanduser().resolve()
    model_bundle = Path(os.environ.get("MIT2_MODEL_BUNDLE", root / "bin")).expanduser().resolve()
    voice_store = root / "voices"
    return MetalPaths(root=root, bin=bin_path, model_bundle=model_bundle, voice_store=voice_store)


def metal_voice_cache_path(config: TtsConfig) -> Path:
    return Path(config.output_dir) / "tts" / "metal_voices.json"


def load_voice_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def save_voice_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ref_fingerprint(ref_audio: Path) -> dict[str, Any]:
    stat = ref_audio.stat()
    return {"ref_audio": str(ref_audio.resolve()), "ref_mtime_ns": stat.st_mtime_ns, "ref_size": stat.st_size}


def cache_entry_valid(entry: dict[str, Any], ref_audio: Path) -> bool:
    voice_id = str(entry.get("voice_id", ""))
    if not voice_id:
        return False
    fp = ref_fingerprint(ref_audio)
    return (
        str(entry.get("ref_audio", "")) == fp["ref_audio"]
        and int(entry.get("ref_mtime_ns", -1)) == fp["ref_mtime_ns"]
        and int(entry.get("ref_size", -1)) == fp["ref_size"]
    )


def wait_for_health(client: httpx.Client, *, timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = client.get("/health", timeout=5.0)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                return
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except (httpx.HTTPError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
        if attempt == 1 or attempt % 20 == 0:
            logger.info("Waiting for mtts health (%s) ... %s", attempt, last_error)
        time.sleep(1.0)
    raise TimeoutError(f"mtts server not healthy: {last_error}")


def _stop_server_proc() -> None:
    global _server_proc
    if _server_proc is None or _server_proc.poll() is not None:
        _server_proc = None
        return
    _server_proc.terminate()
    try:
        _server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _server_proc.kill()
        _server_proc.wait(timeout=5)
    _server_proc = None


def ensure_metal_server(config: TtsConfig, paths: MetalPaths) -> None:
    """Ensure mtts HTTP server is reachable; optionally start a local subprocess."""
    global _server_proc, _server_registered
    base_url = config.indextts_metal_url.rstrip("/")
    client = httpx.Client(base_url=base_url, timeout=5.0, trust_env=False)
    try:
        wait_for_health(client, timeout_s=3.0)
        logger.info("Using existing mtts server at %s", base_url)
        return
    except TimeoutError:
        if not config.indextts_metal_manage_server:
            raise RuntimeError(
                f"mtts server not reachable at {base_url}. "
                "Start it manually or set indextts_metal_manage_server: true in indextts.yaml"
            ) from None

    if _server_proc is not None and _server_proc.poll() is None:
        wait_for_health(client)
        return

    paths.voice_store.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MIT2_CFM_STEPS"] = str(config.indextts_metal_cfm_steps)
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    cmd = [
        str(paths.bin),
        "--server",
        "--host",
        config.indextts_metal_host,
        "--port",
        str(config.indextts_metal_port),
        "--model_bundle",
        str(paths.model_bundle),
        "--voice_store",
        str(paths.voice_store),
    ]
    logger.info("Starting mtts server: %s", " ".join(cmd))
    _server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    wait_for_health(client)

    if not _server_registered:
        atexit.register(_stop_server_proc)
        _server_registered = True


def clone_voice(client: httpx.Client, ref_audio: Path, *, name: str) -> str:
    with ref_audio.open("rb") as handle:
        resp = client.post(
            "/api/voices",
            files={"audio_sample": (ref_audio.name, handle, "audio/wav")},
            data={"name": name, "description": "light-tts dub"},
            timeout=600.0,
        )
    resp.raise_for_status()
    payload = resp.json()
    voice_id = str(payload.get("id", ""))
    if not voice_id:
        raise RuntimeError(f"Voice clone failed: {payload}")
    return voice_id


def resolve_metal_voice_id(
    config: TtsConfig,
    client: httpx.Client,
    speaker: str,
    *,
    cache: dict[str, dict[str, Any]],
) -> str:
    ref_audio = resolve_ref_audio_path(config, speaker)
    label = speaker.strip() or "__default__"
    entry = cache.get(label)
    if entry and cache_entry_valid(entry, ref_audio):
        return str(entry["voice_id"])

    voice_id = clone_voice(client, ref_audio, name=f"light-{label}")
    cache[label] = {"voice_id": voice_id, **ref_fingerprint(ref_audio)}
    save_voice_cache(metal_voice_cache_path(config), cache)
    logger.info("Cloned mtts voice for %s -> %s", label, voice_id)
    return voice_id


def synthesize_wav(client: httpx.Client, *, voice_id: str, text: str) -> bytes:
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "mtts",
            "input": text,
            "voice": {"id": voice_id},
            "response_format": "wav",
        },
        timeout=600.0,
    )
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "json" in content_type:
        raise RuntimeError(resp.text)
    return resp.content


def create_metal_client(config: TtsConfig) -> httpx.Client:
    return httpx.Client(
        base_url=config.indextts_metal_url.rstrip("/"),
        timeout=600.0,
        trust_env=False,
    )
