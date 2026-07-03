from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RUN_DIR = Path("output/Dan_Carlins_Hardcore_History_62_Supernova_in_the_East_1/.seg1")
DEFAULT_OFFICIAL_ROOT = Path(".cache/indextts-official/index-tts")
TERMINAL_PUNCT = tuple("。！？!?；;…")
PAUSE_PUNCT = tuple("，,、：:" + "".join(TERMINAL_PUNCT))
EMOTION_INDEX = {
    "happy": 0,
    "angry": 1,
    "sad": 2,
    "afraid": 3,
    "disgusted": 4,
    "melancholic": 5,
    "surprised": 6,
    "calm": 7,
}


@dataclass(frozen=True)
class TextCue:
    cue_id: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    cue_ids: list[str]
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official IndexTTS2 preview POC for the Dan Carlin segment.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Pipeline run directory containing cues.json.")
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT, help="Cloned index-tts repo path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="POC output directory.")
    parser.add_argument("--checkpoints", type=Path, default=None, help="IndexTTS2 checkpoints directory.")
    parser.add_argument("--ref-audio", type=Path, default=None, help="Speaker reference WAV.")
    parser.add_argument("--preview-duration", type=float, default=180.0, help="Source timeline seconds to include.")
    parser.add_argument("--lang", default="zh", help="Cue language to synthesize.")
    parser.add_argument("--max-chars", type=int, default=160, help="Maximum characters per chunk before splitting.")
    parser.add_argument("--min-chars", type=int, default=45, help="Minimum characters before allowing soft splits.")
    parser.add_argument("--crossfade-ms", type=float, default=30.0, help="Crossfade duration between generated chunks.")
    parser.add_argument(
        "--emotion",
        default="calm",
        choices=["none", *EMOTION_INDEX.keys()],
        help="IndexTTS2 emotion vector.",
    )
    parser.add_argument("--emotion-weight", type=float, default=0.6, help="Weight for the selected emotion vector.")
    parser.add_argument("--use-fp16", action="store_true", help="Enable official FP16 inference.")
    parser.add_argument("--use-random", action="store_true", help="Enable stochastic IndexTTS2 inference.")
    parser.add_argument("--verbose", action="store_true", help="Pass verbose=True to IndexTTS2.")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a short warmup infer after model load (Mac MPS: limited benefit).",
    )
    parser.add_argument("--warmup-text", default="测试。", help="Warmup synthesis text.")
    parser.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="GPT beam width; official default is 3.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable official s2mel torch.compile (CUDA only; auto-disabled on MPS).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only build chunks and manifest; do not load IndexTTS2.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_text_cues(run_dir: Path, *, lang: str, preview_duration: float) -> tuple[list[TextCue], Path]:
    source_path = run_dir / "translations" / "raw.json"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing {source_path}. IndexTTS2 POC requires translations/raw.json "
            "(display cues.json lacks translation punctuation)."
        )
    data = read_json(source_path)
    raw = data.get("cues", []) if isinstance(data, dict) else data
    cues = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict) or str(item.get("lang", "")) != lang:
            continue
        start = float(item.get("start", 0.0))
        if start >= preview_duration:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        cues.append(
            TextCue(
                cue_id=str(item.get("cue_id") or item.get("id") or f"cue_{idx:04d}"),
                start=start,
                end=float(item.get("end", start)),
                text=text,
            )
        )
    if cues:
        return cues, source_path

    raise ValueError(f"No cues with lang={lang!r} in {source_path}")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text.strip())
    text = re.sub(r"[，,]{2,}", "，", text)
    text = text.replace("，。", "。").replace(",。", "。")
    text = re.sub(r"[，,、：:]+$", "", text)
    if text and not text.endswith(TERMINAL_PUNCT):
        text += "。"
    return text


def join_separator(current_text: str, next_text: str) -> str:
    if not current_text or current_text.endswith(PAUSE_PUNCT) or next_text.startswith(PAUSE_PUNCT):
        return ""
    return "，"


def build_chunks(cues: list[TextCue], *, max_chars: int, min_chars: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current: list[TextCue] = []
    current_text = ""

    def flush() -> None:
        nonlocal current, current_text
        if not current:
            return
        text = normalize_text(current_text)
        chunks.append(
            TextChunk(
                chunk_id=f"chunk_{len(chunks):04d}",
                cue_ids=[cue.cue_id for cue in current],
                start=current[0].start,
                end=current[-1].end,
                text=text,
            )
        )
        current = []
        current_text = ""

    for cue in cues:
        text = cue.text.strip()
        separator = join_separator(current_text, text)
        next_text = current_text + separator + text
        overflow = len(current_text) >= min_chars and len(next_text) > max_chars
        boundary = current_text.endswith(TERMINAL_PUNCT)
        if current and overflow and boundary:
            flush()
        elif current and overflow:
            flush()

        current.append(cue)
        separator = join_separator(current_text, text)
        current_text += separator + text

    flush()
    return chunks


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def trim_edge_silence(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.012,
    pad_ms: float = 20.0,
) -> np.ndarray:
    if len(samples) == 0:
        return samples
    mask = np.abs(samples) > threshold
    if not mask.any():
        return samples[:0]
    start = int(np.argmax(mask))
    end = int(len(samples) - np.argmax(mask[::-1]))
    pad = int(sample_rate * pad_ms / 1000.0)
    return samples[max(0, start - pad) : min(len(samples), end + pad)]


def concat_with_crossfade(chunks: list[np.ndarray], sample_rate: int, *, crossfade_ms: float) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    out = chunks[0]
    fade_len = max(0, int(sample_rate * crossfade_ms / 1000.0))
    for chunk in chunks[1:]:
        if fade_len <= 0 or len(out) < fade_len or len(chunk) < fade_len:
            out = np.concatenate([out, chunk])
            continue
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        blend = out[-fade_len:] * fade_out + chunk[:fade_len] * fade_in
        out = np.concatenate([out[:-fade_len], blend, chunk[fade_len:]])
    return out


def emotion_vector(name: str, weight: float) -> list[float] | None:
    if name == "none":
        return None
    vector = [0.0] * len(EMOTION_INDEX)
    vector[EMOTION_INDEX[name]] = weight
    return vector


def load_official_tts(
    official_root: Path,
    checkpoints: Path,
    *,
    use_fp16: bool,
    use_torch_compile: bool,
) -> Any:
    sys.path.insert(0, str(official_root.resolve()))
    from indextts.infer_v2 import IndexTTS2

    cfg_path = checkpoints / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing official checkpoint config: {cfg_path}")
    return IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(checkpoints),
        use_fp16=use_fp16,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_torch_compile=use_torch_compile,
    )


def build_infer_kwargs(
    args: argparse.Namespace,
    *,
    ref_audio: Path,
    text: str,
    output_path: Path,
    emo_vector: list[float] | None,
) -> dict[str, Any]:
    infer_kwargs: dict[str, Any] = {
        "spk_audio_prompt": str(ref_audio),
        "text": text,
        "output_path": str(output_path),
        "use_random": args.use_random,
        "verbose": args.verbose,
        "num_beams": args.num_beams,
    }
    if emo_vector is not None:
        infer_kwargs["emo_vector"] = emo_vector
    return infer_kwargs


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_speed_options(args: argparse.Namespace) -> argparse.Namespace:
    """torch.compile only works reliably on CUDA; MPS hits s2mel compile errors."""
    args.torch_compile_requested = args.torch_compile
    if not args.torch_compile:
        return args
    try:
        import torch
    except ImportError:
        return args
    if torch.backends.mps.is_available():
        print(">> torch.compile disabled on MPS (unsupported by official s2mel path)")
        args.torch_compile = False
    return args


def official_python(official_root: Path) -> Path | None:
    candidate = official_root / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def maybe_reexec_in_official_venv(args: argparse.Namespace) -> None:
    """IndexTTS2 deps live in the official repo's uv-managed `.venv`, not `.venv-indextts`."""
    if args.dry_run:
        return
    official_root = args.official_root.resolve()
    official_py = official_python(official_root)
    if official_py is None or Path(sys.executable).resolve() == official_py.resolve():
        return
    env = os.environ.copy()
    root = str(official_root)
    env["PYTHONPATH"] = root if not env.get("PYTHONPATH") else f"{root}{os.pathsep}{env['PYTHONPATH']}"
    os.execve(str(official_py), [str(official_py), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def official_run_hint(official_root: Path) -> str:
    script = Path(__file__).resolve()
    return (
        "Run with the official index-tts uv environment instead of `.venv-indextts`:\n"
        f"  cd {official_root} && PYTHONPATH=. uv run python {script}"
    )


def main() -> int:
    args = parse_args()
    args = resolve_speed_options(args)
    maybe_reexec_in_official_venv(args)
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "tts_indextts").resolve()
    official_root = args.official_root.resolve()
    checkpoints = (args.checkpoints or official_root / "checkpoints").resolve()
    ref_audio = (args.ref_audio or output_dir / "ref.wav").resolve()
    chunks_dir = output_dir / "chunks"
    preview_path = output_dir / "preview.wav"
    manifest_path = output_dir / "manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    cues, source_path = load_text_cues(run_dir, lang=args.lang, preview_duration=args.preview_duration)
    chunks = build_chunks(cues, max_chars=args.max_chars, min_chars=args.min_chars)
    manifest: dict[str, Any] = {
        "status": "prepared",
        "run_dir": str(run_dir),
        "source": str(source_path),
        "official_root": str(official_root),
        "checkpoints": str(checkpoints),
        "ref_audio": str(ref_audio),
        "preview_path": str(preview_path),
        "preview_duration_s": args.preview_duration,
        "lang": args.lang,
        "emotion": args.emotion,
        "speed": {
            "warmup": args.warmup,
            "warmup_text": args.warmup_text,
            "num_beams": args.num_beams,
            "torch_compile": args.torch_compile,
            "torch_compile_requested": args.torch_compile_requested,
        },
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
    write_manifest(manifest_path, manifest)

    if args.dry_run:
        manifest["status"] = "dry_run"
        write_manifest(manifest_path, manifest)
        print(f"Prepared {len(chunks)} chunks; manifest: {manifest_path}")
        return 0

    if not official_root.is_dir():
        manifest["status"] = "blocked"
        manifest["error"] = f"Official index-tts repo not found: {official_root}"
        write_manifest(manifest_path, manifest)
        print(manifest["error"], file=sys.stderr)
        return 2
    if not ref_audio.is_file():
        manifest["status"] = "blocked"
        manifest["error"] = f"Reference audio not found: {ref_audio}"
        write_manifest(manifest_path, manifest)
        print(manifest["error"], file=sys.stderr)
        return 2

    try:
        tts = load_official_tts(
            official_root,
            checkpoints,
            use_fp16=args.use_fp16,
            use_torch_compile=args.torch_compile,
        )
    except ModuleNotFoundError as exc:
        manifest["status"] = "blocked"
        manifest["error"] = f"Unable to initialize official IndexTTS2: {type(exc).__name__}: {exc}"
        manifest["hint"] = official_run_hint(official_root)
        write_manifest(manifest_path, manifest)
        print(manifest["error"], file=sys.stderr)
        print(manifest["hint"], file=sys.stderr)
        return 2
    except Exception as exc:
        manifest["status"] = "blocked"
        manifest["error"] = f"Unable to initialize official IndexTTS2: {type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        print(manifest["error"], file=sys.stderr)
        return 2

    audio_chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    emo_vector = emotion_vector(args.emotion, args.emotion_weight)

    if args.warmup:
        warmup_path = output_dir / "warmup.wav"
        warmup_started = time.perf_counter()
        print(f">> warmup: {args.warmup_text!r}")
        try:
            tts.infer(
                **build_infer_kwargs(
                    args,
                    ref_audio=ref_audio,
                    text=args.warmup_text,
                    output_path=warmup_path,
                    emo_vector=emo_vector,
                )
            )
            manifest["warmup"] = {
                "status": "ok",
                "text": args.warmup_text,
                "output_path": str(warmup_path),
                "elapsed_s": round(time.perf_counter() - warmup_started, 3),
            }
        except Exception as exc:
            manifest["warmup"] = {
                "status": "failed",
                "text": args.warmup_text,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - warmup_started, 3),
            }
            print(f"Warmup failed: {exc}", file=sys.stderr)
        write_manifest(manifest_path, manifest)

    for index, chunk in enumerate(chunks):
        chunk_path = chunks_dir / f"{chunk.chunk_id}.wav"
        started = time.perf_counter()
        entry = manifest["chunks"][index]
        try:
            tts.infer(
                **build_infer_kwargs(
                    args,
                    ref_audio=ref_audio,
                    text=chunk.text,
                    output_path=chunk_path,
                    emo_vector=emo_vector,
                )
            )
            samples, sr = read_wav(chunk_path)
            trimmed = trim_edge_silence(samples, sr)
            write_wav(chunk_path, trimmed, sr)
            if len(trimmed) > 0:
                audio_chunks.append(trimmed)
                sample_rate = sample_rate or sr
            entry["status"] = "ok" if len(trimmed) > 0 else "empty"
            entry["duration_s"] = round(len(trimmed) / sr, 3) if sr else 0.0
            entry["elapsed_s"] = round(time.perf_counter() - started, 3)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["elapsed_s"] = round(time.perf_counter() - started, 3)
        write_manifest(manifest_path, manifest)

    if audio_chunks and sample_rate is not None:
        preview = concat_with_crossfade(audio_chunks, sample_rate, crossfade_ms=args.crossfade_ms)
        write_wav(preview_path, preview, sample_rate)
        manifest["status"] = "completed"
        manifest["preview_duration_actual_s"] = round(len(preview) / sample_rate, 3)
        chunk_elapsed = sum(float(c.get("elapsed_s", 0.0)) for c in manifest["chunks"] if c.get("status") == "ok")
        manifest["total_chunk_elapsed_s"] = round(chunk_elapsed, 3)
        if manifest["preview_duration_actual_s"] > 0:
            manifest["chunk_rtf"] = round(chunk_elapsed / float(manifest["preview_duration_actual_s"]), 4)
    else:
        manifest["status"] = "failed"
        manifest["error"] = "No non-empty chunks were generated."
    write_manifest(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    print(f"Preview: {preview_path}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
