from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_RUN_DIR = Path("output/Dan_Carlins_Hardcore_History_62_Supernova_in_the_East_1/.seg1")
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare official IndexTTS2 vs Metal RTF POC manifests.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Pipeline run directory.")
    parser.add_argument("--preview-duration", type=float, default=180.0, help="Preview window in seconds.")
    parser.add_argument("--cfm-steps", type=int, default=16, help="Metal CFM steps passed to indextts2_metal_poc.py.")
    parser.add_argument("--skip-official", action="store_true", help="Only read existing official manifest.")
    parser.add_argument("--skip-metal", action="store_true", help="Only read existing metal manifest.")
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=None,
        help="Override official manifest path.",
    )
    parser.add_argument(
        "--metal-manifest",
        type=Path,
        default=None,
        help="Override metal manifest path.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_script(script: str, args: list[str]) -> int:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    print(f">> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def summarize(label: str, manifest: dict) -> None:
    status = manifest.get("status", "unknown")
    rtf = manifest.get("chunk_rtf")
    elapsed = manifest.get("total_chunk_elapsed_s")
    audio_s = manifest.get("preview_duration_actual_s")
    print(f"{label}:")
    print(f"  status: {status}")
    print(f"  chunk_rtf: {rtf if rtf is not None else '-'}")
    print(f"  total_chunk_elapsed_s: {elapsed if elapsed is not None else '-'}")
    print(f"  preview_duration_actual_s: {audio_s if audio_s is not None else '-'}")
    print(f"  manifest: {manifest.get('_path', '-')}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    common = ["--run-dir", str(run_dir), "--preview-duration", str(args.preview_duration)]

    official_manifest_path = args.official_manifest or (run_dir / "tts_indextts" / "manifest.json")
    metal_manifest_path = args.metal_manifest or (run_dir / "poc_indextts2_metal" / "manifest.json")

    if not args.skip_official:
        code = run_script("indextts2_poc.py", common)
        if code != 0:
            print(f"Official POC failed with exit code {code}", file=sys.stderr)
            if not official_manifest_path.is_file():
                return code

    if not args.skip_metal:
        code = run_script(
            "indextts2_metal_poc.py",
            [*common, "--cfm-steps", str(args.cfm_steps)],
        )
        if code != 0:
            print(f"Metal POC failed with exit code {code}", file=sys.stderr)
            if not metal_manifest_path.is_file():
                return code

    official = read_manifest(official_manifest_path)
    official["_path"] = str(official_manifest_path)
    metal = read_manifest(metal_manifest_path)
    metal["_path"] = str(metal_manifest_path)

    print("")
    summarize("Official PyTorch", official)
    print("")
    summarize(f"Metal (cfm={args.cfm_steps})", metal)
    print("")

    official_rtf = official.get("chunk_rtf")
    metal_rtf = metal.get("chunk_rtf")
    if isinstance(official_rtf, (int, float)) and isinstance(metal_rtf, (int, float)) and metal_rtf > 0:
        speedup = round(float(official_rtf) / float(metal_rtf), 2)
        print(f"Speedup (official_rtf / metal_rtf): {speedup}x")
        if metal_rtf < official_rtf:
            print("Metal is faster for this run.")
        elif metal_rtf > official_rtf:
            print("Official PyTorch is faster for this run.")
        else:
            print("RTF is equal for this run.")
    else:
        print("Could not compute speedup (missing chunk_rtf in one or both manifests).")
        return 1

    if official.get("status") != "completed" or metal.get("status") != "completed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
