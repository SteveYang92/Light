import json
import subprocess
from pathlib import Path

from light_models import Word


def run_whisper(
    audio_path: str, model_path: str, whisper_path: str, output_dir: str, language: str = "auto"
) -> list[Word]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_base = str(output / "whisper_output")

    cmd = [
        whisper_path,
        "-m",
        model_path,
        "-f",
        audio_path,
        "-l",
        language,
        "--max-len",
        "0",
        "--word-thold",
        "0.01",
        "-ojf",
        "-of",
        output_base,
    ]
    # whisper-cli dropped the 'main' binary; if whisper_path points to 'main'
    # try the canonical name 'whisper-cli' in the same directory.
    whisper_bin = Path(whisper_path)
    if whisper_bin.name == "main":
        sibling = whisper_bin.parent / "whisper-cli"
        if sibling.exists():
            cmd[0] = str(sibling)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    json_path = output / "whisper_output.json"
    if not json_path.exists():
        json_path = Path(output_base + ".json")

    return parse_whisper_json(str(json_path))


def parse_whisper_json(json_path: str) -> list[Word]:
    with open(json_path) as f:
        data = json.load(f)

    words = []
    if "transcription" in data:
        segments = data["transcription"]
    else:
        segments = data.get("segments", data.get("result", []))

    for segment in segments:
        tokens = segment.get("tokens", segment.get("words", []))
        speaker = segment.get("speaker", None)
        for w in tokens:
            word_obj = w if isinstance(w, dict) else {"text": w, "t0": 0, "t1": 0, "p": 0.0}
            text = str(word_obj.get("text", word_obj.get("word", "")))

            if not text.strip() or text.strip().startswith("[_") or text == "":
                continue

            start = _extract_start(word_obj)
            end = _extract_end(word_obj)
            confidence = float(word_obj.get("p", word_obj.get("confidence", 0.0)))

            words.append(
                Word(
                    text=text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    speaker=speaker,
                )
            )

    return words


def _extract_start(token: dict) -> float:
    if "offsets" in token and isinstance(token["offsets"], dict):
        return float(token["offsets"].get("from", 0)) / 1000.0
    if "timestamps" in token and isinstance(token["timestamps"], dict):
        ts = token["timestamps"].get("from", "00:00:00,000")
        return _parse_timestamp(ts)
    return float(token.get("t0", token.get("start", 0)))


def _extract_end(token: dict) -> float:
    if "offsets" in token and isinstance(token["offsets"], dict):
        return float(token["offsets"].get("to", 0)) / 1000.0
    if "timestamps" in token and isinstance(token["timestamps"], dict):
        ts = token["timestamps"].get("to", "00:00:00,000")
        return _parse_timestamp(ts)
    return float(token.get("t1", token.get("end", 0)))


def _parse_timestamp(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
