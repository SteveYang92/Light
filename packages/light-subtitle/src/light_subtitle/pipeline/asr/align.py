"""wav2vec2 forced alignment for whisper word timestamps.

Reads whisper.cpp's raw JSON output and runs wav2vec2 forced
alignment on the original whisper segments directly — no
re-grouping or word-list reconstruction needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import whisperx
from light_models import Word


def run(words: list[Word], audio_path: str, language: str = "en") -> list[Word]:
    """Correct word timestamps via wav2vec2 forced alignment.

    Uses whisper.cpp's original segment boundaries from
    ``asr/asr_whisper-cpp.raw.json`` (or legacy ``whisper_output.json``)
    as alignment anchors.
    """
    if not words:
        return words

    # ── 1. Read whisper.cpp's raw segments ──
    output_dir = Path(audio_path).parent
    wcpp_json = output_dir / "asr" / "asr_whisper-cpp.raw.json"
    if not wcpp_json.exists():
        wcpp_json = output_dir / "asr" / "whisper_output.json"
    if not wcpp_json.exists():
        wcpp_json = output_dir.parent / "asr" / "asr_whisper-cpp.raw.json"
    if not wcpp_json.exists():
        wcpp_json = output_dir.parent / "asr" / "whisper_output.json"

    if wcpp_json.exists():
        with open(wcpp_json, encoding="utf-8") as f:
            wcpp = json.load(f)
        segments = []
        for seg in wcpp.get("transcription", []):
            text = seg["text"].strip()
            if not text:
                continue
            segments.append(
                {
                    "text": text,
                    "start": seg["offsets"]["from"] / 1000.0,
                    "end": seg["offsets"]["to"] / 1000.0,
                }
            )
    else:
        # Fallback: build a single segment from all words
        text = "".join(w.text for w in words).strip()
        if text and text[0] == " ":
            text = text[1:]
        segments = [{"start": words[0].start, "end": words[-1].end, "text": text}]

    # ── 2. Split long segments for better wav2vec2 alignment ──
    # CTC alignment performs best on 2-5 s segments.  Longer
    # segments cause boundary drift and dropped tails.
    MAX_SEG_S = 5.0
    segments = _split_long_segments(segments, MAX_SEG_S)

    # ── 3. Load audio & alignment model ──
    audio = whisperx.load_audio(str(Path(audio_path).resolve()) if Path(audio_path).exists() else audio_path)
    device = "cpu"
    align_model, align_metadata = whisperx.load_align_model(language_code=language, device=device)

    # ── 4. Run forced alignment ──
    result = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # ── 5. Extract aligned words ──
    aligned_words: list[Word] = []
    for aligned_seg in result.get("segments", []):
        for aw in aligned_seg.get("words", []):
            aligned_words.append(
                Word(
                    text=" " + aw.get("word", ""),
                    start=aw.get("start", 0.0),
                    end=aw.get("end", 0.0),
                    confidence=aw.get("score", 0.0),
                )
            )

    return aligned_words


# ═══════════════════════════════════════════════════════════════
# Segment splitting for wav2vec2 alignment quality
# ═══════════════════════════════════════════════════════════════


def _split_long_segments(segments: list[dict], max_dur: float) -> list[dict]:
    """Split segments longer than *max_dur* into equal-sized sub-segments.

    Text is distributed by character count; time ranges are split
    proportionally so each sub-segment has its own whisper-anchored
    boundary.  wav2vec2 CTC alignment works best on 2-5 s chunks.
    """
    result: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur <= max_dur:
            result.append(seg)
            continue

        n = max(2, int(dur / max_dur) + 1)
        text = seg["text"]
        total_chars = len(text)
        chars_per = max(1, total_chars // n)

        for i in range(n):
            frac = i / n
            next_frac = (i + 1) / n
            sub_start = seg["start"] + dur * frac
            sub_end = seg["start"] + dur * next_frac
            sub_text = text[i * chars_per : (i + 1) * chars_per if i < n - 1 else total_chars].strip()
            if sub_text:
                result.append({"text": sub_text, "start": sub_start, "end": sub_end})

    return result
