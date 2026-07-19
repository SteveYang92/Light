"""SRT/VTT parsing and writing for the merge step (timestamp shifting)."""

from __future__ import annotations

import re
from pathlib import Path

from light_models import seconds_to_srt

# ── Time utilities ──────────────────────────────────────
#
# Formatting converged on light_models' timecode helpers.  The parsers stay
# local: ``_srt_to_seconds`` must accept BOTH separators (SRT ``00:00:01,500``
# and VTT ``00:00:01.500``) while light_models' ``srt_to_seconds`` only splits
# on the comma, and light_models has no ASS-timestamp parser.

_SRT_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_ASS_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[.](\d+)")

_EPS = 0.5  # tolerance for split-point boundary filtering (seconds)


def _srt_to_seconds(ts: str) -> float:
    m = _SRT_TIME_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid timestamp: {ts}")
    h, mm, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mm * 60 + s + ms / 1000


def _seconds_to_vtt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _ass_to_seconds(ts: str) -> float:
    m = _ASS_TIME_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid ASS timestamp: {ts}")
    h, mm, s, cs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mm * 60 + s + cs / 100


# ── SRT ─────────────────────────────────────────────────


def _parse_srt(path: Path) -> list[tuple[float, float, str]]:
    if not path.exists():
        return []
    cues: list[tuple[float, float, str]] = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Find the timestamp line
        ts_idx = 0 if "-->" in lines[0] else 1
        if ts_idx >= len(lines):
            continue
        m = re.match(r"(.+?)\s*-->\s*(.+)", lines[ts_idx])
        if not m:
            continue
        try:
            start = _srt_to_seconds(m.group(1))
            end = _srt_to_seconds(m.group(2))
        except ValueError:
            continue
        text_lines = lines[ts_idx + 1 :]
        text = "\n".join(text_lines).strip()
        if text:
            cues.append((start, end, text))
    return cues


def _write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(cues, 1):
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt(start)} --> {seconds_to_srt(end)}\n")
            f.write(f"{text}\n\n")


# ── VTT ─────────────────────────────────────────────────


def _parse_vtt(path: Path) -> list[tuple[float, float, str, str]]:
    """Return (start, end, text, settings)."""
    if not path.exists():
        return []
    cues: list[tuple[float, float, str, str]] = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or lines[0].strip() == "WEBVTT":
            continue
        ts_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_idx = i
                break
        if ts_idx < 0:
            continue
        m = re.match(r"(.+?)\s*-->\s*(\S+)(.*)", lines[ts_idx])
        if not m:
            continue
        settings = m.group(3).strip()
        try:
            start = _srt_to_seconds(m.group(1).strip())
            end = _srt_to_seconds(m.group(2).strip())
        except ValueError:
            continue
        text = "\n".join(lines[ts_idx + 1 :]).strip()
        if text:
            cues.append((start, end, text, settings))
    return cues


def _write_vtt(cues: list[tuple[float, float, str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (start, end, text, settings) in enumerate(cues, 1):
            ts = f"{_seconds_to_vtt(start)} --> {_seconds_to_vtt(end)}"
            if settings:
                ts += " " + settings
            f.write(f"{i}\n{ts}\n{text}\n\n")
