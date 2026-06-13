"""Word→Cue alignment engine.

Reads a standardized ``transcript.json`` (``light-transcript.v1``) and
assigns word-level timestamps to subtitle cues by time-overlap matching.

Reference
---------
docs/light_qc_transcript_alignment.md — full design documentation.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right

from light_models import SubtitleCue, Word, seconds_to_srt

# ═══════════════════════════════════════════════════════════════════
# Transcript loading
# ═══════════════════════════════════════════════════════════════════


def load_transcript(path: str) -> list[Word]:
    """Parse a ``light-transcript.v1`` JSON file into a flat word list.

    Only the ``words`` array is consumed; ``segments`` is ignored by
    the alignment engine but preserved for future use.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    fmt = data.get("format", "")
    if not fmt.startswith("light-transcript"):
        raise ValueError(f"Unsupported transcript format: {fmt!r}. Expected 'light-transcript.v1'.")

    words: list[Word] = []
    for w in data.get("words", []):
        words.append(
            Word(
                text=str(w["text"]),
                start=float(w["start"]),
                end=float(w["end"]),
                confidence=float(w.get("confidence", 0.0)),
                speaker=w.get("speaker"),
            )
        )
    return words


# ═══════════════════════════════════════════════════════════════════
# Alignment engine
# ═══════════════════════════════════════════════════════════════════


def align_words_to_cues(
    cues: dict[str, list[SubtitleCue]],
    words: list[Word],
    tolerance: float = 0.08,
) -> list[Word]:
    """Assign *words* to *cues* by time overlap and return uncovered words.

    Parameters
    ----------
    cues:
        Language-keyed cue lists.  Each cue's ``.words`` list is
        populated in-place.
    words:
        Flat word list, already sorted by ``start``.  Callers should
        ensure this before invoking.
    tolerance:
        Seconds of grace added to each cue's ``[start, end]`` window
        when searching for candidate words (default 0.08 s ≈ 2 frames
        at 25 fps).

    Returns
    -------
    list[Word]
        Words that could not be assigned to any cue (coverage gaps).
    """
    if not words:
        return []

    # Ensure words are sorted by start time.
    words_sorted = sorted(words, key=lambda w: w.start)
    word_starts = [w.start for w in words_sorted]

    # ── Step 1: collect candidates ──────────────────────────────
    # word_idx → list of (cue_key, cue_index, overlap_seconds)
    candidates: dict[int, list[tuple[str, int, float]]] = {i: [] for i in range(len(words_sorted))}

    for lang, cue_list in cues.items():
        for ci, cue in enumerate(cue_list):
            ws = cue.start - tolerance
            we = cue.end + tolerance
            if we <= ws:
                continue

            # Bisect to find word range overlapping [ws, we].
            lo = bisect_left(word_starts, ws)
            hi = bisect_right(word_starts, we)

            for wi in range(lo, hi):
                w = words_sorted[wi]
                overlap_start = max(w.start, ws)
                overlap_end = min(w.end, we)
                overlap = overlap_end - overlap_start
                # Zero-duration words (whisper.cpp artifact where
                # start == end) are treated as having a 1 ms duration
                # so they are not silently dropped.
                if overlap <= 0 and w.start >= ws and w.start <= we:
                    overlap = 0.001
                if overlap > 0:
                    candidates[wi].append((lang, ci, overlap))

    # ── Step 2: resolve conflicts ───────────────────────────────
    # Each word goes to the cue with:
    #   1. highest overlap *ratio* (overlap / word_duration)
    #   2. ties broken by largest absolute overlap
    #   3. remaining ties → first cue encountered

    assigned: dict[str, dict[int, list[Word]]] = {}  # lang → {cue_idx → [words]}

    for wi, cand_list in candidates.items():
        if not cand_list:
            continue

        w = words_sorted[wi]
        w_dur = max(w.end - w.start, 0.001)

        best = max(
            cand_list,
            key=lambda c: (
                c[2] / w_dur,  # overlap ratio
                c[2],  # absolute overlap
            ),
        )
        lang, ci, _ = best
        assigned.setdefault(lang, {}).setdefault(ci, []).append(w)

    # ── Step 3: write back to cues ──────────────────────────────
    uncovered: list[Word] = []

    for wi, cand_list in candidates.items():
        if not cand_list:
            uncovered.append(words_sorted[wi])

    for lang, cue_map in assigned.items():
        cue_list = cues.get(lang)
        if cue_list is None:
            continue
        for ci, ws_list in cue_map.items():
            if ci < len(cue_list):
                cue_list[ci].words = sorted(ws_list, key=lambda w: w.start)

    return uncovered


# ═══════════════════════════════════════════════════════════════════
# Coverage issue builder
# ═══════════════════════════════════════════════════════════════════


def build_coverage_issues(
    uncovered: list[Word],
    total_words: int,
    coverage_min: float = 0.95,
) -> list:
    """Generate QCIssue objects for uncovered transcript words.

    Returns an empty list when coverage meets *coverage_min*.
    """
    from light_models import QCIssue

    if not uncovered:
        return []

    coverage = 1.0 - len(uncovered) / max(total_words, 1)

    issues: list = []

    # Global coverage warning when below threshold.
    if coverage < coverage_min:
        issues.append(
            QCIssue(
                severity="suggestion",
                category="柔性策略",
                rule="TranscriptionCoverage",
                cue_id=None,
                time=None,
                detail=(
                    f"转录覆盖率仅 {coverage:.1%}"
                    f"（{len(uncovered)}/{total_words} 词未被覆盖），"
                    f"低于阈值 {coverage_min:.0%}"
                ),
                fix="检查是否存在大段漏字幕，或确认转录与字幕是否为同一内容",
            )
        )

    # Per-word issues (limit to avoid flooding the report).
    max_detail = min(len(uncovered), 30)
    for w in uncovered[:max_detail]:
        issues.append(
            QCIssue(
                severity="suggestion",
                category="硬性规则",
                rule="TranscriptionCoverage",
                cue_id=None,
                time=seconds_to_srt(w.start),
                detail=f"转录词 '{w.text}' 未被任何字幕覆盖",
                fix="检查该时间点是否存在漏字幕",
            )
        )

    if len(uncovered) > max_detail:
        issues.append(
            QCIssue(
                severity="suggestion",
                category="柔性策略",
                rule="TranscriptionCoverage",
                cue_id=None,
                time=None,
                detail=f"... 还有 {len(uncovered) - max_detail} 个未覆盖词（省略）",
                fix="",
            )
        )

    return issues
