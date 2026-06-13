"""BadWordSplit — detect tokens split by line-breaks or cue boundaries.

Covers three layers:

  L1  Chinese cross-cue token integrity  (jieba)
      A multi-character Chinese token should not be split across a
      cue boundary.  Intra-cue newline splits are NOT checked here —
      they are too common in subtitle formatting and produce excessive
      false positives.

  L2  Grammatical particle at line/cue start  (heuristic)
      Particles like 了,着,过,吗,呢,吧,啊 should not start a line
      or cue — they naturally attach to the preceding word.
      Note: 的 is excluded — it starts lines too commonly in
      Chinese subtitle formatting.

  L3  English word integrity  (character-level)
      A line (within a cue) or cue boundary should not split an
      English word or proper name.  Checks both:
      - Intra-cue: "Transforme\nr" (line break inside a word)
      - Cross-cue: cue N ends with "Transforme" and cue N+1 starts with "r"
      Handles Unicode letters (accented chars like é in names).

The rule loads a custom wordlist (see wordlist.py) into jieba to
improve segmentation of domain terminology.
"""

from __future__ import annotations

import re

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues
from .wordlist import load_into_jieba

# ── Constants ──────────────────────────────────────────────────

# Particles that should rarely start a line or cue.
# 的 is excluded — it starts lines too commonly in subtitle formatting.
_PARTICLES = set("了着过吗呢吧啊么嘛呗")

# Regex matching any letter including Latin-1 Supplement (for accented
# characters like é, ñ, ü in names like "Stéphane Denis").
_LETTER = re.compile(r"[a-zA-Z\u00C0-\u024F]")

# Common complete words that can legitimately end a line followed by
# another word on the next line.  When the trailing fragment matches
# one of these, skip (likely a clean line break, not a word split).
_SAFE_ENGLISH_WORDS: set[str] = {
    "the",
    "and",
    "for",
    "not",
    "are",
    "was",
    "had",
    "has",
    "but",
    "all",
    "can",
    "any",
    "its",
    "than",
    "that",
    "this",
    "with",
    "from",
    "have",
    "they",
    "been",
    "were",
    "some",
    "what",
    "when",
    "will",
    "would",
    "could",
    "about",
    "into",
    "over",
    "after",
    "before",
    "because",
    "which",
    "while",
    "where",
    "there",
    "their",
    "these",
}


class BadWordSplit(SoftRule):
    """Detect words and phrases split by line-breaks or cue boundaries."""

    name = "BadWordSplit"
    default_severity = "error"
    languages = {"zh"}

    def __init__(self):
        super().__init__()
        load_into_jieba()

    # ── Public entry point ─────────────────────────────────────

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues: list[QCIssue] = []
        for _lang, cue_list in _iter_cues(cues):
            issues.extend(self._check_chinese(cue_list))
            issues.extend(self._check_english(cue_list))
        return issues

    # ════════════════════════════════════════════════════════════
    #  Chinese  (L1 + L2)
    # ════════════════════════════════════════════════════════════

    def _check_chinese(self, cue_list: list[SubtitleCue]) -> list[QCIssue]:
        issues: list[QCIssue] = []
        import jieba

        for i, cue in enumerate(cue_list):
            # L2: particle at line start (within-cue)
            issues.extend(self._check_particle_line_start(cue, i))

        # L2: particle at cue start  +  L1: cross-cue token split
        for i in range(len(cue_list) - 1):
            a, b = cue_list[i], cue_list[i + 1]
            issues.extend(self._check_particle_cue_start(a, b, i))
            issues.extend(self._check_cross_cue_token(a, b, i, jieba))

        return issues

    # ── L1: jieba cross-cue token split ────────────────────────

    def _check_cross_cue_token(self, a: SubtitleCue, b: SubtitleCue, a_idx: int, jieba_module) -> list[QCIssue]:
        """Flag multi-char Chinese tokens split across cue boundary."""
        issues: list[QCIssue] = []
        a_last = a.text.split("\n")[-1].strip()
        b_first = b.text.split("\n")[0].strip()
        if not a_last or not b_first:
            return issues

        combined = a_last + b_first
        boundary = len(a_last)
        tokens = list(jieba_module.tokenize(combined))

        for word, start, end in tokens:
            if len(word) < 2:
                continue
            if start < boundary < end:
                issues.append(
                    QCIssue(
                        severity=self.default_severity,
                        category="柔性策略",
                        rule="BadWordSplit",
                        cue_id=a_idx + 1,
                        time=f"{seconds_to_srt(a.start)} → {seconds_to_srt(b.start)}",
                        detail=f"词 '{word}' 被跨cue切断 "
                        f"(#{a_idx + 1}末'…{a_last[-4:]}' → "
                        f"#{a_idx + 2}首'{b_first[:6]}…')",
                        fix=f"将'{word}'保留在同一个cue中",
                    )
                )
                break
        return issues

    # ── L2: particle at line / cue start ───────────────────────

    def _check_particle_line_start(self, cue: SubtitleCue, cue_idx: int) -> list[QCIssue]:
        """Flag grammatical particles starting a new line within a cue."""
        issues: list[QCIssue] = []
        lines = cue.text.split("\n")
        if len(lines) < 2:
            return issues

        import jieba

        for j in range(1, len(lines)):
            line = lines[j].strip()
            if not line:
                continue
            ch = line[0]
            if ch not in _PARTICLES:
                continue
            # Verify the char is a standalone particle, not the first
            # character of a longer word (e.g. '过' in '过程').
            tokens = list(jieba.tokenize(line))
            if tokens and len(tokens[0][0]) > 1:
                continue  # part of compound word, not a particle
            issues.append(
                QCIssue(
                    severity=self.default_severity,
                    category="柔性策略",
                    rule="BadWordSplit",
                    cue_id=cue_idx + 1,
                    time=seconds_to_srt(cue.start),
                    detail=f"第{j + 1}行以语法虚词'{ch}'开头 (上行末'…{lines[j - 1].strip()[-4:]}')",
                    fix=f"将'{ch}'合并到上一行末尾",
                )
            )
        return issues

    def _check_particle_cue_start(self, a: SubtitleCue, b: SubtitleCue, a_idx: int) -> list[QCIssue]:
        """Flag particles at the start of a new cue."""
        issues: list[QCIssue] = []
        b_first_line = b.text.split("\n")[0].strip()
        if not b_first_line:
            return issues

        ch = b_first_line[0]
        if ch not in _PARTICLES:
            return issues

        import jieba

        tokens = list(jieba.tokenize(b_first_line))
        if tokens and len(tokens[0][0]) > 1:
            return issues  # part of compound word

        a_last = a.text.split("\n")[-1].strip()
        issues.append(
            QCIssue(
                severity=self.default_severity,
                category="柔性策略",
                rule="BadWordSplit",
                cue_id=a_idx + 1,
                time=f"{seconds_to_srt(a.start)} → {seconds_to_srt(b.start)}",
                detail=f"Cue #{a_idx + 2}以虚词'{ch}'开头 (上cue末'…{a_last[-4:]}')",
                fix=f"考虑将'{ch}'合并到上一cue",
            )
        )
        return issues

    # ════════════════════════════════════════════════════════════
    #  English  (L3)
    # ════════════════════════════════════════════════════════════

    def _check_english(self, cue_list: list[SubtitleCue]) -> list[QCIssue]:
        """Detect English words / proper names split by a newline
        within a single cue or across adjacent cues."""
        issues: list[QCIssue] = []

        for i, cue in enumerate(cue_list):
            # ── Intra-cue: adjacent lines within one cue ──
            issues.extend(self._check_intra_cue_english(cue, i))

        # ── Cross-cue: last line of cue i vs first line of cue i+1 ──
        for i in range(len(cue_list) - 1):
            issues.extend(self._check_cross_cue_english(cue_list[i], cue_list[i + 1], i))

        return issues

    # ── L3a: within a single cue ────────────────────────────────

    @staticmethod
    def _check_intra_cue_english(cue: SubtitleCue, cue_idx: int) -> list[QCIssue]:
        """Flag English word split by \\n inside a single cue."""
        issues: list[QCIssue] = []
        lines = cue.text.split("\n")
        if len(lines) < 2:
            return issues

        for j in range(len(lines) - 1):
            trail = _trailing_letters(lines[j])
            lead = _leading_letters(lines[j + 1])
            if trail is None or lead is None:
                continue
            # Skip if the trailing fragment is a common function word
            # that can legitimately end a line (e.g. "with", "the", "from").
            if trail.lower() in _SAFE_ENGLISH_WORDS:
                continue
            combined = trail + lead
            if len(combined) < 4:
                continue

            issues.append(
                QCIssue(
                    severity="error",
                    category="柔性策略",
                    rule="BadWordSplit",
                    cue_id=cue_idx + 1,
                    time=seconds_to_srt(cue.start),
                    detail=(
                        f"英文词/名被换行拆散: 第{j + 1}行末'…{trail}' + 第{j + 2}行首'{lead}…' (合并为'{combined}')"
                    ),
                    fix=f"将'{combined}'保持在同一行",
                )
            )
        return issues

    # ── L3b: across adjacent cues ───────────────────────────────

    @staticmethod
    def _check_cross_cue_english(a: SubtitleCue, b: SubtitleCue, a_idx: int) -> list[QCIssue]:
        """Flag English word split across two adjacent cues."""
        issues: list[QCIssue] = []
        a_last_line = a.text.split("\n")[-1]
        b_first_line = b.text.split("\n")[0]

        trail = _trailing_letters(a_last_line)
        lead = _leading_letters(b_first_line)
        if trail is None or lead is None:
            return issues
        if trail.lower() in _SAFE_ENGLISH_WORDS:
            return issues
        combined = trail + lead
        if len(combined) < 4:
            return issues

        issues.append(
            QCIssue(
                severity="error",
                category="柔性策略",
                rule="BadWordSplit",
                cue_id=a_idx + 1,
                time=f"{seconds_to_srt(a.start)} → {seconds_to_srt(b.start)}",
                detail=(
                    f"英文词/名被跨cue切断: #{a_idx + 1}末'…{trail}' → #{a_idx + 2}首'{lead}…' (合并为'{combined}')"
                ),
                fix=f"将'{combined}'保留在同一个cue中",
            )
        )
        return issues


# ════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════


def _trailing_letters(line: str) -> str | None:
    """Extract trailing contiguous letter sequence (≥3 chars) from a line.
    Returns None if no such sequence exists."""
    stripped = line.rstrip()
    match = re.search(r"([a-zA-Z\u00C0-\u024F]{3,})$", stripped)
    return match.group(1) if match else None


def _leading_letters(line: str) -> str | None:
    """Extract leading contiguous letter sequence (≥1 char) from a line.
    Returns None if no such sequence exists."""
    stripped = line.lstrip()
    match = re.search(r"^([a-zA-Z\u00C0-\u024F]+)", stripped)
    return match.group(1) if match else None
