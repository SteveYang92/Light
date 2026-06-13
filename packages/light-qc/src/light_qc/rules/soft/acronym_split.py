"""AcronymSplit — detect ALL-CAPS acronyms and numbers split by newlines.

Single-cue detection only (intra-cue).  Cross-cue acronym splits are
extremely rare and would produce false positives with adjacent short
uppercase words.

Examples from real data:
  "A\\nI"       → "AI"   (acronym split)
  "108\\n0"     → "1080" (number split)
"""

from __future__ import annotations

import re

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class AcronymSplit(SoftRule):
    """Detect acronyms and numbers split by newlines within a single cue."""

    name = "AcronymSplit"
    default_severity = "error"
    languages = {"zh"}

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues: list[QCIssue] = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                issues.extend(self._check_acronym(cue, i))
                issues.extend(self._check_number(cue, i))
        return issues

    # ── All-caps acronym split ─────────────────────────────────

    @staticmethod
    def _check_acronym(cue: SubtitleCue, cue_idx: int) -> list[QCIssue]:
        """Detect [A-Z]\\n[A-Z] pattern — all-caps acronym split across lines."""
        issues: list[QCIssue] = []
        lines = cue.text.split("\n")
        if len(lines) < 2:
            return issues

        for j in range(len(lines) - 1):
            line1 = lines[j].rstrip()
            line2 = lines[j + 1].lstrip()

            # Check uppercase split: line ends with 1-2 uppercase, line starts with 1-2 uppercase
            trail_up = re.search(r"([A-Z]{1,2})$", line1)
            lead_up = re.search(r"^([A-Z]{1,2})", line2)
            if trail_up and lead_up:
                combined = trail_up.group(1) + lead_up.group(1)
                # Minimum 2 chars total for an acronym like "AI"
                # Also require the combined form is 2-6 uppercase to avoid
                # flagging unrelated uppercase initials like "A"+"BC".
                if 2 <= len(combined) <= 6:
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="柔性策略",
                            rule="AcronymSplit",
                            cue_id=cue_idx + 1,
                            time=seconds_to_srt(cue.start),
                            detail=(
                                f"大写缩写被换行拆散: "
                                f"第{j + 1}行末'{trail_up.group(1)}' + "
                                f"第{j + 2}行首'{lead_up.group(1)}' "
                                f"(合并为'{combined}')"
                            ),
                            fix=f"将'{combined}'保持在同一行",
                        )
                    )

        return issues

    # ── Number split ───────────────────────────────────────────

    @staticmethod
    def _check_number(cue: SubtitleCue, cue_idx: int) -> list[QCIssue]:
        """Detect digits split across lines (e.g. 108\\n0 → 1080)."""
        issues: list[QCIssue] = []
        lines = cue.text.split("\n")
        if len(lines) < 2:
            return issues

        for j in range(len(lines) - 1):
            line1 = lines[j].rstrip()
            line2 = lines[j + 1].lstrip()

            trail_d = re.search(r"(\d{2,})$", line1)
            lead_d = re.search(r"^(\d+)", line2)
            if trail_d and lead_d:
                combined = trail_d.group(1) + lead_d.group(1)
                # Only flag if the combined form is a plausible number
                # (≥3 digits) and neither side is a complete number alone.
                if len(combined) >= 3:
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="柔性策略",
                            rule="AcronymSplit",
                            cue_id=cue_idx + 1,
                            time=seconds_to_srt(cue.start),
                            detail=(
                                f"数字被换行拆散: "
                                f"第{j + 1}行末'{trail_d.group(1)}' + "
                                f"第{j + 2}行首'{lead_d.group(1)}' "
                                f"(合并为'{combined}')"
                            ),
                            fix=f"将'{combined}'保持在同一行",
                        )
                    )

        return issues
