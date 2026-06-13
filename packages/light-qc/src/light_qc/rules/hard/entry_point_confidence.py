"""Detect low-confidence word alignment at cue entry points.

When wav2vec2 alignment produces low-confidence timestamps for the first
word(s) of a cue (often at segment boundaries where acoustic features are
ambiguous), the entry point may be unreliable — the subtitle can appear
too early (before the speaker actually begins).

This rule complements EntryPointAccuracy: the latter checks the offset
between the SRT start and the aligned first-word timestamp; this rule
checks whether that aligned timestamp itself is trustworthy.

Signals
-------
The primary signal is a **confidence cliff**: the first word has markedly
lower confidence than later words in the same cue.  Two secondary signals
(stretched first word, large gap to second word) are only used to
reinforce when combined with a confidence cliff — they are too noisy on
their own (natural pauses, slow speech).

Severity
--------
- **WARNING**: confidence cliff with diff ≥ 0.50, *or* cliff (≥ 0.30)
  combined with severe stretch (≥ 800 ms) or large gap (≥ 0.50 s).
- **SUGGESTION**: confidence cliff with diff ≥ 0.30 but no secondary support.
"""

from __future__ import annotations

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class EntryPointConfidence(HardRule):
    """Flag cues where first-word alignment confidence suggests an unreliable entry point."""

    name = "EntryPointConfidence"
    default_severity = "suggestion"

    # ── Primary signal: confidence cliff ──────────────────────
    CLIFF_FIRST_MAX = 0.40   # first word must be below this
    CLIFF_PEAK_MIN = 0.70    # later words must reach at least this
    CLIFF_DIFF_MIN = 0.30    # minimum diff for SUGGESTION
    CLIFF_DIFF_SEVERE = 0.50  # diff this large → WARNING even alone

    # ── Secondary signals (only meaningful with a cliff) ─────
    STRETCH_MIN_MS = 800      # flag only when > 800 ms
    GAP_MIN_S = 0.50          # flag only when > 0.50 s

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues: list[QCIssue] = []

        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                words = cue.words
                if not words or len(words) < 2:
                    continue

                first = words[0]
                first_conf = first.confidence

                later_confs = [w.confidence for w in words[1:6]]
                peak_later = max(later_confs) if later_confs else first_conf
                conf_diff = peak_later - first_conf

                has_cliff = (
                    first_conf < self.CLIFF_FIRST_MAX
                    and peak_later >= self.CLIFF_PEAK_MIN
                    and conf_diff >= self.CLIFF_DIFF_MIN
                )
                if not has_cliff:
                    continue

                first_dur_ms = (first.end - first.start) * 1000
                first_text_len = len(first.text.strip())
                is_stretched = first_text_len <= 5 and first_dur_ms > self.STRETCH_MIN_MS

                second = words[1]
                gap = second.start - first.end
                has_large_gap = gap > self.GAP_MIN_S

                detail_parts = [
                    f"入点首个词 '{first.text}' 置信度 {first_conf:.2f}，"
                    f"后续词峰值 {peak_later:.2f}（差 {conf_diff:.2f}）"
                ]
                fix_hints = ["对齐可能偏早，建议根据音频确认入点"]

                if is_stretched:
                    detail_parts.append(
                        f"首个词时长 {first_dur_ms:.0f}ms（{first_text_len}字符），"
                        f"可能覆盖了前置静默"
                    )
                    fix_hints.append("首个词可能覆盖了前置静默，建议推迟入点")

                if has_large_gap:
                    detail_parts.append(f"前两个词 '{first.text}' → '{second.text}' 间隔 {gap:.3f}s")
                    fix_hints.append("入点词间存在大段静默，入点可能偏早")

                severe = (
                    conf_diff >= self.CLIFF_DIFF_SEVERE
                    or (is_stretched and conf_diff >= self.CLIFF_DIFF_MIN)
                    or (has_large_gap and conf_diff >= self.CLIFF_DIFF_MIN)
                )
                severity = "warning" if severe else "suggestion"

                issues.append(
                    QCIssue(
                        severity=severity,
                        category="硬性规则",
                        rule=self.name,
                        cue_id=i + 1,
                        time=seconds_to_srt(cue.start),
                        detail="；".join(detail_parts),
                        fix="; ".join(fix_hints),
                    )
                )

        return issues
