from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class EntryPointAccuracy(HardRule):
    """Check that each cue's start time is within tolerance of its first word.

    Per subtitle.md §1.3-1 and §3.2: entry point should stay close to
    the first spoken word.  Thresholds differ by language because:
      - Chinese single-line cues display longer → larger offsets are
        visually imperceptible.
      - English alignment from whisper/wav2vec2 is typically tighter.
    """

    name = "EntryPointAccuracy"
    default_severity = "suggestion"

    # ── Language-aware thresholds ──
    # (suggestion_threshold, warning_threshold) in seconds.
    _THRESHOLDS: dict[str, tuple[float, float]] = {
        "zh": (0.30, 0.50),
        "en": (0.15, 0.30),
    }
    _DEFAULT_THRESHOLDS = (0.15, 0.30)

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []

        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if not cue.words:
                    continue

                first_word = cue.words[0]
                offset = abs(cue.start - first_word.start)

                sugg, warn = self._THRESHOLDS.get(cue.lang, self._DEFAULT_THRESHOLDS)

                if offset >= warn:
                    issues.append(
                        QCIssue(
                            severity="warning",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"入点偏离首个词语 {offset:.3f}s，超过 {warn:.2f}s 警告阈值",
                            fix=f"将入点对齐至 {seconds_to_srt(first_word.start)}",
                        )
                    )
                elif offset >= sugg:
                    issues.append(
                        QCIssue(
                            severity="suggestion",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"入点偏离首个词语 {offset:.3f}s，超过 {sugg:.2f}s 建议阈值",
                            fix=f"将入点对齐至 {seconds_to_srt(first_word.start)}",
                        )
                    )

        return issues
