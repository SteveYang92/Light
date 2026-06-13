from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class ExitPointPrecision(SoftRule):
    """Validate cue end-time padding against the last word timestamp.

    Per subtitle.md §1.3-2 and §3.2:
      - Too little padding: reader has no eye-movement time → suggestion.
      - Too much padding (> 1.0s): redundant silence → suggestion.

    Thresholds differ by language:
      - Chinese (1-line, 20-char): faster reading, less padding needed.
      - English (2-line, 42-char): more visual content, needs more breathing room.
    """

    name = "ExitPointPrecision"
    default_severity = "suggestion"

    # ── Language-aware padding thresholds ──
    _MIN_PADDING: dict[str, float] = {"zh": 0.08, "en": 0.12}
    _MAX_PADDING = 1.0
    _DEFAULT_MIN = 0.12

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if not cue.words:
                    continue

                # Only consider words that END within or before the cue.
                # Words ending after cue.end are alignment artifacts that
                # belong to the next cue (captured by tolerance expansion).
                own_words = [w for w in cue.words if w.end <= cue.end]
                if not own_words:
                    continue
                last_word_end = max(w.end for w in own_words)
                padding = cue.end - last_word_end
                min_pad = self._MIN_PADDING.get(cue.lang, self._DEFAULT_MIN)

                if padding < min_pad:
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=(f"出点紧贴语音结束，阅读 padding 仅 {padding:.3f}s（不足 {min_pad:.2f}s）"),
                            fix="将结束时间延后 0.2-0.5s",
                        )
                    )
                elif padding > self._MAX_PADDING:
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="柔性策略",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=(f"阅读 padding {padding:.3f}s 过长（超过 {self._MAX_PADDING}s），存在冗余静默"),
                            fix="将结束时间提前以减少冗余静默",
                        )
                    )

        return issues
