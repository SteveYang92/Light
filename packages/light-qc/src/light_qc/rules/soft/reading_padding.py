from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class ReadingPadding(SoftRule):
    """Check that there is adequate reading padding (0.2-0.5s) between the
    last word end and the cue end.

    Requires word timestamps via ``cue.words``.
    """

    name = "ReadingPadding"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if not cue.words:
                    continue
                own_words = [w for w in cue.words if w.end <= cue.end]
                if not own_words:
                    continue
                last_word_end = max(w.end for w in own_words)
                padding = cue.end - last_word_end

                if padding < 0.15:
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"阅读padding仅 {padding:.2f}s，不足 0.2s",
                            fix="将结束时间延后 0.2-0.5s",
                        )
                    )
                elif padding > 1.0:
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"阅读padding {padding:.2f}s 过长，超过 1.0s",
                            fix="将结束时间提前，减少冗余静默",
                        )
                    )

        return issues
