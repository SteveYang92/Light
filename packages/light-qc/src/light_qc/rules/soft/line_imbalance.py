from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class LineImbalance(SoftRule):
    """Chinese two-line cues should have roughly balanced line lengths.

    A disparity > 50% suggests an unnatural line break.
    """

    name = "LineImbalance"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if cue.lang != "zh":
                    continue
                lines = cue.text.split("\n")
                if len(lines) != 2:
                    continue
                l1 = len(lines[0])
                l2 = len(lines[1])
                if l1 == 0 or l2 == 0:
                    continue
                imbalance = abs(l1 - l2) / max(l1, l2)
                if imbalance > 0.6:
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"双行长度失衡 {imbalance:.0%} (L1: {l1}字, L2: {l2}字)",
                            fix="调整断行位置，使两行长度更接近",
                        )
                    )
        return issues
