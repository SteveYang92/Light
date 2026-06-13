from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class SpeakerConsistency(SoftRule):
    """Check that each cue contains only one speaker.

    Uses word-level speaker annotations from ``cue.words``.  If words within
    a single cue come from multiple speakers, flag a warning.
    """

    name = "SpeakerConsistency"
    default_severity = "warning"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if not cue.words:
                    continue
                speakers = {w.speaker for w in cue.words if w.speaker}
                if len(speakers) > 1:
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"字幕包含多个说话人: {', '.join(sorted(speakers))}",
                            fix="将此字幕拆分为多条，每条一个说话人",
                        )
                    )
        return issues
