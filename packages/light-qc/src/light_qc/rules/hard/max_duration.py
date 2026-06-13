from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class MaxDuration(HardRule):
    name = "MaxDuration"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                duration = cue.end - cue.start
                if duration > config.max_duration:
                    # Escalate to warning when duration exceeds 9s — the cue
                    # is so long it's likely to cause noticeable reading fatigue.
                    severity = "warning" if duration > 9.0 else "suggestion"
                    issues.append(
                        QCIssue(
                            severity=severity,
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"字幕时长 {duration:.2f}s，超过最大 {config.max_duration}s",
                            fix="将字幕拆分为多条",
                        )
                    )
        return issues
