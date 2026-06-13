from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class MinDuration(HardRule):
    name = "MinDuration"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                duration = cue.end - cue.start
                if duration < config.min_duration - 0.01:
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"字幕时长 {duration:.2f}s，低于最小 {config.min_duration}s",
                            fix="适当延长出点时间",
                        )
                    )
        return issues
