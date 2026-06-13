from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class Overlap(HardRule):
    name = "Overlap"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i in range(len(cue_list) - 1):
                if cue_list[i + 1].start < cue_list[i].end:
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 2,
                            time=seconds_to_srt(cue_list[i + 1].start),
                            detail=f"字幕与上一条重叠（上条结束 {seconds_to_srt(cue_list[i].end)}，"
                            f"本条开始 {seconds_to_srt(cue_list[i + 1].start)}）",
                            fix="调整本条开始时间或上条结束时间",
                        )
                    )
        return issues
