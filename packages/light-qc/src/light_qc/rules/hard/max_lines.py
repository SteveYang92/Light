from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class MaxLines(HardRule):
    name = "MaxLines"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                lines = cue.text.split("\n")
                limit = config.max_lines_zh if cue.lang == "zh" else config.max_lines
                if len(lines) > limit:
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"字幕有 {len(lines)} 行，超过最大 {limit} 行限制",
                            fix=f"将字幕压缩为最多 {limit} 行",
                        )
                    )
        return issues
