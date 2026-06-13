from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class EmptyText(HardRule):
    name = "EmptyText"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if not cue.text.strip():
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail="字幕文本为空",
                            fix="删除该条字幕或填入文本",
                        )
                    )
        return issues
