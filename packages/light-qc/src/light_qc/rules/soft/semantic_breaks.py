from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class SemanticBreaks(SoftRule):
    name = "SemanticBreaks"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                lines = cue.text.split("\n")
                if len(lines) == 2:
                    first_line = lines[0].strip()
                    second_line = lines[1].strip()
                    if cue.lang == "zh" and len(first_line) <= 3 and len(second_line) >= 10:
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"首行过短({first_line})，语义切分可能不自然",
                                fix="调整断行位置到语义分界处",
                            )
                        )
                    if cue.lang == "en" and len(first_line) <= 5 and len(second_line) >= 20:
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"首行过短({first_line})，语义切分可能不自然",
                                fix="调整断行位置到语义分界处",
                            )
                        )
        return issues
