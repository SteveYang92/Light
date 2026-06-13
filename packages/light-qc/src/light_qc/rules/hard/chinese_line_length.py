from light_models import QCIssue, SubtitleCue, is_cjk, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class ChineseLineLength(HardRule):
    name = "ChineseLineLength"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if cue.lang != "zh":
                    continue
                for j, line in enumerate(cue.text.split("\n"), 1):
                    char_count = sum(1 for ch in line if is_cjk(ch))
                    if char_count > config.max_chars_per_line_zh:
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="硬性规则",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"第 {j} 行有 {char_count} 个汉字，超过 {config.max_chars_per_line_zh} 字上限",
                                fix=f"将该行压缩为最多 {config.max_chars_per_line_zh} 个汉字",
                            )
                        )
        return issues
