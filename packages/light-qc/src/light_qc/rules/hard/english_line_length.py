from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class EnglishLineLength(HardRule):
    name = "EnglishLineLength"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if cue.lang != "en":
                    continue
                for j, line in enumerate(cue.text.split("\n"), 1):
                    if len(line) > config.max_chars_per_line_en:
                        issues.append(
                            QCIssue(
                                severity="error",
                                category="硬性规则",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"第 {j} 行有 {len(line)} 个字符，超过 {config.max_chars_per_line_en} 字符上限",
                                fix=f"将该行压缩为最多 {config.max_chars_per_line_en} 个字符",
                            )
                        )
        return issues
