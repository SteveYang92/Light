from light_models import QCIssue, SubtitleCue, is_cjk, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class ReadingSpeed(HardRule):
    name = "ReadingSpeed"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                duration = cue.end - cue.start
                if duration <= 0:
                    continue
                if cue.lang == "zh":
                    char_count = sum(1 for ch in cue.text if is_cjk(ch))
                    cps = char_count / duration
                    if cps > config.cps_limit:
                        issues.append(
                            QCIssue(
                                severity="error",
                                category="硬性规则",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"阅读速度 {cps:.1f} 字/秒，超过上限 {config.cps_limit} 字/秒",
                                fix="压缩文本或将字幕拆分为多条",
                            )
                        )
                else:
                    chars = len(cue.text.replace("\n", ""))
                    cps = chars / duration
                    if cps > config.cps_limit_en:
                        issues.append(
                            QCIssue(
                                severity="error",
                                category="硬性规则",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"阅读速度 {cps:.1f} 字符/秒，超过上限",
                                fix="压缩文本或将字幕拆分为多条",
                            )
                        )
        return issues
