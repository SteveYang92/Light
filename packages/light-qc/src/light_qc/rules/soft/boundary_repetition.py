from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class BoundaryRepetition(SoftRule):
    """Detect words repeated across adjacent cue boundaries.

    When segment splits a sentence mid-clause, consecutive cues may
    repeat the same word at the boundary (e.g. cue A ends with "如果"
    and cue B starts with "如果").
    """

    name = "BoundaryRepetition"
    default_severity = "warning"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i in range(len(cue_list) - 1):
                a = cue_list[i].text.replace("\n", "").strip()
                b = cue_list[i + 1].text.replace("\n", "").strip()
                if not a or not b:
                    continue
                for n in [2, 3]:
                    if len(a) < n or len(b) < n:
                        continue
                    a_end = a[-n:]
                    b_start = b[:n]
                    if a_end == b_start and any("\u4e00" <= c <= "\u9fff" for c in a_end):
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 2,
                                time=seconds_to_srt(cue_list[i + 1].start),
                                detail=f"边界重复: '{a_end}' 出现在上一条末尾和本条开头",
                                fix="合并或调整拆分边界，消除重复词",
                            )
                        )
                        break
        return issues
