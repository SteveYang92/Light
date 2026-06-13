from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class GapFlash(HardRule):
    name = "GapFlash"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i in range(len(cue_list) - 1):
                gap = cue_list[i + 1].start - cue_list[i].end
                if 0 <= gap < config.min_gap:
                    if gap == 0:
                        detail = f"字幕间隔为 0s（无间隙），建议至少 {config.min_gap}s"
                        fix = "增大间隔或使用 chaining"
                    else:
                        detail = f"字幕间隔 {gap:.2f}s 过短，可能导致闪烁"
                        fix = "增大间隔或使用 chaining 连接"
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 2,
                            time=seconds_to_srt(cue_list[i].end),
                            detail=detail,
                            fix=fix,
                        )
                    )
        return issues
