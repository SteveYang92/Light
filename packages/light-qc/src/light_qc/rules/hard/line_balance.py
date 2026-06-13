from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, pair_bilingual


class LineBalance(HardRule):
    """Bilingual line count imbalance: |zh_lines - en_lines| > 2 is an error."""

    name = "LineBalance"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        if len(cues) < 2:
            return issues

        MAX_IMBALANCE = 2

        for s_idx, sc, _t_idx, tc in pair_bilingual(cues, config.source_lang):
            zh_lines = len(sc.text.split("\n"))
            en_lines = len(tc.text.split("\n"))
            imbalance = abs(zh_lines - en_lines)

            if imbalance > MAX_IMBALANCE:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=s_idx + 1,
                        time=seconds_to_srt(sc.start),
                        detail=f"中英文行数差距 {imbalance} (zh: {zh_lines}行, en: {en_lines}行)",
                        fix="调整断行以使中英行数接近",
                    )
                )

        return issues
