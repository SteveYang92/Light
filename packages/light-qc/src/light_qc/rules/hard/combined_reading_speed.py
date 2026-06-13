from light_models import QCIssue, SubtitleCue, is_cjk, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, pair_bilingual


class CombinedReadingSpeed(HardRule):
    """When zh + en overlap, each language must stay within its CPS limit."""

    name = "CombinedReadingSpeed"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        if len(cues) < 2:
            return issues

        for s_idx, sc, t_idx, tc in pair_bilingual(cues, config.source_lang):
            overlap = min(sc.end, tc.end) - max(sc.start, tc.start)
            if overlap <= 0:
                continue

            zh_cjk = sum(1 for ch in sc.text if is_cjk(ch))
            zh_cps = zh_cjk / overlap
            en_chars = len(tc.text.replace("\n", ""))
            en_cps = en_chars / overlap

            if zh_cps > config.cps_limit:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=s_idx + 1,
                        time=seconds_to_srt(sc.start),
                        detail=f"源语言阅读速度 {zh_cps:.1f} 字/秒超过上限 {config.cps_limit}",
                        fix="压缩源语言文本",
                    )
                )
            if en_cps > config.cps_limit_en:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=t_idx + 1,
                        time=seconds_to_srt(tc.start),
                        detail=f"目标语言阅读速度 {en_cps:.1f} 字符/秒超过上限 {config.cps_limit_en}",
                        fix="压缩目标语言文本",
                    )
                )

        return issues
