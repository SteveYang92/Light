from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, pair_bilingual


class VisualDensity(HardRule):
    """Total visual characters per second (zh + en) should not be excessive.

    Threshold: 18 chars/sec for bilingual display.
    """

    name = "VisualDensity"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        if len(cues) < 2:
            return issues

        DENSITY_LIMIT = 18

        for s_idx, sc, _t_idx, tc in pair_bilingual(cues, config.source_lang):
            overlap = min(sc.end, tc.end) - max(sc.start, tc.start)
            if overlap <= 0:
                continue

            zh_chars = len(sc.text.replace("\n", ""))
            en_chars = len(tc.text.replace("\n", ""))
            density = (zh_chars + en_chars) / overlap

            if density > DENSITY_LIMIT:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=s_idx + 1,
                        time=seconds_to_srt(sc.start),
                        detail=f"双语视觉密度 {density:.1f} 字符/秒超过 {DENSITY_LIMIT}"
                        f" (zh: {zh_chars}, en: {en_chars}, overlap: {overlap:.1f}s)",
                        fix="压缩文本或拆分字幕",
                    )
                )

        return issues
