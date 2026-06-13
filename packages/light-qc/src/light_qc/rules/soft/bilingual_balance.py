from light_models import QCIssue, SubtitleCue, is_cjk, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, pair_bilingual


class BilingualBalance(SoftRule):
    """Check bilingual pair balance: similar line counts, no word-for-word
    over-alignment."""

    name = "BilingualBalance"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        if len(cues) < 2:
            return issues

        # Count per-unit line counts across both languages.
        for s_idx, sc, _t_idx, tc in pair_bilingual(cues, config.source_lang):
            zh_lines = len(sc.text.split("\n"))
            en_lines = len(tc.text.split("\n"))

            # 1. Line count imbalance (softer threshold than hard rule)
            imbalance = abs(zh_lines - en_lines)
            if imbalance >= 2:
                issues.append(
                    QCIssue(
                        severity=self.default_severity,
                        category="柔性策略",
                        rule=self.name,
                        cue_id=s_idx + 1,
                        time=seconds_to_srt(sc.start),
                        detail=f"中英文行数差距 {imbalance} (zh: {zh_lines}行, en: {en_lines}行)",
                        fix="调整断行位置以平衡双语行数",
                    )
                )

            # 2. Over-alignment detection: if every zh cue has a 1:1 en cue
            #    with identical line structure, flag for review.
            zh_line_words = [sum(1 for ch in line if is_cjk(ch) or ch.isalpha()) for line in sc.text.split("\n")]
            en_line_words = [len(line.split()) for line in tc.text.split("\n")]

            if zh_lines == en_lines and zh_lines >= 2:
                pairs = list(zip(zh_line_words, en_line_words, strict=False))
                same_count = sum(1 for z, e in pairs if abs(z - e) <= 1)
                if same_count == len(pairs):
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=s_idx + 1,
                            time=seconds_to_srt(sc.start),
                            detail="中英文断行结构完全一致，可能存在逐行硬对齐",
                            fix="尝试按语言习惯独立断行，而非逐行对齐",
                        )
                    )

        return issues
