from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule


class BilingualMapping(HardRule):
    """Check that every cue has at least one counterpart in the other language.

    A cue is "orphaned" if it has zero time overlap with any cue from the
    other language.
    """

    name = "BilingualMapping"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        if len(cues) < 2:
            return issues

        source_lang = config.source_lang
        langs = [lang for lang in cues if lang != source_lang]
        if not langs:
            return issues

        target_lang = langs[0]
        source_list = cues[source_lang]
        target_list = cues[target_lang]

        # For every source cue, check if at least one target cue overlaps.
        matched_source: set[int] = set()
        matched_target: set[int] = set()

        for si, sc in enumerate(source_list):
            for ti, tc in enumerate(target_list):
                overlap_start = max(sc.start, tc.start)
                overlap_end = min(sc.end, tc.end)
                if overlap_start < overlap_end - 0.05:
                    matched_source.add(si)
                    matched_target.add(ti)

        # Report orphans
        for si, sc in enumerate(source_list):
            if si not in matched_source:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=si + 1,
                        time=seconds_to_srt(sc.start),
                        detail=f"源语言字幕无对应目标语言字幕: '{sc.text[:30]}'",
                        fix="为这条字幕添加对应翻译",
                    )
                )

        for ti, tc in enumerate(target_list):
            if ti not in matched_target:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=ti + 1,
                        time=seconds_to_srt(tc.start),
                        detail=f"目标语言字幕无对应源语言字幕: '{tc.text[:30]}'",
                        fix="这条翻译对应的源语言字幕缺失",
                    )
                )

        return issues
