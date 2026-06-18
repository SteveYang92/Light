from light_models import QCIssue, SubtitleCue, covered_time_window, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule


class TimeAxisNotOverflow(HardRule):
    """Check that every translated cue falls within its source unit's time window.

    Requires two language entries in *cues*:
      - The source-language list (lang == source_lang) whose cues carry word
        timestamps via ``cue.words``.
      - The target-language list.
    """

    name = "TimeAxisNotOverflow"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []

        # Identify source / target language lists.
        source_cues = cues.get(config.source_lang)
        if not source_cues:
            return issues

        # Build a lookup of source-unit time bounds from word timestamps.
        unit_times: dict[str, tuple[float, float]] = {}
        for sc in source_cues:
            if sc.words and sc.unit_id:
                unit_times[sc.unit_id] = (sc.words[0].start, sc.words[-1].end)

        if not unit_times:
            return issues

        # Collect all target cues (every language that is not the source).
        target_cues: list[SubtitleCue] = []
        for lang, cue_list in cues.items():
            if lang != config.source_lang:
                target_cues.extend(cue_list)

        for i, tc in enumerate(target_cues):
            if not tc.unit_id:
                continue
            window = covered_time_window(tc, unit_times)
            if window is None:
                continue
            src_start, src_end = window

            if tc.start < src_start - 0.05:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=i + 1,
                        time=seconds_to_srt(tc.start),
                        detail=f"目标语言字幕起始时间 {seconds_to_srt(tc.start)} "
                        f"早于源语言词级起始 {seconds_to_srt(src_start)}",
                        fix="调整起始时间至源语言词级时间范围内",
                    )
                )
            if tc.end > src_end + 0.05:
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=i + 1,
                        time=seconds_to_srt(tc.end),
                        detail=f"目标语言字幕结束时间 {seconds_to_srt(tc.end)} "
                        f"晚于源语言词级结束 {seconds_to_srt(src_end)}",
                        fix="调整结束时间至源语言词级时间范围内",
                    )
                )

        return issues
