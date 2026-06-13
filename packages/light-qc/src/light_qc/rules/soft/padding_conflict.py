from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class PaddingConflict(SoftRule):
    """Detect when reading padding exceeds the available speech gap to the
    next cue.

    Per subtitle.md §3.2: "如果下一条字幕很近，则压缩 padding".  If the
    natural speech gap to the next cue is already small, the current cue's
    reading padding should be compressed so that the display does not bleed
    into the start of the next utterance.
    """

    name = "PaddingConflict"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i in range(len(cue_list) - 1):
                cur, nxt = cue_list[i], cue_list[i + 1]
                if not cur.words or not nxt.words:
                    continue

                cur_last_end = max(w.end for w in cur.words)
                nxt_first_start = nxt.words[0].start
                speech_gap = nxt_first_start - cur_last_end

                if speech_gap <= 0 or speech_gap >= 0.3:
                    continue

                cur_padding = cur.end - cur_last_end
                if cur_padding <= speech_gap:
                    continue

                issues.append(
                    QCIssue(
                        severity=self.default_severity,
                        category="柔性策略",
                        rule=self.name,
                        cue_id=i + 1,
                        time=seconds_to_srt(cur.end),
                        detail=(f"说话间隙仅 {speech_gap:.3f}s，但阅读 padding 有 {cur_padding:.3f}s，可能冲突"),
                        fix=f"压缩阅读 padding 至约 {max(0.05, speech_gap * 0.5):.3f}s",
                    )
                )

        return issues
