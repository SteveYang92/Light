from light_models import QCIssue, SubtitleCue, seconds_to_srt
from light_models.punctuation import SENTENCE_ENDS

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class ConsecutiveFragments(SoftRule):
    """Flag sequences of ≥ 5 consecutive cues without sentence-ending
    punctuation as potential over-segmentation.

    Under the minimal-punctuation convention, periods are stripped so
    most cues lack terminal punctuation.  The threshold is raised to
    reduce false positives.
    """

    name = "ConsecutiveFragments"
    default_severity = "warning"
    _TERMINAL = set(SENTENCE_ENDS)

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            # Only for English — English sentence-ending punctuation (.!?) is
            # well-defined; Chinese punctuation is stripped in Scene B output.
            if cue_list and cue_list[0].lang != "en":
                continue
            i = 0
            while i < len(cue_list):
                cue = cue_list[i]
                text = cue.text.strip()
                if text and text[-1] in self._TERMINAL:
                    i += 1
                    continue
                run_start = i
                while i < len(cue_list):
                    c = cue_list[i]
                    t = c.text.strip()
                    if t and t[-1] in self._TERMINAL:
                        break
                    i += 1
                run_len = i - run_start
                if run_len >= 5:
                    start_cue = run_start + 1
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=start_cue,
                            time=seconds_to_srt(cue_list[run_start].start),
                            detail=f"连续 {run_len} 条 cue 无句末标点，可能 segment 过度切分",
                            fix="考虑合并部分片段或检查 segment 的 gap 阈值",
                        )
                    )
        return issues
