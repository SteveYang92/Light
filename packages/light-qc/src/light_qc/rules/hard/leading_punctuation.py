from light_models import QCIssue, SubtitleCue, seconds_to_srt
from light_models.punctuation import CJK_SENTENCE_ENDS

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class LeadingPunctuation(HardRule):
    """Chinese cues should not start a line with CJK punctuation.

    Under the minimal-punctuation convention，，、。；： are stripped
    before display， so only ？ ！… can appear.  These still shouldn't
    appear at the start of a line.
    """

    name = "LeadingPunctuation"
    _CJK_PUNCT = set(CJK_SENTENCE_ENDS)

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if cue.lang != "zh":
                    continue
                for j, line in enumerate(cue.text.split("\n"), 1):
                    stripped = line.strip()
                    if stripped and stripped[0] in self._CJK_PUNCT:
                        issues.append(
                            QCIssue(
                                severity="suggestion",
                                category="硬性规则",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"第 {j} 行以标点 '{stripped[0]}' 开头",
                                fix="将标点移至上一行末尾",
                            )
                        )
        return issues
