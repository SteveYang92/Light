from light_models import QCIssue, SubtitleCue, seconds_to_srt
from light_models.punctuation import CJK_SENTENCE_ENDS

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class MissingPunctuation(HardRule):
    name = "MissingPunctuation"

    # Under the minimal-punctuation convention:
    # - Periods (。) are omitted — the line break is the visual separator.
    # - Commas/semicolons/colons are replaced by full-width spaces.
    # - Only ？, ！, … carry essential semantic/tone information.
    _REQUIRED = set(CJK_SENTENCE_ENDS)

    # Patterns that suggest a question intonation but lack ？.
    _QUESTION_HINTS = ("吗", "呢", "吧", "啊", "么", "嘛", "呗")

    # Patterns that suggest exclamation but lack ！.
    _EXCLAMATION_HINTS = ("太好了", "真", "多么", "太", "竟然", "居然", "简直")

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if cue.lang != "zh":
                    continue
                text = cue.text.strip()
                if not text:
                    continue

                lines = text.split("\n")
                last_line = lines[-1].strip()
                if not last_line:
                    continue

                # Only check when the line has already got one of ？！… — they are correct.
                if last_line[-1] in self._REQUIRED:
                    continue

                # Skip if next cue is tightly adjacent (shared sentence).
                if i + 1 < len(cue_list) and cue_list[i + 1].start <= cue.end + 0.8:
                    continue

                # Check for question/explanation intonation hints without matching punctuation.
                if any(q in last_line for q in self._QUESTION_HINTS):
                    issues.append(
                        QCIssue(
                            severity="suggestion",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"中文句尾可能缺少问号: '{last_line}'",
                            fix="如为疑问句，添加 ？",
                        )
                    )
                elif any(e in last_line for e in self._EXCLAMATION_HINTS):
                    issues.append(
                        QCIssue(
                            severity="suggestion",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"中文句尾可能缺少感叹号: '{last_line}'",
                            fix="如为感叹句，添加 ！",
                        )
                    )
        return issues
