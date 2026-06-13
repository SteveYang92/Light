import difflib

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class DuplicateTranslation(HardRule):
    """Detect adjacent cues with near-identical text.

    When the LLM produces two translations for the same semantic unit,
    they often end up as adjacent cues with very similar content.
    """

    name = "DuplicateTranslation"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i in range(len(cue_list) - 1):
                a = cue_list[i].text.replace("\n", "").strip()
                b = cue_list[i + 1].text.replace("\n", "").strip()
                if len(a) < 4 or len(b) < 4:
                    continue
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                if ratio > 0.85:
                    issues.append(
                        QCIssue(
                            severity="error",
                            category="硬性规则",
                            rule=self.name,
                            cue_id=i + 2,
                            time=seconds_to_srt(cue_list[i + 1].start),
                            detail=f"与上一条字幕相似度 {ratio:.0%}，可能为重复翻译",
                            fix="删除重复的翻译，或合并为一条",
                        )
                    )
        return issues
