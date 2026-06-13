from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues

# Chinese short utterances confirmed as legitimate standalone cues
# from real pipeline output.  Only includes forms actually observed.
_ZH_STANDALONE: set[str] = {
    "对",
    "嗯",
    "好",
    "是啊",
    "没错",
    "确实",
    "也许吧",
    "结果呢",
    "有争议",
    "哦对",
    "嗯 对",
    "数亿年",
    "另一个",
    "对吧？",
    "是吧？",
    "我觉得",
    "好吧",
    "是的",
}


class TinyCue(HardRule):
    """Cues with ≤ 3 visible characters are almost certainly orphans.

    Exemptions:
    - English alphabetic words like "AI", "to" are legitimate short cues.
    - Chinese conversational responses, fillers, and discourse markers
      are natural standalone utterances in dialogue.
    """

    name = "TinyCue"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                stripped = cue.text.replace("\n", "").strip()
                if len(stripped) > 3:
                    continue
                if cue.lang == "zh" and stripped in _ZH_STANDALONE:
                    continue
                if cue.lang == "en" and stripped.replace(".", "").replace("!", "").replace("?", "").isalpha():
                    continue
                if len(stripped) == 0:
                    continue
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=i + 1,
                        time=seconds_to_srt(cue.start),
                        detail=f"字幕仅 {len(stripped)} 个字符: '{stripped}'",
                        fix="合并到上一条或下一条字幕",
                    )
                )
        return issues
