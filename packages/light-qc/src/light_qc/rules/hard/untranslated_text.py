import re

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class UntranslatedText(HardRule):
    """Detect source-language text leaking into translated cues.

    e.g. English words in a Chinese cue, or CJK characters in an
    English cue when translation is involved.
    """

    name = "UntranslatedText"

    # Latin alphabet ratio threshold for detecting mixed-language cues.
    _LATIN_RATIO_THRESHOLD = 0.5

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                text = cue.text.replace("\n", "").lower()
                if not text:
                    continue
                if cue.lang == "zh":
                    # Only flag when ≥ 4 English words > 2 chars appear.
                    # 1-2 words are likely proper nouns (JEPA, LeCun, AlexNet)
                    # that should remain untranslated.
                    alpha_words = [w for w in re.findall(r"[a-zA-Z]+", text) if len(w) > 2]
                    if len(alpha_words) >= 4:
                        issues.append(
                            QCIssue(
                                severity="error",
                                category="硬性规则",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"中文 cue 中混入英文: {', '.join(alpha_words[:4])}",
                                fix="将英文替换为中文翻译",
                            )
                        )
        return issues
