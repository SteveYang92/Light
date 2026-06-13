import re

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class CJKSpaceAnomaly(SoftRule):
    """Detect accidental multiple half-width spaces between CJK characters.

    Under the minimal-punctuation convention, single half-width spaces
    between CJK characters are intentional pause markers (replacing
    commas, periods, etc.).  Only 2+ consecutive spaces are likely
    accidents (e.g. double-space from LLM output, formatting artifact).
    """

    name = "CJKSpaceAnomaly"
    default_severity = "error"
    # Match 2+ consecutive half-width spaces (U+0020) between CJK chars.
    # Single spaces are intentional pipeline output.
    _CJK_SPACE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf] {2,}[\u4e00-\u9fff\u3400-\u4dbf]")

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if cue.lang != "zh":
                    continue
                if self._CJK_SPACE_RE.search(cue.text):
                    m = self._CJK_SPACE_RE.search(cue.text)
                    issues.append(
                        QCIssue(
                            severity=self.default_severity,
                            category="柔性策略",
                            rule=self.name,
                            cue_id=i + 1,
                            time=seconds_to_srt(cue.start),
                            detail=f"CJK 字符间有多余空格: '{m.group()[:20]}'",
                            fix="移除 CJK 字符间的不必要空格",
                        )
                    )
        return issues
