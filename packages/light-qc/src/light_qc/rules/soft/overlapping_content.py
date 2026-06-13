from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class OverlappingContent(SoftRule):
    """Detect adjacent cues with substantial content overlap.

    Uses word-level Jaccard similarity for English and character-level
    difflib for Chinese (where word boundaries are less clear).
    """

    name = "OverlappingContent"
    default_severity = "warning"

    # Only consider words ≥ 4 chars for English overlap detection.
    # This filters out short function words and single-letter variables
    # that inflate similarity on technical content.
    _MIN_WORD_LEN = 4
    _STOP_WORDS = {
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "they",
        "them",
        "their",
        "have",
        "been",
        "were",
        "some",
        "what",
        "when",
        "will",
        "would",
        "could",
        "about",
        "into",
        "over",
        "after",
        "before",
    }

    def _word_jaccard(self, a: str, b: str) -> float:
        """Jaccard similarity on significant words (≥ 4 chars, no stop words)."""
        import re as _re

        words_a = {
            w.lower()
            for w in _re.findall(r"[a-zA-Z]+", a)
            if len(w) >= self._MIN_WORD_LEN and w.lower() not in self._STOP_WORDS
        }
        words_b = {
            w.lower()
            for w in _re.findall(r"[a-zA-Z]+", b)
            if len(w) >= self._MIN_WORD_LEN and w.lower() not in self._STOP_WORDS
        }
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        import difflib

        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i in range(len(cue_list) - 1):
                a = cue_list[i].text.replace("\n", " ").strip()
                b = cue_list[i + 1].text.replace("\n", " ").strip()
                if len(a) < 6 or len(b) < 6:
                    continue

                if cue_list[i].lang == "zh":
                    # Chinese: character-level difflib.
                    a_clean = cue_list[i].text.replace("\n", "").strip()
                    b_clean = cue_list[i + 1].text.replace("\n", "").strip()
                    ratio = difflib.SequenceMatcher(None, a_clean, b_clean).ratio()
                    if 0.4 < ratio <= 0.85:
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 2,
                                time=seconds_to_srt(cue_list[i + 1].start),
                                detail=f"与上一条字幕信息重叠 (相似度 {ratio:.0%})",
                                fix="检查是否为 segment 过度切分导致的内容重复",
                            )
                        )
                else:
                    # English: word-level Jaccard.
                    jaccard = self._word_jaccard(a, b)
                    if jaccard >= 0.3:
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 2,
                                time=seconds_to_srt(cue_list[i + 1].start),
                                detail=f"与上一条字幕词汇重叠 {jaccard:.0%}",
                                fix="检查是否为 segment 过度切分导致的内容重复",
                            )
                        )
        return issues
