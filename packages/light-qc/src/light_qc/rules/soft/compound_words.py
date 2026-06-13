"""CompoundWords — detect multi-character words split by newlines.

Uses jieba (augmented by the wordlist) to find all multi-character tokens
in a cue, then checks if any such token is split across lines.

Since jieba does not emit tokens that span newline characters, the rule
works in two passes:

1. Segment the cue text *without* newlines to discover compound words.
2. Build a position map (clean → original) and check each discovered
   compound's span against newline positions in the original text.

Severity: error (was suggestion — real-world data confirms these are
always formatting defects, not stylistic choices).
"""

from __future__ import annotations

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues
from .wordlist import load_into_jieba


class CompoundWords(SoftRule):
    """Detect multi-character Chinese words split by newlines within a cue."""

    name = "CompoundWords"
    default_severity = "error"
    languages = {"zh"}

    def __init__(self):
        super().__init__()
        load_into_jieba()

    @staticmethod
    def _build_position_map(full_text: str) -> list[int]:
        """Build a mapping: clean_text_position → original_text_position.

        Returns a list where index i (in clean_text) maps to the
        corresponding index in full_text (skipping newlines).
        """
        pos_map = []
        for i, ch in enumerate(full_text):
            if ch == "\n":
                continue
            pos_map.append(i)
        return pos_map

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        import jieba

        issues: list[QCIssue] = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if "\n" not in cue.text:
                    continue

                full_text = cue.text
                clean_text = full_text.replace("\n", "")
                pos_map = self._build_position_map(full_text)

                # Positions of all newlines in the original text.
                nl_positions = {p for p, ch in enumerate(full_text) if ch == "\n"}

                tokens = list(jieba.tokenize(clean_text))

                for word, start_c, end_c in tokens:
                    if len(word) < 2:
                        continue
                    # Only CJK words
                    if not all("\u4e00" <= ch <= "\u9fff" for ch in word):
                        continue

                    # Map clean-text span to original-text span.
                    # end_c is exclusive, so we need end_c - 1 for the
                    # last character, then +1 for exclusive end.
                    if start_c >= len(pos_map) or end_c > len(pos_map):
                        continue
                    orig_start = pos_map[start_c]
                    orig_end = pos_map[end_c - 1] + 1

                    # Check if any newline falls inside this span.
                    if any(orig_start < nl < orig_end for nl in nl_positions):
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"复合词 '{word}' 被换行拆散",
                                fix=f"将'{word}'保持在同一行",
                            )
                        )
        return issues
