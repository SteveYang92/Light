from light_models import QCIssue, SubtitleCue, is_cjk, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class PunctuationMidLine(SoftRule):
    """Sentence-ending punctuation should appear at the end of a line,
    not in the middle.

    e.g. "推理基础的。领域" — the 。splits what should be one phrase.
    """

    name = "PunctuationMidLine"
    default_severity = "error"
    # Only flag 。mid-line — ？！are semantically meaningful and
    # intentionally preserved by strip_chinese_punct for Scene B.
    # 。mid-line is converted to a full-width space by strip_punct,
    # so this rule only catches unprocessed 。left by the pipeline.
    _TERMINAL = set("。")

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                if cue.lang != "zh":
                    continue
                for j, line in enumerate(cue.text.split("\n"), 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    for k, ch in enumerate(stripped):
                        if ch in self._TERMINAL and k < len(stripped) - 1:
                            after = stripped[k + 1 :].strip()
                            if after and is_cjk(after[0]):
                                issues.append(
                                    QCIssue(
                                        severity=self.default_severity,
                                        category="柔性策略",
                                        rule=self.name,
                                        cue_id=i + 1,
                                        time=seconds_to_srt(cue.start),
                                        detail=f"第 {j} 行句号出现在行中 '{stripped[:20]}'",
                                        fix="将句号后的文本移至下一行，或检查是否为 LLM 误插标点",
                                    )
                                )
                                break
        return issues
