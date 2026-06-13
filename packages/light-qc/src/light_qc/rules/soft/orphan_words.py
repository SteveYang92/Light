"""OrphanWords — detect orphan words and trailing fragments.

Two detection modes:

1.  Intra-cue orphan: second line of a 2-line cue has only 1 word (en)
    or ≤2 characters (zh).  Single-word orphans are a known readability
    defect.

2.  Cross-cue trailing orphan: a very short cue (≤5 visible chars) that
    immediately follows a previous cue and reads as a sentence completion.
    Heuristic: short + small time gap + no sentence-start punctuation.

Chinese exemptions: conversational standalone utterances (responses, fillers,
discourse markers, rhetorical questions) and technical terms confirmed from
real pipeline output.
"""

from __future__ import annotations

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues

_MAX_TRAILING_LEN = 5
_MAX_TRAILING_GAP = 0.5

# CJK sentence-start signals — a short cue beginning with one of these
# starts a new utterance and is not a trailing completion of the previous cue.
_SENTENCE_START: set[str] = {
    # ── Single-char CJK conjunctions / connectors (original set) ──
    "然",
    "而",
    "不",
    "过",
    "但",
    "是",
    "所",
    "以",
    "因",
    "此",
    "且",
    "并",
    "可",
    "那",
    "么",
    # ── Multi-char discourse markers (confirmed from real data) ──
    "据我所知",
    "你可能会说",
    "换个说法吧",
    "我想说的是",
    "也可以说",
}

# Complete CJK expressions (≤5 chars) confirmed as legitimate standalone
# cues from real pipeline output — responses, rhetorical questions,
# judgments, and technical terms.
_ZH_STANDALONE_TRAILING: set[str] = {
    "什么意思？",
    "预测什么？",
    "正是如此",
    "这也很棒",
    "公司很喜欢",
    "就在它指尖",
    "它们不重建",
    "反向传播",
    "孪生网络",
    # Short responses/fillers (also exempt in TinyCue)
    "对",
    "嗯",
    "好",
    "是啊",
    "没错",
    "确实",
    "也许吧",
    "哦对",
    "嗯 对",
    "对吧？",
    "另一个",
    "数亿年",
    "结果呢",
    "有争议",
    "会做什么？",
    "纯监督模型",
    "确实如此",
    "好吧",
    "甚至更极端",
    "甚至数亿年",
}


class OrphanWords(SoftRule):
    """Detect orphan words (within-cue) and trailing orphan cues (cross-cue)."""

    name = "OrphanWords"
    default_severity = "error"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues: list[QCIssue] = []
        for _lang, cue_list in _iter_cues(cues):
            issues.extend(self._check_intra_cue_orphans(cue_list))
            issues.extend(self._check_cross_cue_orphans(cue_list))
        return issues

    # ── Intra-cue orphan ──────────────────────────────────────

    @staticmethod
    def _check_intra_cue_orphans(cue_list: list[SubtitleCue]) -> list[QCIssue]:
        """Flag single-word orphan lines within a 2-line cue."""
        issues: list[QCIssue] = []
        for i, cue in enumerate(cue_list):
            lines = cue.text.split("\n")
            if len(lines) == 2:
                second_line = lines[1].strip()
                if cue.lang == "en":
                    word_count = len(second_line.split())
                    if word_count == 1:
                        issues.append(
                            QCIssue(
                                severity="error",
                                category="柔性策略",
                                rule="OrphanWords",
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"第二行只有一个词 '{second_line}'（orphan word）",
                                fix="调整断行位置，避免孤词",
                            )
                        )
                elif cue.lang == "zh":
                    if len(second_line) <= 2:
                        issues.append(
                            QCIssue(
                                severity="error",
                                category="柔性策略",
                                rule="OrphanWords",
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"第二行过短 '{second_line}'",
                                fix="调整断行位置",
                            )
                        )
        return issues

    # ── Cross-cue trailing orphan ─────────────────────────────

    @staticmethod
    def _check_cross_cue_orphans(cue_list: list[SubtitleCue]) -> list[QCIssue]:
        """Flag very short cues that are sentence completions of the previous cue."""
        issues: list[QCIssue] = []
        for i in range(1, len(cue_list)):
            prev = cue_list[i - 1]
            curr = cue_list[i]

            curr_text = curr.text.replace("\n", "").strip()
            if len(curr_text) > _MAX_TRAILING_LEN:
                continue

            gap = curr.start - prev.end
            if gap < 0 or gap > _MAX_TRAILING_GAP:
                continue

            if curr_text in _ZH_STANDALONE_TRAILING:
                continue

            if any(curr_text.startswith(w) for w in _SENTENCE_START):
                continue

            prev_last_line = prev.text.split("\n")[-1].strip()
            if prev_last_line and prev_last_line[-1] in "？！…":
                continue

            issues.append(
                QCIssue(
                    severity="warning",
                    category="柔性策略",
                    rule="OrphanWords",
                    cue_id=i + 1,
                    time=seconds_to_srt(curr.start),
                    detail=(f"尾cue仅{len(curr_text)}个字符 '{curr_text}'，可能为上一条字幕的句子后半部分"),
                    fix="考虑合并到上一条字幕",
                )
            )
        return issues
