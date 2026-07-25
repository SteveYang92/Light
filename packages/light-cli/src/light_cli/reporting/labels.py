"""Shared stage presentation data — Chinese labels and status icons.

Single source for both Plain and Rich renderers so the two views agree on
every stage's label and icon.
"""

from __future__ import annotations

from ..steps.progress import (
    STAGE_ANNOTATE,
    STAGE_ASR,
    STAGE_COMPOSE,
    STAGE_CONTEXT,
    STAGE_CORRECT,
    STAGE_FORMAT,
    STAGE_PUNCT,
    STAGE_SEGMENT,
    STAGE_TRANSLATE,
)
from .events import STAGE_DONE, STAGE_DOWNLOAD, STAGE_MERGE, STAGE_SPLIT, StageStatus

# stage 字符串 → 中文标签（与运行输出语义一一对应）
STAGE_LABELS: dict[str, str] = {
    STAGE_DOWNLOAD: "下载",
    STAGE_SPLIT: "切分",
    STAGE_ASR: "语音转录",
    STAGE_CORRECT: "转录矫正",
    STAGE_PUNCT: "标点恢复",
    STAGE_SEGMENT: "语义断句",
    STAGE_CONTEXT: "翻译上下文",
    STAGE_COMPOSE: "规划字幕边界",
    STAGE_TRANSLATE: "翻译",
    STAGE_ANNOTATE: "注解",
    STAGE_FORMAT: "格式化",
    STAGE_MERGE: "合并",
    STAGE_DONE: "完成",
}

# status → 图标（Plain 与 Rich 共用；spinner 帧在 rich_ui 单独定义）
STATUS_ICONS: dict[StageStatus, str] = {
    StageStatus.started: "▶",
    StageStatus.progress: "▶",
    StageStatus.finished: "✓",
    StageStatus.failed: "✗",
    StageStatus.skipped: "–",
}

# 阶段进行中但尚无 started 事件时的占位图标（等待）
ICON_WAITING = "⏳"


def stage_label(stage: str) -> str:
    """Chinese label for *stage*, falling back to the raw stage name."""
    return STAGE_LABELS.get(stage, stage)
