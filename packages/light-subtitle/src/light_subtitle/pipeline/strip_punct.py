"""Strip Chinese punctuation — post-formatting → pre-export.

Converts conventional CJK punctuation to the minimal-punctuation convention
used for on-screen display.  Must run AFTER layout/pace/polish so that
break-point signals (punctuation) are preserved for the entire formatting
pipeline.

Rules
-----
+-------+---------------+-------------------------------------------+
| 字符   | 位置           | 处理                                      |
+=======+===============+===========================================+
| ，、；： | 任意           | → 全角空格（　）                            |
+-------+---------------+-------------------------------------------+
| 。     | 行末           | → 移除（换行就是视觉句号）                    |
+-------+---------------+-------------------------------------------+
| 。     | 行中           | → 全角空格（断句引擎未换行的句子边界）         |
+-------+---------------+-------------------------------------------+
| ？！…   | 任意           | → 保留（疑问/感叹/省略语义不可省略）           |
+-------+---------------+-------------------------------------------+

Edge cases
----------
- Leading/trailing full-width spaces are trimmed per line.
- Consecutive full-width spaces are collapsed to one.

Usage::
    from .strip_punct import strip_chinese_punct
    cues = strip_chinese_punct(cues, config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from light_models import SubtitleCue

    from light_subtitle.config import SubtitleConfig

_FULLWIDTH_SPACE = " "

# Punctuation replaced with full-width space (pause semantics).
_REPLACE_WITH_SPACE: set[str] = {"，", "、", "；", "："}

# 。is special — line-end → remove, mid-line → full-width space.
_PERIOD = "。"

# Punctuation preserved (irreplaceable semantic/tone information).
_KEEP: set[str] = {"？", "！", "…"}


def strip_chinese_punct(
    cues: list[SubtitleCue],
    config: SubtitleConfig | None = None,
) -> list[SubtitleCue]:
    """Strip/replace Chinese punctuation for on-screen display.

    Must be called AFTER layout/pace/polish — punctuation is needed
    as break-point signal throughout the formatting pipeline.

    Chinese subtitle convention: screen space is precious.  The line
    break itself serves as the visual period; full-width spaces（　）
    replace commas and other pause marks.  Only ？！… are preserved
    for their irreplaceable semantic/tone information.
    """
    for cue in cues:
        if cue.lang != "zh":
            continue
        lines = cue.text.split("\n")
        new_lines: list[str] = []
        for line in lines:
            new_lines.append(_strip_line(line))
        cue.text = "\n".join(new_lines)

    return cues


def _strip_line(line: str) -> str:
    """Strip punctuation from a single line and return the cleaned version."""
    result: list[str] = []
    for i, ch in enumerate(line):
        is_line_end = i == len(line) - 1

        if ch == _PERIOD and is_line_end:
            # 行末句号 → 丢弃（换行即句号）
            continue
        elif ch == _PERIOD:
            # 行中句号 → 全角空格
            _append_space(result)
        elif ch in _REPLACE_WITH_SPACE:
            # ，、；：→ 全角空格
            _append_space(result)
        elif ch in _KEEP:
            # ？！…→ 保留
            result.append(ch)
        else:
            result.append(ch)

    cleaned = "".join(result).strip(_FULLWIDTH_SPACE)
    return cleaned


def _append_space(result: list[str]) -> None:
    """Append a full-width space, avoiding consecutive duplicates."""
    if not result or result[-1] != _FULLWIDTH_SPACE:
        result.append(_FULLWIDTH_SPACE)
