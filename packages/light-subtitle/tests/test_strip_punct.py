"""Tests for strip_chinese_punct — Chinese punctuation stripping."""

from __future__ import annotations

from light_models import SubtitleCue
from light_subtitle.pipeline.strip_punct import strip_chinese_punct


def _make_cue(
    cue_id: str = "c1",
    start: float = 1.0,
    end: float = 4.0,
    text: str = "",
    lang: str = "zh",
) -> SubtitleCue:
    return SubtitleCue(cue_id=cue_id, unit_id="u1", start=start, end=end, text=text, lang=lang)


# ═══════════════════════════════════════════════════════════════════
# Line-end period → removed
# ═══════════════════════════════════════════════════════════════════


def test_period_line_end_removed():
    cues = [_make_cue(text="这是一句完整的话。")]
    result = strip_chinese_punct(cues)
    assert len(result) == 1
    assert result[0].text == "这是一句完整的话"


def test_period_line_end_multi_line():
    cues = [_make_cue(text="第一行内容。\n第二行也是完整句。")]
    result = strip_chinese_punct(cues)
    # Trailing full-width space trimmed per-line, line-end 。removed.
    assert result[0].text == "第一行内容\n第二行也是完整句"


def test_period_line_end_both_lines():
    cues = [_make_cue(text="上行结束。\n下行结束。")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "上行结束\n下行结束"


# ═══════════════════════════════════════════════════════════════════
# Mid-line period → full-width space
# ═══════════════════════════════════════════════════════════════════


def test_period_mid_line_to_space():
    """。mid-line → full-width space."""
    cues = [_make_cue(text="这是第一句。这是第二句。")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "这是第一句\u3000这是第二句"


def test_period_mid_line_multi_line():
    cues = [_make_cue(text="第一句。继续第一行。\n第二行内容。")]
    result = strip_chinese_punct(cues)
    # Mid-line 。→　, line-end 。removed, trailing 　trimmed.
    assert result[0].text == "第一句\u3000继续第一行\n第二行内容"


# ═══════════════════════════════════════════════════════════════════
# Comma / 、/ ；/ ：→ full-width space
# ═══════════════════════════════════════════════════════════════════


def test_comma_to_fullwidth_space():
    cues = [_make_cue(text="第一部分，第二部分")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "第一部分\u3000第二部分"


def test_dunhao_to_fullwidth_space():
    cues = [_make_cue(text="苹果、香蕉、橘子")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "苹果\u3000香蕉\u3000橘子"


def test_semicolon_to_fullwidth_space():
    cues = [_make_cue(text="已经完成；仍需改进")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "已经完成\u3000仍需改进"


def test_colon_to_fullwidth_space():
    cues = [_make_cue(text="结论：可行")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "结论\u3000可行"


def test_mixed_mid_line_punctuation():
    """Various mid-line punctuation → full-width space."""
    cues = [_make_cue(text="首先，分析数据；然后，得出结论。最终报告。")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "首先\u3000分析数据\u3000然后\u3000得出结论\u3000最终报告"


# ═══════════════════════════════════════════════════════════════════
# ？！…→ preserved
# ═══════════════════════════════════════════════════════════════════


def test_question_mark_preserved():
    cues = [_make_cue(text="真的吗？")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "真的吗？"


def test_exclamation_preserved():
    cues = [_make_cue(text="太好了！")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "太好了！"


def test_ellipsis_preserved():
    cues = [_make_cue(text="这个嘛…")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "这个嘛…"


def test_mixed_keep_and_replace():
    cues = [_make_cue(text="第一部分，真的吗？第二部分：太好了！这个…没说完。")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "第一部分\u3000真的吗？第二部分\u3000太好了！这个…没说完"


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


def test_leading_fullwidth_trimmed():
    """Full-width space at line start is trimmed."""
    # "。开头" → "　开头" → strip → "开头"
    cues = [_make_cue(text="。开头的内容")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "开头的内容"


def test_trailing_fullwidth_trimmed():
    """Full-width space at line end is trimmed."""
    cues = [_make_cue(text="中间停顿，")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "中间停顿"


def test_consecutive_spaces_collapsed():
    """，。→ single full-width space."""
    cues = [_make_cue(text="先暂停，。然后继续")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "先暂停\u3000然后继续"


def test_all_punctuation_only():
    """Line with only punctuation → empty."""
    cues = [_make_cue(text="，。、；：")]
    result = strip_chinese_punct(cues)
    assert result[0].text == ""


def test_non_chinese_cues_untouched():
    """English cues pass through unchanged."""
    cues = [_make_cue(text="Hello, world.", lang="en")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "Hello, world."


def test_multi_cue_with_mixed_lang():
    cues = [
        _make_cue(cue_id="c1", text="第一条，中文。", lang="zh"),
        _make_cue(cue_id="c2", text="Second, English.", lang="en"),
        _make_cue(cue_id="c3", text="第三条；中文？", lang="zh"),
    ]
    result = strip_chinese_punct(cues)
    assert result[0].text == "第一条\u3000中文"
    assert result[1].text == "Second, English."
    assert result[2].text == "第三条\u3000中文？"


def test_preserve_multiline_newlines():
    """Multi-line structure is preserved after stripping."""
    cues = [_make_cue(text="第一行，内容。\n第二行；继续。")]
    result = strip_chinese_punct(cues)
    assert "\n" in result[0].text
    assert result[0].text == "第一行\u3000内容\n第二行\u3000继续"


# ═══════════════════════════════════════════════════════════════════
# Always-on — Chinese punctuation stripping is default behavior
# ═══════════════════════════════════════════════════════════════════


def test_strip_always_on_for_chinese():
    """Chinese cues always stripped — no config toggle needed."""
    cues = [_make_cue(text="第一句。第二句，")]
    result = strip_chinese_punct(cues)
    assert result[0].text == "第一句\u3000第二句"
