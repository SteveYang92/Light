"""Tests for CPS compression (subtitle/compress.py + pace hook)."""

from __future__ import annotations

from unittest.mock import patch

from light_models import SubtitleCue, Word
from light_subtitle.config import LayoutConfig
from light_subtitle.subtitle import pace
from light_subtitle.subtitle.compress import _count_chars, _parse_results, compress_over_cps


def _config(**kwargs) -> LayoutConfig:
    return LayoutConfig(target_lang="zh", cps_limit=9, **kwargs)


def _cue(text: str, start: float, end: float, lang: str = "zh") -> SubtitleCue:
    return SubtitleCue(
        cue_id="c0",
        unit_id="p0000",
        start=start,
        end=end,
        text=text,
        lang=lang,
        words=[Word(text="src", start=start, end=end, confidence=1.0)],
    )


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


# ── parsing / counting ────────────────────────────────────


def test_parse_results_accepts_valid_and_skips_bad_items() -> None:
    raw = '{"results": [{"id": 0, "text": "短文本"}, {"id": "x", "text": "skip"}, {"id": 2, "text": "  "}]}'
    assert _parse_results(raw) == {0: "短文本"}
    assert _parse_results("not json") is None
    assert _parse_results('{"results": "nope"}') is None


def test_count_chars_excludes_newlines() -> None:
    assert _count_chars("两行\n字幕") == 4


# ── compress_over_cps ─────────────────────────────────────


def test_compress_applies_verified_texts() -> None:
    cues = [_cue("这是一段需要被压缩的超长中文字幕文本", 0.0, 2.0)]  # 18 chars / 2s = 9+ cps
    client = _FakeClient(['{"results": [{"id": 0, "text": "压缩后的字幕"}]}'])
    usage = compress_over_cps(cues, [0], _config(), llm=client)
    assert cues[0].text == "压缩后的字幕"
    assert usage is not None
    # payload carries max_chars derived from duration × cps limit
    assert '"max_chars": 18' in client.calls[0][1]["content"]


def test_compress_retries_with_feedback_and_keeps_original_on_failure() -> None:
    cues = [_cue("原始字幕文本保持不动", 0.0, 1.0)]  # cap = 9 chars
    client = _FakeClient(
        [
            '{"results": [{"id": 0, "text": "这个压缩结果实在是太长了超标了"}]}',
            '{"results": [{"id": 0, "text": "这个压缩结果实在是太长了超标了"}]}',
        ]
    )
    compress_over_cps(cues, [0], _config(), llm=client)
    assert cues[0].text == "原始字幕文本保持不动"  # untouched
    assert "previous_error" in client.calls[1][1]["content"]


def test_compress_noop_without_llm_or_indices() -> None:
    cues = [_cue("文本", 0.0, 1.0)]
    assert compress_over_cps(cues, [], _config(), llm=_FakeClient([])) is None
    assert compress_over_cps(cues, [0], _config()) is None


# ── pace integration ──────────────────────────────────────


def test_pace_compresses_over_cps_translation() -> None:
    # 21 zh chars over 2s = 10.5 cps > 9, no neighbouring gap to borrow from.
    cues = [
        _cue("一段太长太长太长太长太长太长太长太长的字幕", 0.0, 2.0),
        _cue("短", 2.105, 3.0),
    ]

    def fake_compress(result, indices, config, **_kwargs):
        result[0].text = "压缩后的短字幕"
        return {"total_tokens": 5}

    with patch("light_subtitle.subtitle.pace.compress.compress_over_cps", side_effect=fake_compress) as m:
        out, usage = pace.correct(cues, _config())
    assert m.called
    assert out[0].text == "压缩后的短字幕"
    assert usage.get("total_tokens") == 5


def test_pace_skips_compression_for_source_lang_cues() -> None:
    cues = [_cue("an english cue that is dense but stays within en cps tolerance", 0.0, 2.0, lang="en")]
    with patch("light_subtitle.subtitle.pace.compress.compress_over_cps") as m:
        pace.correct(cues, _config())
    assert not m.called


def test_pace_skips_compression_without_target_lang() -> None:
    cues = [_cue("an english cue that is way too dense for the english cps ceiling here", 0.0, 2.0, lang="en")]
    config = LayoutConfig(cps_limit_en=10)
    with patch("light_subtitle.subtitle.pace.compress.compress_over_cps") as m:
        pace.correct(cues, config)
    assert not m.called


def test_fix_cue_duration_bounded_exemption_for_merged() -> None:
    from light_subtitle.subtitle.pace import _fix_cue_duration

    merged = _cue("合并后的较长字幕", 0.0, 6.6)
    merged.merged_from = ["p2"]
    out = _fix_cue_duration(merged, _config(max_duration=5.0))
    assert out[0].end == 6.6  # within ×1.5 bound → untouched

    over = _cue("更长的字幕", 0.0, 8.0)
    over.merged_from = ["p2"]
    out = _fix_cue_duration(over, _config(max_duration=5.0))
    assert out[0].end == 7.5  # hard-capped at ×1.5

    plain = _cue("普通字幕", 0.0, 6.6)
    out = _fix_cue_duration(plain, _config(max_duration=5.0))
    assert out[0].end == 5.0  # plain cues still capped at max_duration
