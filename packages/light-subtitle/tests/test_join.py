"""Tests for the join pass (translate/join.py)."""

from __future__ import annotations

import json
from unittest.mock import patch

from light_models import Segment, SubtitleCue, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.join import (
    _apply_merge,
    _apply_shift,
    _find_candidates,
    _parse_ops,
    _validate_merge,
    _validate_ops,
    _validate_shift,
    join_cues,
    save_joined_units,
)


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=1.0)


def _cue(uid: str, zh: str, words: list[Word], *, merged_from: list[str] | None = None) -> SubtitleCue:
    return SubtitleCue(
        cue_id=f"zh_{uid}",
        unit_id=uid,
        start=words[0].start,
        end=words[-1].end,
        text=zh,
        lang="zh",
        words=words,
        merged_from=merged_from or [],
    )


def _unit(uid: str, words: list[Word]) -> Segment:
    return Segment(
        unit_id=uid,
        start=words[0].start,
        end=words[-1].end,
        speaker="",
        source_text=" ".join(w.text for w in words),
        words=words,
    )


def _config(max_duration: float = 5.0, **kwargs) -> SubtitleConfig:
    kwargs.setdefault("llm_api_key", "k")
    return SubtitleConfig(input_path="d.mp4", target_lang="zh", max_duration=max_duration, cps_limit=9, **kwargs)


def _p0002_pair() -> tuple[list[SubtitleCue], list[Segment]]:
    """The #3 review case: "你自然会 | 想 文本压缩有没有一个"."""
    w1 = [_w("so", 0.0, 0.3), _w("you", 0.3, 0.5), _w("might", 0.5, 0.8), _w("naturally", 0.8, 1.4)]
    w2 = [
        _w("wonder,", 1.4, 1.9),
        _w("is", 1.9, 2.1),
        _w("there", 2.1, 2.4),
        _w("some", 2.4, 2.7),
        _w("limit", 2.7, 3.2),
    ]
    cues = [_cue("p0002_0", "所以你自然会", w1), _cue("p0002_1", "想 文本压缩有没有一个", w2)]
    units = [_unit("p0002_0", w1), _unit("p0002_1", w2)]
    return cues, units


# ── parsing ───────────────────────────────────────────────


def test_parse_ops_accepts_merge_and_shift() -> None:
    raw = '{"ops": [{"type": "merge", "from": 1, "to": 3}, {"type": "shift", "boundary": 5, "en_words": -2}]}'
    ops = _parse_ops(raw)
    assert ops == [
        {"type": "merge", "from": 1, "to": 3},
        {"type": "shift", "boundary": 5, "en_words": -2},
    ]


def test_parse_ops_rejects_bad_shapes() -> None:
    assert _parse_ops("not json") is None
    assert _parse_ops('{"ops": [{"type": "merge", "from": "x", "to": 3}]}') is None
    assert _parse_ops('{"ops": [{"type": "split", "at": 3}]}') is None


# ── merge validation ──────────────────────────────────────


def test_validate_merge_ok() -> None:
    cues, _ = _p0002_pair()
    assert _validate_merge({"type": "merge", "from": 0, "to": 1}, cues, _config()) is None


def test_validate_merge_rejects_single_and_speakers_and_caps() -> None:
    cues, _ = _p0002_pair()
    assert _validate_merge({"type": "merge", "from": 0, "to": 0}, cues, _config())
    cues[1].speaker = "S2"
    cues[0].speaker = "S1"
    assert "speaker" in (_validate_merge({"type": "merge", "from": 0, "to": 1}, cues, _config()) or "")
    # char cap: 49+ chars merged
    long_w = [_w(f"w{i}", i * 0.2, i * 0.2 + 0.15) for i in range(20)]
    long_cues = [
        _cue("a", "很长的中文" * 5, long_w[:10]),
        _cue("b", "也很长的中文" * 5, long_w[10:]),
    ]
    assert "chars" in (_validate_merge({"type": "merge", "from": 0, "to": 1}, long_cues, _config()) or "")


# ── shift validation ──────────────────────────────────────


def test_validate_shift_ok() -> None:
    cues, _ = _p0002_pair()
    op = {"type": "shift", "boundary": 0, "en_words": 1}
    assert _validate_shift(op, cues, _config()) is None


def test_validate_shift_rejects_bad_inputs() -> None:
    cues, _ = _p0002_pair()
    base = {"type": "shift", "boundary": 0, "en_words": 1}
    assert _validate_shift({**base, "en_words": 0}, cues, _config())
    assert _validate_shift({**base, "en_words": 5}, cues, _config())  # donor emptied
    cues[1].speaker = "S2"
    cues[0].speaker = "S1"
    assert _validate_shift(base, cues, _config())  # speaker mixing


def test_validate_shift_caps_use_moved_word_timing() -> None:
    # prev is 0.8s; moving one 0.5s word in makes it 1.3s — duration fine.
    # But ask for a config where the cap is tiny so it must fail.
    cues, _ = _p0002_pair()
    op = {"type": "shift", "boundary": 0, "en_words": 1}
    assert _validate_shift(op, cues, _config(max_duration=1.0))


def test_validate_ops_rejects_overlap() -> None:
    cues, _ = _p0002_pair()
    cues = cues + [_cue("p9", "第三条", [_w("x", 3.2, 3.6)])]
    ops = [
        {"type": "merge", "from": 0, "to": 1},
        {"type": "shift", "boundary": 1, "en_words": 1},
    ]
    _, problems = _validate_ops(ops, cues, _config())
    assert any("overlap" in p for p in problems)


def _mock_retranslate(texts_by_id: dict[str, str]):
    """Patch translate_missing used by _retranslate_pair."""

    def fake(segments, missing_ids, config):
        cues = [
            SubtitleCue(
                cue_id=f"r_{s.unit_id}",
                unit_id=s.unit_id,
                start=s.start,
                end=s.end,
                text=texts_by_id[s.unit_id],
                lang="zh",
            )
            for s in segments
            if s.unit_id in missing_ids and s.unit_id in texts_by_id
        ]
        return cues, {}

    return fake


def _patch_retranslate(prev_text="所以你自然会想", next_text="文本压缩有没有一个"):
    return patch(
        "light_subtitle.pipeline.translate.translate.translate_missing",
        side_effect=_mock_retranslate({"p0002_0": prev_text, "p0002_1b": next_text}),
    )


# ── apply ─────────────────────────────────────────────────


def test_apply_merge_joins_text_time_and_chains() -> None:
    cues, _ = _p0002_pair()
    out = _apply_merge(cues, {"type": "merge", "from": 0, "to": 1})
    assert len(out) == 1
    assert out[0].text == "所以你自然会想 文本压缩有没有一个"
    assert out[0].end == 3.2
    assert out[0].merged_from == ["p0002_1"]
    assert len(out[0].words) == 9


def test_apply_shift_forward_splits_donor_unit() -> None:
    cues, units = _p0002_pair()
    op = {"type": "shift", "boundary": 0, "en_words": 1}
    with _patch_retranslate():
        cues, units = _apply_shift(cues, units, op, _config())

    prev, nxt = cues
    assert prev.text == "所以你自然会想" and nxt.text == "文本压缩有没有一个"
    assert [w.text for w in prev.words] == ["so", "you", "might", "naturally", "wonder,"]
    assert prev.end == 1.9
    assert nxt.start == 1.9

    # donor unit p0002_1 split into a (moved) + b (kept); chains updated
    unit_ids = [u.unit_id for u in units]
    assert unit_ids == ["p0002_0", "p0002_1a", "p0002_1b"]
    assert prev.merged_from == ["p0002_1a"]
    assert nxt.unit_id == "p0002_1b" and nxt.merged_from == []
    assert units[1].source_text == "wonder,"


def test_apply_shift_backward_splits_prev_unit() -> None:
    cues, units = _p0002_pair()
    # move prev's last word ("naturally") to next
    op = {"type": "shift", "boundary": 0, "en_words": -1}
    with patch(
        "light_subtitle.pipeline.translate.translate.translate_missing",
        side_effect=_mock_retranslate({"p0002_0a": "所以你", "p0002_0b": "自然会想 文本压缩有没有一个"}),
    ):
        cues, units = _apply_shift(cues, units, op, _config())

    prev, nxt = cues
    assert [w.text for w in prev.words] == ["so", "you", "might"]
    assert [w.text for w in nxt.words][:1] == ["naturally"]
    assert prev.end == 0.8 and nxt.start == 0.8
    unit_ids = [u.unit_id for u in units]
    assert unit_ids == ["p0002_0a", "p0002_0b", "p0002_1"]
    assert prev.unit_id == "p0002_0a"
    assert nxt.unit_id == "p0002_0b" and nxt.merged_from == ["p0002_1"]


def test_apply_shift_whole_unit_moves_without_split() -> None:
    w1 = [_w("really", 0.0, 0.4)]
    w2 = [_w("it", 0.4, 0.6)]
    w3 = [_w("works", 0.6, 1.0)]
    # next cue's chain has two units; the first moves wholesale (no split).
    cues = [_cue("p1", "真的", w1), _cue("p2", "它", w2, merged_from=["p3"])]
    cues[1].words = w2 + w3
    units = [_unit("p1", w1), _unit("p2", w2), _unit("p3", w3)]
    op = {"type": "shift", "boundary": 0, "en_words": 1}
    with patch(
        "light_subtitle.pipeline.translate.translate.translate_missing",
        side_effect=lambda segments, missing_ids, config: (
            [
                SubtitleCue(cue_id="r_p1", unit_id="p1", start=0.0, end=0.6, text="真的它", lang="zh"),
                SubtitleCue(cue_id="r_p3", unit_id="p3", start=0.6, end=1.0, text="有效", lang="zh"),
            ],
            {},
        ),
    ):
        out_cues, out_units = _apply_shift(cues, units, op, _config())
    assert out_cues[0].merged_from == ["p2"]  # whole unit moved, id unchanged
    assert [u.unit_id for u in out_units] == ["p1", "p2", "p3"]  # no split
    assert out_cues[1].unit_id == "p3" and out_cues[1].merged_from == []


# ── end to end (mocked LLM) ───────────────────────────────


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def test_join_cues_applies_llm_ops() -> None:
    cues, units = _p0002_pair()
    response = json.dumps({"ops": [{"type": "shift", "boundary": 0, "en_words": 1}]})
    client = _FakeClient([response])
    with (
        patch("light_subtitle.pipeline.translate.join.client_from_config", return_value=client),
        _patch_retranslate(),
    ):
        result = join_cues(cues, units, _config())
    assert result.ops_applied == 1
    assert result.cues[0].text == "所以你自然会想"
    assert result.usage is not None


def test_join_cues_drops_invalid_ops_and_retries() -> None:
    cues, units = _p0002_pair()
    bad = json.dumps({"ops": [{"type": "shift", "boundary": 0, "en_words": 99}]})  # donor emptied → invalid
    good = json.dumps({"ops": [{"type": "merge", "from": 0, "to": 1}]})
    client = _FakeClient([bad, good])
    with patch("light_subtitle.pipeline.translate.join.client_from_config", return_value=client):
        result = join_cues(cues, units, _config())
    assert "previous_error" in client.calls[1][1]["content"]
    assert result.cues[0].text == "所以你自然会想 文本压缩有没有一个"


def test_join_cues_noop_without_api_key() -> None:
    cues, units = _p0002_pair()
    result = join_cues(cues, units, _config(llm_api_key=""))
    assert result.ops_applied == 0 and result.usage is None


# ── candidate enumeration ─────────────────────────────────


def test_find_candidates_flags_dangling_and_flash() -> None:
    cues = [
        _cue("p0", "这非常复杂 但", [_w("a", 0.0, 0.5), _w("b", 0.5, 1.0)]),
        _cue("p1", "先用简单例子热身", [_w("c", 1.0, 1.5), _w("d", 1.5, 2.0)]),
        _cue("p2", "这种情况", [_w("e", 2.0, 2.3)]),
        _cue("p3", "这是完整的一句。", [_w("f", 2.3, 2.8), _w("g", 2.8, 3.3)]),
    ]
    reasons = {c["boundary"]: c["reason"] for c in _find_candidates(cues)}
    assert 0 in reasons and "但" in reasons[0]  # dangling conjunction
    assert 2 in reasons  # flash fragment
    assert 3 not in reasons  # complete sentence, no candidate


def test_candidates_included_in_payload_for_core_range() -> None:
    cues, units = _p0002_pair()  # cue 0 ends with "会" → candidate
    client = _FakeClient(['{"ops": []}'])
    with patch("light_subtitle.pipeline.translate.join.client_from_config", return_value=client):
        join_cues(cues, units, _config())
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["candidates"] and payload["candidates"][0]["boundary"] == 0


def test_join_cues_merges_applied_descending_keeps_indices_stable() -> None:
    """Two merges from one batch: applying high-index first keeps the
    low-index merge at the intended cues."""
    words = [[_w(f"w{i}{j}", float(i + j), float(i + j) + 0.4) for j in range(2)] for i in range(6)]
    cues = [_cue(f"p{i}", f"第{i}条", words[i]) for i in range(6)]
    units = [_unit(f"p{i}", words[i]) for i in range(6)]
    response = json.dumps({"ops": [{"type": "merge", "from": 0, "to": 1}, {"type": "merge", "from": 3, "to": 4}]})
    client = _FakeClient([response])
    with patch("light_subtitle.pipeline.translate.join.client_from_config", return_value=client):
        result = join_cues(cues, units, _config())
    texts = [c.text for c in result.cues]
    assert texts == ["第0条第1条", "第2条", "第3条第4条", "第5条"]
    assert result.cues[0].merged_from == ["p1"]
    assert result.cues[2].merged_from == ["p4"]


# ── persistence ───────────────────────────────────────────


def test_save_joined_units(tmp_path) -> None:
    _, units = _p0002_pair()
    units = units[:1] + [
        Segment(
            unit_id="p0002_1a",
            start=1.4,
            end=1.9,
            speaker="",
            source_text="wonder,",
            words=units[1].words[:1],
        ),
        Segment(
            unit_id="p0002_1b",
            start=1.9,
            end=3.2,
            speaker="",
            source_text="is there some limit",
            words=units[1].words[1:],
        ),
    ]
    save_joined_units(units, tmp_path)
    data = json.loads((tmp_path / "plan.joined.json").read_text(encoding="utf-8"))
    assert [u["unit_id"] for u in data["units"]] == ["p0002_0", "p0002_1a", "p0002_1b"]
    words = json.loads((tmp_path / "segment_words.joined.json").read_text(encoding="utf-8"))
    assert words["p0002_1a"][0]["text"] == "wonder,"
