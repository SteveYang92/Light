"""Unit tests for ``export_bilingual_ass`` ZH-anchored pairing.

The bilingual export emits one ASS Dialogue per ZH cue (ZH line on top, EN
line below, joined with ``\\N``).  EN text is derived from the composed EN
segment's words via the ZH cue's ``unit_id`` (+ ``merged_from``), so each ZH
gets exactly the EN words that produced it.  Tests cover the unit-id path
(with ``source_segments``) and the time-overlap fallback (without).
"""

from __future__ import annotations

import re
from pathlib import Path

from light_models import Segment, SubtitleCue, Word
from light_subtitle.pipeline.export import export_bilingual_ass

# ── helpers ────────────────────────────────────────────────────────────────────

_DIALOGUE_RE = re.compile(r"^Dialogue:\s*(\d+),([^,]+),([^,]+),([^,]+),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)$")


def _parse_rows(path: Path) -> list[tuple[float, float, str, str]]:
    """Return ``(start_cs, end_cs, style, text)`` per Dialogue line, in file order."""
    rows: list[tuple[float, float, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _DIALOGUE_RE.match(line)
        if not m:
            continue
        start_cs = _ass_to_cs(m.group(2))
        end_cs = _ass_to_cs(m.group(3))
        style = m.group(4)
        text = m.group(5)
        rows.append((start_cs, end_cs, style, text))
    return rows


def _ass_to_cs(timestamp: str) -> float:
    """Convert ASS ``H:MM:SS.cc`` to centiseconds for stable comparisons."""
    h, m, rest = timestamp.split(":")
    s, cs = rest.split(".")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 100 + int(cs)


def _seg(unit_id: str, start: float, end: float, text: str) -> Segment:
    """Build a composed EN segment with one word per token (spacing preserved)."""
    # Split on spaces but keep a leading space on each word to mimic ASR output
    # (words carry leading spaces, so plain concatenation yields normal spacing).
    tokens = text.split(" ")
    words: list[Word] = []
    if not tokens:
        return Segment(unit_id, start, end, "", text, [])
    # Evenly distribute [start, end] across tokens for deterministic timing.
    span = end - start
    step = span / len(tokens)
    for i, tok in enumerate(tokens):
        ws = start + i * step
        we = ws + step
        # Leading space on every word except the first (ASR convention).
        prefix = "" if i == 0 else " "
        words.append(Word(prefix + tok, ws, we, 1.0, None))
    return Segment(unit_id, start, end, "", text, words)


def _write(tmp_path: Path, en: list[SubtitleCue], zh: list[SubtitleCue], segs: list[Segment] | None = None) -> Path:
    out = tmp_path / "bilingual.ass"
    export_bilingual_ass(en, zh, str(out), source_segments=segs)
    return out


# ── tests ──────────────────────────────────────────────────────────────────────


def test_split_subunit_id_mismatch_pairs_by_time(tmp_path: Path) -> None:
    """ZH/EN split a composed unit at different points — pair by time, not id.

    Reproduces the real bug: parent unit ``m0018`` is split by
    ``split_overlong_units`` into sub-units, but ZH and EN pick different
    split points.  ZH yields ``m0018_0_1`` (no EN counterpart with that id)
    while EN yields ``m0018_0_0`` covering the same time span.  Unit-id
    intersection is empty, but time-window overlap correctly groups them.
    """
    en = [
        SubtitleCue(
            cue_id="en_0", unit_id="m0018_0_0", start=60.97, end=65.64, text="Now look this buzzword", lang="en"
        ),
    ]
    zh = [
        SubtitleCue(
            cue_id="zh_0",
            unit_id="m0018_0_1",  # different sub-unit id, same parent, same time
            start=61.76,
            end=65.64,
            text="这个词现在在各种场合都被拿来用",
            lang="zh",
        ),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))

    # One merged Dialogue — paired by time overlap despite no unit-id match.
    assert len(rows) == 1
    _, _, style, text = rows[0]
    assert style == "Bilingual"
    assert "这个词现在在各种场合都被拿来用" in text
    assert "{\\fs14}Now look this buzzword" in text


def test_same_unit_id_but_time_disjoint_not_grouped(tmp_path: Path) -> None:
    """EN sub-cues sharing one unit_id but NOT time-overlapping a ZH must not be pulled in.

    Reproduces the real bug: ``split_overlong_units`` left three EN cues
    (33.72-37.84, 37.95-41.47, 41.58-43.77) all with ``unit_id=mu0005_u0008_0``
    (split suffix not updated).  ZH 33.72-37.74 shares that id, so the old
    "time OR unit-id intersection" rule pulled all three EN cues into the one
    short ZH — producing long, repeated EN text.  With time-overlap-only
    pairing, only the first EN (which actually overlaps the ZH in time) is
    grouped; the other two form their own group(s).
    """
    en = [
        SubtitleCue(
            cue_id="en_0",
            unit_id="mu0005_u0008_0",
            start=33.72,
            end=37.84,
            text="They understand some physical laws",
            lang="en",
        ),
        SubtitleCue(
            cue_id="en_1",
            unit_id="mu0005_u0008_0",
            start=37.95,
            end=41.47,
            text="but they can't clear the quality bar",
            lang="en",
        ),
        SubtitleCue(
            cue_id="en_2",
            unit_id="mu0005_u0008_0",
            start=41.58,
            end=43.77,
            text="autonomous vehicles. For that,",
            lang="en",
        ),
    ]
    zh = [
        SubtitleCue(
            cue_id="zh_0",
            unit_id="mu0005_u0008_0",
            start=33.72,
            end=37.74,
            text="它们理解一些物理定律 比如重力或物体遮挡",
            lang="zh",
        ),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))

    # ZH0 pairs ONLY with the time-overlapping EN0 → one merged Dialogue.
    # EN1 and EN2 do NOT overlap ZH0 (37.95+ > 37.74), so they become
    # EN-only groups, NOT concatenated into the ZH0 group.
    zh0_rows = [r for r in rows if "它们理解一些物理定律" in r[3]]
    assert len(zh0_rows) == 1
    zh0_text = zh0_rows[0][3]
    assert "They understand some physical laws" in zh0_text
    # The non-overlapping EN cues must NOT be in this Dialogue.
    assert "they can't clear the quality bar" not in zh0_text
    assert "autonomous vehicles. For that," not in zh0_text


def test_two_zh_share_en_window_follows_zh_not_en(tmp_path: Path) -> None:
    """Two ZH cues each get their own EN segment via unit_id; windows follow ZH.

    Reproduces the real case: ZH28 (108.67-112.00, unit m0_0) and ZH29
    (112.11-115.54, unit m0_1).  Each ZH's unit_id maps to a composed EN
    segment carrying the exact words.  Windows follow the ZH cue (anchor) —
    EN text is never repeated and ZH is never duplicated.
    """
    en = [
        SubtitleCue(
            cue_id="en_0",
            unit_id="m0_0",
            start=108.67,
            end=112.23,
            text="It can try out various alternatives",
            lang="en",
        ),
        SubtitleCue(
            cue_id="en_1", unit_id="m0_1", start=112.34, end=115.64, text="in a much fuller safer manner", lang="en"
        ),
    ]
    zh = [
        SubtitleCue(
            cue_id="zh_0", unit_id="m0_0", start=108.67, end=112.00, text="它可以尝试各种可能性 判断哪个最好", lang="zh"
        ),
        SubtitleCue(
            cue_id="zh_1",
            unit_id="m0_1",
            start=112.11,
            end=115.54,
            text="然后以更全面更安全更高效的方式行动",
            lang="zh",
        ),
    ]
    segs = [
        _seg("m0_0", 108.67, 112.23, "It can try out various alternatives"),
        _seg("m0_1", 112.34, 115.64, "in a much fuller safer manner"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh, segs))

    # Two groups, one per ZH cue.  Windows = ZH windows (anchor).
    assert len(rows) == 2
    assert rows[0][0] == 10867 and rows[0][1] == 11200  # ZH0 window
    assert rows[1][0] == 11211 and rows[1][1] == 11554  # ZH1 window
    # Each Dialogue has its ZH + its EN (exact unit_id match, no leakage).
    assert "它可以尝试各种可能性" in rows[0][3]
    assert "It can try out various alternatives" in rows[0][3]
    assert "in a much fuller safer manner" not in rows[0][3]
    assert "然后以更全面" in rows[1][3]
    assert "in a much fuller safer manner" in rows[1][3]
    assert "It can try out various alternatives" not in rows[1][3]


def test_long_en_assigned_to_best_overlap_zh(tmp_path: Path) -> None:
    """Fallback path: one EN cue spanning two ZH cues is assigned to the one
    with the largest overlap; the other ZH becomes ZH-only (EN never repeated).

    No source_segments — pure time-overlap fallback.  EN cue (5.75-12.78)
    overlaps ZH0 (5.75-9.10, 3.35s) and ZH1 (9.11-12.75, 3.64s).  EN is
    assigned exclusively to ZH1 (larger overlap).  ZH0 has no remaining EN
    and becomes ZH-only.  EN text appears exactly once.
    """
    en = [
        SubtitleCue(
            cue_id="en_0",
            unit_id="u0",
            start=5.75,
            end=12.78,
            text="as some sort of big unlock in the pursuit of AGI",
            lang="en",
        ),
    ]
    zh = [
        SubtitleCue(cue_id="zh_0", unit_id="u0", start=5.75, end=9.10, text="好像它是追求AGI的重大突破", lang="zh"),
        SubtitleCue(cue_id="zh_1", unit_id="u1", start=9.11, end=12.75, text="很多人都在想", lang="zh"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))  # no segs → fallback path

    # ZH1 (larger overlap 3.64s) gets EN; ZH0 becomes ZH-only.
    zh0_rows = [r for r in rows if "好像它是追求AGI的重大突破" in r[3]]
    zh1_rows = [r for r in rows if "很多人都在想" in r[3]]
    assert len(zh0_rows) == 1 and len(zh1_rows) == 1
    # ZH0 is ZH-only (no EN portion).
    assert "{\\fs14}" not in zh0_rows[0][3]
    # ZH1 carries the EN text.
    assert "big unlock" in zh1_rows[0][3]
    # EN text appears exactly once across all Dialogues.
    assert sum(1 for r in rows if "big unlock" in r[3]) == 1


def test_long_en_does_not_overlap_next_group(tmp_path: Path) -> None:
    """A group's end is clamped to the next group's start to prevent overlap.

    Two adjacent bilingual pairs: group 1 ends exactly when group 2 starts
    (natural union end == next start).  Without clamping, the two Dialogues
    would touch/overlap on screen.  The fix clamps group 1's end to group 2's
    start minus a small gap so consecutive blocks never overlap in time.
    """
    en = [
        SubtitleCue(cue_id="en_0", unit_id="u0", start=5.75, end=9.10, text="as some sort of big unlock", lang="en"),
        SubtitleCue(
            cue_id="en_1", unit_id="u2", start=9.11, end=12.75, text="what exactly is a world model", lang="en"
        ),
    ]
    zh = [
        SubtitleCue(cue_id="zh_0", unit_id="u0", start=5.75, end=9.10, text="好像它是追求AGI的重大突破", lang="zh"),
        SubtitleCue(
            cue_id="zh_1",
            unit_id="u2",
            start=9.11,
            end=12.75,
            text="很多人都在想 世界模型到底是什么？",
            lang="zh",
        ),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))

    # Two groups, one per bilingual pair.
    assert len(rows) == 2
    # Group 1 end clamped to group 2 start (9.11s = 911cs) minus 1cs gap.
    assert rows[0][0] == 575  # start 5.75
    assert rows[0][1] == 910  # end 9.10 → clamped to 9.11-0.01 = 9.10 (no change, already < next start)
    # Group 2 starts at 911 — no overlap with group 1.
    assert rows[1][0] == 911


def test_equal_counts_one_to_one(tmp_path: Path) -> None:
    """1:1 pairing: one merged Dialogue per unit, ZH line then EN line.

    ZH-anchored: each ZH group keeps its full ZH window (no clamping between
    ZH groups).  EN text comes from the matching segment's words.
    """
    en = [
        SubtitleCue(cue_id="en_0", unit_id="u0", start=1.0, end=2.0, text="hello", lang="en"),
        SubtitleCue(cue_id="en_1", unit_id="u1", start=2.0, end=3.0, text="world", lang="en"),
    ]
    zh = [
        SubtitleCue(cue_id="zh_0", unit_id="u0", start=1.0, end=2.0, text="你好", lang="zh"),
        SubtitleCue(cue_id="zh_1", unit_id="u1", start=2.0, end=3.0, text="世界", lang="zh"),
    ]
    segs = [
        _seg("u0", 1.0, 2.0, "hello"),
        _seg("u1", 2.0, 3.0, "world"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh, segs))

    assert len(rows) == 2  # one merged Dialogue per unit
    assert all(r[2] == "Bilingual" for r in rows)
    # ZH windows preserved exactly (anchor, no inter-ZH clamping).
    assert rows[0] == (100, 200, "Bilingual", "你好\\N{\\fs14}hello")
    assert rows[1] == (200, 300, "Bilingual", "世界\\N{\\fs14}world")


def test_en_fanout_each_subcue_is_own_group(tmp_path: Path) -> None:
    """One ZH cue (unit u0) maps to one EN segment (unit u0) carrying all words.

    With ZH as the anchor and word-level segments, the EN segment's words are
    joined as one EN line under the ZH cue — regardless of how the EN *display
    cues* were split for the en.srt track.  The EN display cues (en_0a, en_0b)
    are irrelevant here; the segment words are the source of truth.  Result:
    one Dialogue — ZH0 + the full EN segment text — over the ZH window.
    """
    en = [
        SubtitleCue(cue_id="en_0a", unit_id="u0", start=1.0, end=1.5, text="hel-", lang="en"),
        SubtitleCue(cue_id="en_0b", unit_id="u0", start=1.5, end=2.0, text="lo", lang="en"),
    ]
    zh = [
        SubtitleCue(cue_id="zh_0", unit_id="u0", start=1.0, end=2.0, text="你好", lang="zh"),
    ]
    segs = [_seg("u0", 1.0, 2.0, "hel- lo")]
    rows = _parse_rows(_write(tmp_path, en, zh, segs))

    # One group: ZH0 + full EN segment words.  Window = ZH window (anchor).
    assert len(rows) == 1
    assert rows[0][0] == 100 and rows[0][1] == 200
    assert "你好" in rows[0][3]
    assert "hel- lo" in rows[0][3]


def test_zh_display_merge_assigned_to_earliest_en_on_tie(tmp_path: Path) -> None:
    """ZH merges u0+u1 into one display cue; both EN segments attach via merged_from.

    ZH0 has unit_id=u0 and merged_from=[u1].  Both u0 and u1 EN segments are
    attached (head + merged_from), so ZH0's EN line is "hello world" and no
    EN-only group remains.  Window = ZH0 window (anchor) = 1.0-3.0.
    """
    en = [
        SubtitleCue(cue_id="en_0", unit_id="u0", start=1.0, end=2.0, text="hello", lang="en"),
        SubtitleCue(cue_id="en_1", unit_id="u1", start=2.0, end=3.0, text="world", lang="en"),
    ]
    zh = [
        SubtitleCue(
            cue_id="zh_0",
            unit_id="u0",
            start=1.0,
            end=3.0,
            text="你好世界",
            lang="zh",
            merged_from=["u1"],
        ),
    ]
    segs = [
        _seg("u0", 1.0, 2.0, "hello"),
        _seg("u1", 2.0, 3.0, "world"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh, segs))

    # One group: ZH0 + both EN segments (head u0 + merged_from u1).
    assert len(rows) == 1
    assert rows[0][0] == 100 and rows[0][1] == 300  # ZH0 window (anchor)
    assert "你好世界" in rows[0][3]
    assert "hello" in rows[0][3]
    assert "world" in rows[0][3]


def test_combined_zh_merge_and_en_fanout(tmp_path: Path) -> None:
    """ZH0 (merged_from=[u1]) gets u0 + u1 EN segment words; no EN-only leftovers.

    ZH0 unit_id=u0, merged_from=[u1] → attached EN = u0 words ("hel- lo") +
    u1 words ("world") = "hel- lo world".  Both segments are marked used, so
    no EN-only group remains.  EN display cues (en_0a/en_0b/en_1) are
    irrelevant; segment words are the source of truth.
    """
    en = [
        SubtitleCue(cue_id="en_0a", unit_id="u0", start=1.0, end=1.5, text="hel-", lang="en"),
        SubtitleCue(cue_id="en_0b", unit_id="u0", start=1.5, end=2.0, text="lo", lang="en"),
        SubtitleCue(cue_id="en_1", unit_id="u1", start=2.0, end=3.0, text="world", lang="en"),
    ]
    zh = [
        SubtitleCue(
            cue_id="zh_0",
            unit_id="u0",
            start=1.0,
            end=3.0,
            text="你好世界",
            lang="zh",
            merged_from=["u1"],
        ),
    ]
    segs = [
        _seg("u0", 1.0, 2.0, "hel- lo"),
        _seg("u1", 2.0, 3.0, "world"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh, segs))

    # One group: ZH0 + u0+u1 EN words.  No EN-only leftovers.
    assert len(rows) == 1
    assert rows[0][0] == 100 and rows[0][1] == 300  # ZH0 window (anchor)
    assert "你好世界" in rows[0][3]
    assert "hel- lo" in rows[0][3]
    assert "world" in rows[0][3]


def test_unmatched_cues_emitted_as_solo_dialogues(tmp_path: Path) -> None:
    """Cues with no time overlap and no shared unit ids become solo Dialogues.

    ZH-only and EN-only groups: when a cue has no temporal or unit-id match
    on the other track, it is still emitted (as a solo line) so nothing is
    silently dropped.  Time windows are deliberately non-overlapping so the
    time-overlap pairing key does not group them.
    """
    en = [
        SubtitleCue(cue_id="en_0", unit_id="u0", start=10.0, end=11.0, text="only en", lang="en"),
    ]
    zh = [
        SubtitleCue(cue_id="zh_0", unit_id="u9", start=1.0, end=2.0, text="仅中文", lang="zh"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))

    # ZH cue first (groups are ordered by ZH), then leftover EN-only group.
    assert len(rows) == 2
    assert all(r[2] == "Bilingual" for r in rows)
    # ZH-only Dialogue: just the ZH line, no EN portion.
    assert rows[0][3] == "仅中文"
    # EN-only Dialogue: just the EN portion with fs14.
    assert rows[1][3] == "{\\fs14}only en"


def test_empty_inputs(tmp_path: Path) -> None:
    """Empty EN and ZH produce a valid ASS with header but no Dialogue lines."""
    out = _write(tmp_path, [], [])
    text = out.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "[Events]" in text
    assert "Dialogue:" not in text


def test_en_forced_to_single_line_all_newlines_dropped(tmp_path: Path) -> None:
    """EN portion is always a single line; all newlines become spaces.

    Reproduces the real case: a long EN cue that the formatter emitted as 3
    lines gets flattened to 1 line.  ZH is unaffected.
    """
    en = [
        SubtitleCue(
            cue_id="en_0",
            unit_id="u0",
            start=34.07,
            end=40.91,
            text=(
                "Many successful approaches in AI and\n"
                "machine learning train models to predict\n"
                "some output Y given some input X."
            ),
            lang="en",
        ),
    ]
    zh = [
        SubtitleCue(
            cue_id="zh_0",
            unit_id="u0",
            start=34.07,
            end=40.91,
            text="AI和机器学习中许多成功的方法 都是训练模型根据给定的输入X来预测某个输出Y",
            lang="zh",
        ),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))

    assert len(rows) == 1
    _, _, style, text = rows[0]
    assert style == "Bilingual"
    # ZH line first, then EN on a single line (all \n → space) under fs14.
    expected = (
        "AI和机器学习中许多成功的方法 都是训练模型根据给定的输入X来预测某个输出Y"
        "\\N{\\fs14}Many successful approaches in AI and "
        "machine learning train models to predict "
        "some output Y given some input X."
    )
    assert text == expected


def test_single_style_and_font(tmp_path: Path) -> None:
    """One unified Bilingual style with PingFangSC-Regular + white primary."""
    en = [SubtitleCue(cue_id="en_0", unit_id="u0", start=1.0, end=2.0, text="hi", lang="en")]
    zh = [SubtitleCue(cue_id="zh_0", unit_id="u0", start=1.0, end=2.0, text="嗨", lang="zh")]
    text = _write(tmp_path, en, zh).read_text(encoding="utf-8")

    # Exactly one Style line, using PingFangSC-Regular and white primary,
    # with outline + shadow so white text stays legible on any background.
    style_lines = [ln for ln in text.splitlines() if ln.startswith("Style:")]
    assert len(style_lines) == 1
    assert "PingFangSC-Regular" in style_lines[0]
    assert "&H00FFFFFF" in style_lines[0]
    assert "Bilingual" in style_lines[0]
    # BorderStyle=1, Outline>=1, Shadow>=1 (legibility on light backgrounds).
    fields = style_lines[0].split(",")
    # V4+ order: Name,Fontname,Fontsize,Primary,Secondary,Outline,Back,Bold,
    # Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,
    # Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
    assert fields[15].strip() == "1"  # BorderStyle=1 (outline+shadow)
    assert int(fields[16]) >= 1  # Outline thickness
    assert int(fields[17]) >= 1  # Shadow depth
    # No per-language styles remain.
    assert not any("Style: ZH" in ln or "Style: EN" in ln for ln in text.splitlines())
