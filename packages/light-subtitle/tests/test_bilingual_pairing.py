"""Unit tests for ``export_bilingual_ass`` merged-Dialogue pairing.

The bilingual export merges EN/ZH into one ASS Dialogue per semantic unit
(ZH line on top, EN line below, joined with ``\\N``).  These cover the
pairing cases the old index-paired writer could not handle: EN fan-out
(one unit → many display cues), ZH display-merge (many units → one
display cue), and their combination.  The function signature is
``export_bilingual_ass(en_cues, zh_cues, ...)`` — EN first to match the
``(source_fmt, target_fmt)`` call site.
"""

from __future__ import annotations

import re
from pathlib import Path

from light_models import SubtitleCue
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


def _write(tmp_path: Path, en: list[SubtitleCue], zh: list[SubtitleCue]) -> Path:
    out = tmp_path / "bilingual.ass"
    export_bilingual_ass(en, zh, str(out))
    return out


# ── tests ──────────────────────────────────────────────────────────────────────


def test_equal_counts_one_to_one(tmp_path: Path) -> None:
    """1:1 pairing: one merged Dialogue per unit, ZH line then EN line."""
    en = [
        SubtitleCue(cue_id="en_0", unit_id="u0", start=1.0, end=2.0, text="hello", lang="en"),
        SubtitleCue(cue_id="en_1", unit_id="u1", start=2.0, end=3.0, text="world", lang="en"),
    ]
    zh = [
        SubtitleCue(cue_id="zh_0", unit_id="u0", start=1.0, end=2.0, text="你好", lang="zh"),
        SubtitleCue(cue_id="zh_1", unit_id="u1", start=2.0, end=3.0, text="世界", lang="zh"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))

    assert len(rows) == 2  # one merged Dialogue per unit
    assert all(r[2] == "Bilingual" for r in rows)
    # ZH line first, EN line second with fs14 override.
    assert rows[0] == (100, 200, "Bilingual", "你好\\N{\\fs14}hello")
    assert rows[1] == (200, 300, "Bilingual", "世界\\N{\\fs14}world")


def test_en_fanout_one_dialogue_concatenates_subcues(tmp_path: Path) -> None:
    """One ZH cue covers u0; EN fans out into two sub-cues both under u0.

    Both EN sub-cues group with the single ZH cue into one Dialogue whose
    time window is the union [1.0, 2.0]; the EN portion concatenates the
    sub-cue texts in time order.  This is the case that breaks the old
    index-paired writer.
    """
    en = [
        SubtitleCue(cue_id="en_0a", unit_id="u0", start=1.0, end=1.5, text="hel-", lang="en"),
        SubtitleCue(cue_id="en_0b", unit_id="u0", start=1.5, end=2.0, text="lo", lang="en"),
    ]
    zh = [
        SubtitleCue(cue_id="zh_0", unit_id="u0", start=1.0, end=2.0, text="你好", lang="zh"),
    ]
    rows = _parse_rows(_write(tmp_path, en, zh))

    assert len(rows) == 1  # one merged Dialogue, not three
    start, end, style, text = rows[0]
    assert start == 100 and end == 200  # union window
    assert style == "Bilingual"
    # ZH line, then both EN sub-cues on one line (space-joined) under fs14.
    assert text == "你好\\N{\\fs14}hel- lo"


def test_zh_display_merge_one_dialoge_includes_all_en(tmp_path: Path) -> None:
    """ZH merges u0+u1 into one display cue (merged_from=[u1]); EN stays 1:1.

    The single ZH cue groups with both EN cues (u0 and u1) into one
    Dialogue over the union window [1.0, 3.0]; the EN portion concatenates
    both EN lines in time order.
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
    rows = _parse_rows(_write(tmp_path, en, zh))

    assert len(rows) == 1  # one merged Dialogue
    start, end, style, text = rows[0]
    assert start == 100 and end == 300  # union window
    assert style == "Bilingual"
    assert text == "你好世界\\N{\\fs14}hello world"


def test_combined_zh_merge_and_en_fanout(tmp_path: Path) -> None:
    """ZH merges u0+u1; EN fans u0 into two sub-cues and u1 into one.

    All four cues (1 ZH + 3 EN) share effective unit ids and collapse into
    one Dialogue over [1.0, 3.0]; EN portion lists all three EN lines in
    time order.
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
    rows = _parse_rows(_write(tmp_path, en, zh))

    assert len(rows) == 1
    start, end, style, text = rows[0]
    assert start == 100 and end == 300
    assert style == "Bilingual"
    # 3 EN cues (hel-, lo, world) collapse to a single line: "hel- lo world".
    assert text == "你好世界\\N{\\fs14}hel- lo world"


def test_unmatched_cues_emitted_as_solo_dialogues(tmp_path: Path) -> None:
    """Cues with no shared unit ids become solo Dialogues (ZH-only / EN-only)."""
    en = [
        SubtitleCue(cue_id="en_0", unit_id="u0", start=1.0, end=2.0, text="only en", lang="en"),
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

    # Exactly one Style line, using PingFangSC-Regular and white primary.
    style_lines = [ln for ln in text.splitlines() if ln.startswith("Style:")]
    assert len(style_lines) == 1
    assert "PingFangSC-Regular" in style_lines[0]
    assert "&H00FFFFFF" in style_lines[0]
    assert "Bilingual" in style_lines[0]
    # No per-language styles remain.
    assert not any("Style: ZH" in ln or "Style: EN" in ln for ln in text.splitlines())
