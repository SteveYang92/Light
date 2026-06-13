from light_models import QCIssue, QCReport, SubtitleCue, Word
from light_qc.config import QCConfig
from light_qc.rules.hard import (
    ChineseLineLength,
    EmptyText,
    EnglishLineLength,
    EntryPointAccuracy,
    GapFlash,
    MaxDuration,
    MaxLines,
    MinDuration,
    MissingPunctuation,
    Overlap,
    ReadingSpeed,
    TranslationCompleteness,
)
from light_qc.rules.soft import ExitPointPrecision


def _make_cue(cue_id="c1", start=1.0, end=4.0, text="", lang="zh", speaker="", unit_id="u1"):
    return SubtitleCue(cue_id=cue_id, unit_id=unit_id, start=start, end=end, text=text, lang=lang, speaker=speaker)


def _wrap(cues: list[SubtitleCue]) -> dict[str, list[SubtitleCue]]:
    """Wrap a flat list of cues into the dict format expected by rule check()."""
    return {"zh": cues}


def _cfg(**kwargs):
    defaults = {
        "source_lang": "zh",
        "max_lines": 2,
        "max_lines_zh": 1,
        "max_chars_per_line_zh": 40,
        "max_chars_per_line_en": 42,
        "cps_limit": 9,
        "cps_limit_en": 25,
        "min_duration": 0.8,
        "max_duration": 7.0,
        "min_gap": 0.1,
    }
    defaults.update(kwargs)
    return QCConfig(**defaults)


class TestMaxLines:
    def test_pass_zh(self):
        """1-line Chinese cue passes max_lines_zh=1."""
        cues = [_make_cue(text="单行中文", lang="zh")]
        issues = MaxLines().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_pass_en(self):
        """2-line English cue passes max_lines=2."""
        cues = [_make_cue(text="First line\nSecond line", lang="en")]
        issues = MaxLines().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail_zh(self):
        """2-line Chinese cue fails max_lines_zh=1."""
        cues = [_make_cue(text="第一行\n第二行", lang="zh")]
        issues = MaxLines().check(_wrap(cues), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_fail_en(self):
        """3-line English cue fails max_lines=2."""
        cues = [_make_cue(text="A\nB\nC", lang="en")]
        issues = MaxLines().check(_wrap(cues), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "error"


class TestChineseLineLength:
    def test_pass(self):
        cues = [_make_cue(text="十九个汉字刚好十九个汉字")]
        issues = ChineseLineLength().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [_make_cue(text="这是一行非常长的中文字幕文本超过四十个汉字限制所以这样写肯定会被检查出来标记为超限")]
        issues = ChineseLineLength().check(_wrap(cues), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "suggestion"

    def test_skip_english(self):
        cues = [_make_cue(text="A" * 50, lang="en")]
        issues = ChineseLineLength().check(_wrap(cues), _cfg())
        assert len(issues) == 0


class TestEnglishLineLength:
    def test_pass(self):
        cues = [_make_cue(text="A" * 40, lang="en")]
        issues = EnglishLineLength().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [_make_cue(text="A" * 50, lang="en")]
        issues = EnglishLineLength().check(_wrap(cues), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "error"


class TestReadingSpeed:
    def test_pass(self):
        cues = [_make_cue(start=1.0, end=4.0, text="四个汉字")]
        issues = ReadingSpeed().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [_make_cue(start=1.0, end=2.0, text="十个汉字十个汉字十个汉字")]
        issues = ReadingSpeed().check(_wrap(cues), _cfg())
        assert len(issues) >= 1


class TestMinDuration:
    def test_pass(self):
        cues = [_make_cue(start=1.0, end=2.0)]
        issues = MinDuration().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [_make_cue(start=1.0, end=1.5)]
        issues = MinDuration().check(_wrap(cues), _cfg())
        assert len(issues) == 1


class TestMaxDuration:
    def test_pass(self):
        cues = [_make_cue(start=1.0, end=5.0)]
        issues = MaxDuration().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [_make_cue(start=1.0, end=10.0)]
        issues = MaxDuration().check(_wrap(cues), _cfg())
        assert len(issues) == 1


class TestOverlap:
    def test_pass(self):
        cues = [
            _make_cue(cue_id="c1", start=1.0, end=3.0),
            _make_cue(cue_id="c2", start=3.0, end=5.0),
        ]
        issues = Overlap().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [
            _make_cue(cue_id="c1", start=1.0, end=4.0),
            _make_cue(cue_id="c2", start=3.0, end=5.0),
        ]
        issues = Overlap().check(_wrap(cues), _cfg())
        assert len(issues) == 1


class TestEmptyText:
    def test_pass(self):
        cues = [_make_cue(text="一些文本")]
        issues = EmptyText().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [_make_cue(text="  ")]
        issues = EmptyText().check(_wrap(cues), _cfg())
        assert len(issues) == 1


class TestMissingPunctuation:
    def test_pass_question_has_mark(self):
        cues = [_make_cue(text="真的吗？")]
        issues = MissingPunctuation().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_pass_exclamation_has_mark(self):
        cues = [_make_cue(text="太好了！")]
        issues = MissingPunctuation().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_pass_no_hint_no_flag(self):
        """Statement without question/exclamation hint → no issue (periods optional)."""
        cues = [_make_cue(text="这是一个没有标点的句子")]
        issues = MissingPunctuation().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail_question_hint_no_mark(self):
        cues = [_make_cue(text="这样做真的可以吗")]
        issues = MissingPunctuation().check(_wrap(cues), _cfg())
        assert len(issues) == 1

    def test_fail_exclamation_hint_no_mark(self):
        cues = [_make_cue(text="这简直不可思议")]
        issues = MissingPunctuation().check(_wrap(cues), _cfg())
        assert len(issues) == 1

    def test_skip_english(self):
        cues = [_make_cue(text="This has no Chinese punctuation", lang="en")]
        issues = MissingPunctuation().check(_wrap(cues), _cfg())
        assert len(issues) == 0


class TestGapFlash:
    def test_pass(self):
        cues = [
            _make_cue(cue_id="c1", start=1.0, end=3.0),
            _make_cue(cue_id="c2", start=3.5, end=5.0),
        ]
        issues = GapFlash().check(_wrap(cues), _cfg())
        assert len(issues) == 0

    def test_fail(self):
        cues = [
            _make_cue(cue_id="c1", start=1.0, end=3.0),
            _make_cue(cue_id="c2", start=3.05, end=5.0),
        ]
        issues = GapFlash().check(_wrap(cues), _cfg(min_gap=0.2))
        assert len(issues) == 1


class TestQCReport:
    def test_passed_empty(self):
        report = QCReport(
            total_cues=5,
            errors=0,
            warnings=0,
            suggestions=0,
            passed=True,
            bilingual=False,
            source_lang="zh",
            target_lang=None,
            issues=[],
        )
        assert report.passed is True

    def test_failed_with_errors(self):
        issue = QCIssue(
            severity="error",
            category="硬性规则",
            rule="MaxLines",
            cue_id=3,
            time="00:00:05,000",
            detail="too many lines",
            fix="reduce lines",
        )
        report = QCReport(
            total_cues=5,
            errors=1,
            warnings=0,
            suggestions=0,
            passed=False,
            bilingual=False,
            source_lang="zh",
            target_lang=None,
            issues=[issue],
        )
        assert report.passed is False
        assert report.errors == 1


class TestTranslationCompleteness:
    """Translation completeness — unit_id mode + time-overlap fallback."""

    # ── unit_id mode (pipeline) ──────────────────────────────────

    def test_all_covered_pipeline_b(self):
        src = [
            _make_cue("s1", unit_id="u001", text="原文一"),
            _make_cue("s2", unit_id="u002", text="原文二"),
        ]
        tgt = [
            _make_cue("t1", unit_id="u001", text="trans one", lang="en"),
            _make_cue("t2", unit_id="u002", text="trans two", lang="en"),
        ]
        cfg = _cfg(source_lang="zh", target_lang="en")
        issues = TranslationCompleteness().check({"zh": src, "target": tgt}, cfg)
        assert len(issues) == 0

    def test_missing_unit_pipeline(self):
        src = [
            _make_cue("s1", unit_id="u001", text="原文一"),
            _make_cue("s2", unit_id="u002", text="原文二缺失"),
        ]
        tgt = [
            _make_cue("t1", unit_id="u001", text="trans one", lang="en"),
        ]
        cfg = _cfg(source_lang="zh", target_lang="en")
        issues = TranslationCompleteness().check({"zh": src, "target": tgt}, cfg)
        assert len(issues) == 1
        assert "u002" in issues[0].detail
        assert issues[0].severity == "error"

    def test_orphan_target_unit(self):
        src = [_make_cue("s1", unit_id="u001", text="原文一")]
        tgt = [
            _make_cue("t1", unit_id="u001", text="trans one", lang="en"),
            _make_cue("t2", unit_id="u999", text="ghost", lang="en"),
        ]
        cfg = _cfg(source_lang="zh", target_lang="en")
        issues = TranslationCompleteness().check({"zh": src, "target": tgt}, cfg)
        assert len(issues) == 1
        assert "u999" in issues[0].detail

    def test_scene_c_pipeline(self):
        src = [_make_cue("s1", unit_id="u001", text="Hello", lang="en")]
        tgt = [_make_cue("t1", unit_id="u001", text="你好", lang="zh")]
        cfg = _cfg(source_lang="en", target_lang="zh", bilingual=True)
        issues = TranslationCompleteness().check({"en": src, "zh": tgt}, cfg)
        assert len(issues) == 0

    # ── time-overlap mode (standalone SRT/VTT) ────────────────────

    def test_standalone_all_covered(self):
        src = [
            _make_cue("s1", start=1.0, end=4.0, unit_id="", text="Hello world", lang="en"),
            _make_cue("s2", start=5.0, end=8.0, unit_id="", text="How are you", lang="en"),
        ]
        tgt = [
            _make_cue("t1", start=1.0, end=4.0, unit_id="", text="你好世界", lang="zh"),
            _make_cue("t2", start=5.0, end=8.0, unit_id="", text="你好吗", lang="zh"),
        ]
        cfg = _cfg(source_lang="en", target_lang="zh", bilingual=True)
        issues = TranslationCompleteness().check({"en": src, "zh": tgt}, cfg)
        assert len(issues) == 0

    def test_standalone_missing_overlap(self):
        src = [
            _make_cue("s1", start=1.0, end=4.0, unit_id="", text="Hello", lang="en"),
            _make_cue("s2", start=5.0, end=8.0, unit_id="", text="Missing", lang="en"),
        ]
        tgt = [
            _make_cue("t1", start=1.0, end=4.0, unit_id="", text="你好", lang="zh"),
        ]
        cfg = _cfg(source_lang="en", target_lang="zh", bilingual=True)
        issues = TranslationCompleteness().check({"en": src, "zh": tgt}, cfg)
        assert len(issues) == 1
        assert "Missing" in issues[0].detail

    def test_single_file_skips(self):
        """Standalone translated — only one file, no source to compare."""
        cues = [_make_cue("t1", unit_id="", text="译文", lang="zh")]
        cfg = _cfg(source_lang="en", target_lang="zh")
        issues = TranslationCompleteness().check({"zh": cues}, cfg)
        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════════
# EntryPointAccuracy
# ═══════════════════════════════════════════════════════════════════


def _cue_with_words(text="测试", lang="zh", start=1.0, end=4.0, word_start=1.0, word_end=3.5):
    """Helper: build a cue with one word for alignment tests."""
    word = Word(text="test", start=word_start, end=word_end, confidence=0.9)
    return SubtitleCue(cue_id="c1", unit_id="u1", start=start, end=end, text=text, lang=lang, words=[word])


class TestEntryPointAccuracy:
    def test_zh_small_offset_no_issue(self):
        """zh offset 0.20s < suggestion threshold 0.30s → no issue."""
        cue = _cue_with_words(lang="zh", start=1.0, word_start=0.80)
        issues = EntryPointAccuracy().check(_wrap([cue]), _cfg())
        assert len(issues) == 0

    def test_zh_suggestion_offset(self):
        """zh offset 0.35s >= suggestion 0.30s → suggestion."""
        cue = _cue_with_words(lang="zh", start=1.0, word_start=0.65)
        issues = EntryPointAccuracy().check(_wrap([cue]), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "suggestion"

    def test_zh_warning_offset(self):
        """zh offset 0.60s >= warning 0.50s → warning."""
        cue = _cue_with_words(lang="zh", start=1.0, word_start=0.40)
        issues = EntryPointAccuracy().check(_wrap([cue]), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_en_small_offset_no_issue(self):
        """en offset 0.10s < suggestion threshold 0.15s → no issue."""
        cue = _cue_with_words(lang="en", start=1.0, word_start=0.90)
        issues = EntryPointAccuracy().check(_wrap([cue]), _cfg())
        assert len(issues) == 0

    def test_en_suggestion_offset(self):
        """en offset 0.20s >= suggestion 0.15s → suggestion."""
        cue = _cue_with_words(lang="en", start=1.0, word_start=0.80)
        issues = EntryPointAccuracy().check(_wrap([cue]), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "suggestion"

    def test_en_warning_offset(self):
        """en offset 0.35s >= warning 0.30s → warning."""
        cue = _cue_with_words(lang="en", start=1.0, word_start=0.65)
        issues = EntryPointAccuracy().check(_wrap([cue]), _cfg())
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_no_words_skip(self):
        """No word data → no issue."""
        cue = _make_cue(start=1.0, end=4.0, text="测试")
        issues = EntryPointAccuracy().check(_wrap([cue]), _cfg())
        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════════
# ExitPointPrecision
# ═══════════════════════════════════════════════════════════════════


class TestExitPointPrecision:
    def test_zh_adequate_padding_no_issue(self):
        """zh padding 0.10s >= min 0.08s → no issue."""
        cue = _cue_with_words(lang="zh", end=3.0, word_end=2.90)
        issues = ExitPointPrecision().check(_wrap([cue]), _cfg())
        assert len(issues) == 0

    def test_zh_too_short_padding(self):
        """zh padding 0.03s < min 0.08s → suggestion."""
        cue = _cue_with_words(lang="zh", end=3.0, word_end=2.97)
        issues = ExitPointPrecision().check(_wrap([cue]), _cfg())
        assert len(issues) == 1

    def test_en_adequate_padding_no_issue(self):
        """en padding 0.15s >= min 0.12s → no issue."""
        cue = _cue_with_words(lang="en", end=3.0, word_end=2.85)
        issues = ExitPointPrecision().check(_wrap([cue]), _cfg())
        assert len(issues) == 0

    def test_en_too_short_padding(self):
        """en padding 0.08s < min 0.12s → suggestion."""
        cue = _cue_with_words(lang="en", end=3.0, word_end=2.92)
        issues = ExitPointPrecision().check(_wrap([cue]), _cfg())
        assert len(issues) == 1

    def test_too_long_padding(self):
        """padding 1.5s > max 1.0s → suggestion."""
        cue = _cue_with_words(lang="zh", end=3.0, word_end=1.5)
        issues = ExitPointPrecision().check(_wrap([cue]), _cfg())
        assert len(issues) == 1

    def test_no_words_skip(self):
        """No word data → no issue."""
        cue = _make_cue(start=1.0, end=4.0, text="测试")
        issues = ExitPointPrecision().check(_wrap([cue]), _cfg())
        assert len(issues) == 0

    def test_normal_padding_no_issue(self):
        """padding 0.5s within [0.08, 1.0] → no issue."""
        cue = _cue_with_words(lang="zh", end=3.0, word_end=2.5)
        issues = ExitPointPrecision().check(_wrap([cue]), _cfg())
        assert len(issues) == 0
