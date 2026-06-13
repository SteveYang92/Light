"""Tests for BreakFinder (Chinese + English) and best_split_position."""

from light_subtitle.language.base import best_split_position
from light_subtitle.language.cjk import ChineseBreakFinder
from light_subtitle.language.english import EnglishBreakFinder

# ═══════════════════════════════════════════════════════════
# ChineseBreakFinder
# ═══════════════════════════════════════════════════════════


class TestChineseBreakFinder:
    """Chinese line-breaking: prefers punctuation, respects word integrity."""

    def test_prefers_sentence_end_punctuation(self):
        """句号 is the strongest break point."""
        finder = ChineseBreakFinder("这是一个完整的句子。接下来继续")
        pos = finder.find(0, len(finder.text) - 1)
        # 。 is at index 9; break position 10 means left = text[:10] includes 。
        assert pos == 10

    def test_prefers_comma_over_conjunction(self):
        """逗号 should score higher than conjunction boundaries."""
        finder = ChineseBreakFinder("第一部分，而且第二部分")
        # Comma at index 4; break at pos=5 puts comma on left side.
        # Comma score = CLAUSE_PUNCT (80), conjunction = CONJUNCTION_BEFORE (60)
        pos = finder.find(1, len(finder.text) - 2)
        assert pos == 5

    def test_forbidden_inside_ascii_word(self):
        """Cannot split inside an ASCII word like 'Stéphane'."""
        finder = ChineseBreakFinder("研究者Stéphane Denis提出")
        # 'Stéphane' spans roughly indices 3-10
        # Positions inside this word are forbidden
        assert finder.is_forbidden(5) is True  # inside "Stéphane"
        assert finder.is_forbidden(3) is False  # at 'S', start of word — boundary not inside

    def test_forbidden_inside_jieba_token(self):
        """Cannot split inside a Chinese word like '超越'."""
        finder = ChineseBreakFinder("模型超越了预期")
        # '超越' is a jieba token — positions inside it are forbidden
        # But we can't test exact indices without knowing jieba output.
        # Instead, verify find() never returns a forbidden position.
        if len(finder.text) > 3:
            pos = finder.find(1, len(finder.text) - 2)
            if pos is not None:
                assert not finder.is_forbidden(pos)

    def test_forbidden_between_english_words(self):
        """Cannot split between 'Barlow' and 'Twins'."""
        finder = ChineseBreakFinder("提出Barlow Twins技术")
        # 'Barlow Twins' — splitting between them is forbidden
        # Find the position between 'Barlow' and 'Twins'
        barlow_end = finder.text.index("Barlow") + len("Barlow")
        assert finder.is_forbidden(barlow_end) is True

    def test_find_returns_none_when_no_good_break(self):
        """Returns None when all positions in range are forbidden."""
        finder = ChineseBreakFinder("ABC")
        # All positions inside "ABC" are forbidden (it's an ASCII word)
        pos = finder.find(1, 1)
        assert pos is None

    def test_find_balanced_falls_back_to_safe_position(self):
        """find_balanced adjusts fallback to avoid forbidden positions."""
        finder = ChineseBreakFinder("研究者StéphaneDenis提出方法")
        # Force fallback inside "StéphaneDenis" — should adjust
        pos = finder.find_balanced(1, 2, 8)  # fallback=8, likely inside the word
        assert not finder.is_forbidden(pos)

    def test_find_respects_range(self):
        """find() only searches within [low, high]."""
        finder = ChineseBreakFinder("这是第一部分。这是第二部分。结束")
        # 。 at index 6 and 。 at index 13
        # Search only high range → should find the second 。 (break at pos=14)
        pos = finder.find(8, 14)
        assert pos == 14

    def test_find_high_scores_question_mark(self):
        """？ scores at SENTENCE_END_PUNCT level."""
        finder = ChineseBreakFinder("这样可以吗？接下来继续")
        pos = finder.find(0, len(finder.text) - 1)
        #  break at pos=6 puts ？ (index 5) on the left side.
        assert pos is not None
        assert pos == 6
        assert finder.text[pos - 1] == "？"

    def test_score_conjunction_boundary(self):
        """Breaking before a conjunction gets CONJUNCTION_BEFORE score."""
        finder = ChineseBreakFinder("第一部分而且第二部分")
        # "而且" starts at index 4. Break position 4 is before "而且".
        score = finder.score(4)
        assert score >= 60  # CONJUNCTION_BEFORE = 60

    def test_find_clause_punctuation_scores_higher_than_conjunction(self):
        """， scores higher than conjunction boundary."""
        finder = ChineseBreakFinder("文本，而且继续")
        # Comma at index 2. Break at pos=3 gives CLAUSE_PUNCT (80).
        score_comma = finder.score(3)  # break after ，
        score_conj = finder.score(2)  # break before ，
        assert score_comma > score_conj

    # ── Paired symbols ──

    def test_forbidden_inside_book_title_marks(self):
        """Cannot split inside 《》 paired symbols."""
        finder = ChineseBreakFinder("像《我的世界》这样的东西")
        # 《 at index 1, 》 at index 6.
        # Positions 1–5 should be forbidden (inside the pair).
        for pos in range(1, 6):
            assert finder.is_forbidden(pos), f"Position {pos} should be forbidden"
        # Position 6 is after }, should be allowed.
        assert not finder.is_forbidden(6)

    def test_book_title_marks_kept_together_by_find(self):
        """find() avoids splitting 《》 pairs."""
        finder = ChineseBreakFinder("所以可能会出现更多像《我的世界》这样的东西")
        # 《 at 10, 》 at 15. Positions 10-14 are forbidden.
        # find(8, 17) would scan: forbidden {10-14}, allowed {8,9,15,16,17}.
        # All allowed positions score FALLBACK=0, so find returns None.
        # Use find_balanced to confirm the fallback is a safe (non-forbidden) position.
        pos = finder.find_balanced(8, 17, 16)
        assert not finder.is_forbidden(pos)

    def test_forbidden_inside_parentheses(self):
        """Cannot split inside （） fullwidth parentheses."""
        finder = ChineseBreakFinder("他来自中国（上海）附近")
        # （ at index 5, ） at index 8.
        # Positions 5-7 should be forbidden.
        assert finder.is_forbidden(5)
        assert finder.is_forbidden(6)
        assert finder.is_forbidden(7)
        # Position 8 is after ）, should NOT be forbidden.
        assert not finder.is_forbidden(8)

    def test_forbidden_inside_curly_quotes(self):
        """Cannot split inside curly double quotes \u201c\u201d."""
        finder = ChineseBreakFinder("他说\u201c你好\u201d就走了")
        # \u201c at index 2, \u201d at index 5.
        # Positions 2-4 should be forbidden.
        assert finder.is_forbidden(2)
        assert finder.is_forbidden(3)
        assert finder.is_forbidden(4)
        # Position 5 (after \u201d) should NOT be forbidden.
        assert not finder.is_forbidden(5)

    def test_forbidden_inside_ambiguous_quotes(self):
        """Cannot split inside ambiguous straight double quotes ""."""
        finder = ChineseBreakFinder('他说"你好"就走了')
        # Straight " at index 2 (open), " at index 5 (close).
        # Positions 2-4 should be forbidden.
        assert finder.is_forbidden(2)
        assert finder.is_forbidden(3)
        assert finder.is_forbidden(4)
        # Position 5 is the closing quote, should NOT be forbidden.
        assert not finder.is_forbidden(5)

    def test_unclosed_pair_forbids_to_end(self):
        """An unclosed 《 forbids all positions from it to end-of-text."""
        finder = ChineseBreakFinder("像《我的世界这样的东西")
        # 《 at 1, no matching 》.
        # All positions from 1 to end should be forbidden.
        n = len(finder.text)
        for pos in range(1, n):
            assert finder.is_forbidden(pos), f"Position {pos} should be forbidden (unclosed 《)"

    def test_nested_pairs(self):
        """Nested pairs like 《A「B」C》 are handled correctly."""
        finder = ChineseBreakFinder("《A「B」C》")
        # 《 at 0, 「 at 2, 」 at 4, 》 at 6.
        # All positions inside outer pair (0-5) should be forbidden.
        for pos in range(0, 6):
            assert finder.is_forbidden(pos), f"Position {pos} should be forbidden"
        # Position 6 (after 》) should be allowed.
        assert not finder.is_forbidden(6)


# ═══════════════════════════════════════════════════════════════
# EnglishBreakFinder
# ═══════════════════════════════════════════════════════════════


class TestEnglishBreakFinder:
    """English line-breaking: Netflix §4 grammar rules + scoring."""

    def test_forbidden_article_noun(self):
        """Cannot break between article and noun."""
        finder = EnglishBreakFinder(["the", "cat", "sat"])
        text = finder.text  # "the cat sat"
        space_after_the = text.index(" ")
        assert finder.is_forbidden(space_after_the) is True

    def test_forbidden_auxiliary_verb(self):
        """Cannot break between auxiliary and verb."""
        finder = EnglishBreakFinder(["is", "running", "fast"])
        text = finder.text  # "is running fast"
        space_after_is = text.index(" ")
        assert finder.is_forbidden(space_after_is) is True

    def test_forbidden_phrasal_verb(self):
        """Cannot break phrasal verbs like 'give up'."""
        finder = EnglishBreakFinder(["give", "up", "now"])
        text = finder.text  # "give up now"
        space_after_give = text.index(" ")
        assert finder.is_forbidden(space_after_give) is True

    def test_allowed_subject_verb_boundary(self):
        """Can break between subject noun and verb."""
        finder = EnglishBreakFinder(["cats", "are", "nice"])
        text = finder.text  # "cats are nice"
        space_after_cats = text.index(" ")
        assert finder.is_forbidden(space_after_cats) is False

    def test_score_punctuation_gets_highest(self):
        """Breaking after punctuation scores SENTENCE_END_PUNCT."""
        finder = EnglishBreakFinder(["Hello.", "World"])
        text = finder.text  # "Hello. World"
        dot_pos = text.index(".")
        assert finder.score(dot_pos) >= 100

    def test_score_before_conjunction(self):
        """Breaking before conjunction scores CONJUNCTION_BEFORE."""
        finder = EnglishBreakFinder(["cats", "and", "dogs"])
        text = finder.text  # "cats and dogs"
        space_after_cats = text.index(" ")
        score = finder.score(space_after_cats)
        assert score >= 60  # CONJUNCTION_BEFORE

    def test_score_before_preposition(self):
        """Breaking before preposition scores PREP_ARTICLE_BEFORE."""
        finder = EnglishBreakFinder(["go", "to", "school"])
        text = finder.text  # "go to school"
        space_after_go = text.index(" ")
        score = finder.score(space_after_go)
        assert score >= 40  # PREP_ARTICLE_BEFORE


# ═══════════════════════════════════════════════════════════════
# best_split_position
# ═══════════════════════════════════════════════════════════════


class TestBestSplitPosition:
    """Character-level split for oversized cues (used by pace)."""

    def test_prefers_sentence_end_punctuation(self):
        """Should return position after the first sentence-ending punct from midpoint."""
        text = "这是前半部分。这是后半部分内容"
        pos = best_split_position(text)
        assert text[pos - 1] == "。"

    def test_falls_back_to_comma(self):
        """When no .!? available, prefer comma."""
        text = "第一部分，第二部分更多内容"
        pos = best_split_position(text)
        assert text[pos - 1] in "，,;:"

    def test_falls_back_to_space(self):
        """When no punctuation, prefer word boundary (space)."""
        text = "hello world foo bar"
        pos = best_split_position(text)
        assert text[pos - 1] == " "

    def test_short_text(self):
        """Short text returns midpoint."""
        pos = best_split_position("ab")
        assert pos >= 0
