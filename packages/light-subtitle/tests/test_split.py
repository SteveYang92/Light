"""Tests for overlong translation-unit splitting."""

import json
import string
from pathlib import Path

from light_models import Segment, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate import split as split_module
from light_subtitle.pipeline.translate.split import (
    _PREPOSITIONS,
    _parse_batch_json,
    _texts_from_parent_chunks,
    _texts_match,
    split_overlong_units,
)


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


def _words_text(words: list[Word]) -> str:
    return " ".join(w.text.strip() for w in words)


def _words_from_text(text: str, *, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [_word(word, start + i * step, start + i * step + step * 0.8) for i, word in enumerate(text.split())]


def _seg(unit_id: str, words: list[Word], source_text: str | None = None) -> Segment:
    return Segment(
        unit_id=unit_id,
        start=words[0].start,
        end=words[-1].end,
        speaker="",
        source_text=source_text if source_text is not None else _words_text(words),
        words=words,
    )


def _config(*, api_key: str = "", max_duration: float = 7.0) -> SubtitleConfig:
    return SubtitleConfig(input_path="dummy.mp4", max_duration=max_duration, llm_api_key=api_key)


def _first_word(text: str) -> str:
    return text.strip().split()[0].lower().strip(string.punctuation)


def _last_word(text: str) -> str:
    return text.strip().split()[-1].lower().strip(string.punctuation)


def _assert_no_stranded_preposition_chunks(segments: list[Segment]) -> None:
    for index, segment in enumerate(segments):
        if index > 0:
            assert _first_word(segment.source_text) not in _PREPOSITIONS
        assert _last_word(segment.source_text) not in _PREPOSITIONS


# ═══════════════════════════════════════════════════════════
# _texts_match — exact match
# ═══════════════════════════════════════════════════════════


class TestTextsMatchExact:
    """Identical texts (after whitespace normalization)."""

    def test_identical(self):
        assert _texts_match("hello world", "hello world")

    def test_whitespace_normalized(self):
        """Extra spaces are collapsed."""
        assert _texts_match("hello   world", "hello world")
        assert _texts_match("  hello world  ", "hello world")

    def test_newlines_normalized(self):
        assert _texts_match("hello\nworld", "hello world")

    def test_tabs_normalized(self):
        assert _texts_match("hello\tworld", "hello world")


# ═══════════════════════════════════════════════════════════
# _texts_match — trailing punctuation tolerance
# ═══════════════════════════════════════════════════════════


class TestTextsMatchTrailingPunct:
    """LLM drops trailing sentence-end punctuation."""

    def test_trailing_period(self):
        """LLM drops final '.' — original has it, rejoined doesn't."""
        assert _texts_match("it was interesting.", "it was interesting")

    def test_trailing_period_reversed(self):
        """LLM adds final '.' — rejoined has it, original doesn't."""
        assert _texts_match("it was interesting", "it was interesting.")

    def test_trailing_exclamation(self):
        assert _texts_match("that's crazy!", "that's crazy")

    def test_trailing_question(self):
        assert _texts_match("you know what?", "you know what")

    def test_trailing_ellipsis(self):
        assert _texts_match("to be continued…", "to be continued")

    def test_trailing_chinese_period(self):
        assert _texts_match("一个完整的句子。", "一个完整的句子")

    def test_trailing_chinese_question(self):
        assert _texts_match("你知道吗？", "你知道吗")

    def test_trailing_chinese_exclamation(self):
        assert _texts_match("太好了！", "太好了")

    def test_multiple_trailing_punct(self):
        """Multiple trailing punct chars stripped."""
        assert _texts_match("really!!!", "really")
        assert _texts_match("what?!.", "what")

    def test_both_have_different_trailing_punct(self):
        """Original ends with '.', LLM ends with '!' — still match after strip."""
        assert _texts_match("hello.", "hello!")


# ═══════════════════════════════════════════════════════════
# _texts_match — mid-text punctuation tolerance
# ═══════════════════════════════════════════════════════════


class TestTextsMatchMidPunct:
    """LLM drops periods at sentence boundaries in the middle of text."""

    def test_mid_text_period_dropped(self):
        """Simulates mu0398_u0402: a trailing period + mid-text preserved."""
        assert _texts_match(
            "a lot. so for example human dexterity",
            "a lot so for example human dexterity",
        )

    def test_multiple_mid_periods_dropped(self):
        """Simulates mu0432_u0451: 3 mid-text periods dropped."""
        original = (
            "evolution has given us a lot. so for example human dexterity. all our ancestors needed great locomotion"
        )
        rejoined = (
            "evolution has given us a lot so for example human dexterity all our ancestors needed great locomotion"
        )
        assert _texts_match(original, rejoined)

    def test_question_mark_mid_text(self):
        assert _texts_match(
            "you know what? it's crazy",
            "you know what it's crazy",
        )

    def test_exclamation_mid_text(self):
        assert _texts_match(
            "that's amazing! really amazing",
            "that's amazing really amazing",
        )

    def test_punct_only_diffs(self):
        """All punct removed → texts are identical → match."""
        assert _texts_match(
            "hello. world! how? fine… great!",
            "hello world how fine great",
        )

    def test_cjk_with_spaces(self):
        """CJK punctuation with surrounding spaces → handled correctly."""
        assert _texts_match(
            "第一部分。 第二部分！ 第三部分？",
            "第一部分 第二部分 第三部分",
        )


# ═══════════════════════════════════════════════════════════
# _texts_match — known limitations
# ═══════════════════════════════════════════════════════════


class TestTextsMatchLimitations:
    """Known edge cases that are NOT handled.

    These are unlikely in practice because the source text being split
    is always English (Latin script with spaces between words).
    """

    def test_cjk_adjacent_no_space(self):
        """CJK punct between adjacent characters (no space) → false negative.

        ``"great。太棒了"`` after stripping ``。`` becomes ``"great太棒了"`` —
        the Latin and CJK characters merge because there's no whitespace
        boundary.  This is inherent to CJK scripts and doesn't occur with
        English source text.
        """
        assert not _texts_match(
            "great。太棒了",
            "great 太棒了",
        )


# ═══════════════════════════════════════════════════════════
# _texts_match — no match (real word changes)
# ═══════════════════════════════════════════════════════════


class TestTextsMatchNoMatch:
    """Genuine text differences beyond punctuation → should NOT match."""

    def test_different_word(self):
        assert not _texts_match("hello world", "hello wrld")

    def test_missing_word(self):
        assert not _texts_match("hello beautiful world", "hello world")

    def test_extra_word(self):
        assert not _texts_match("hello world", "hello beautiful world")

    def test_different_word_order(self):
        assert not _texts_match("hello world", "world hello")

    def test_completely_different(self):
        assert not _texts_match("hello world", "goodbye universe")

    def test_empty_vs_nonempty(self):
        assert not _texts_match("", "hello")

    def test_punct_not_only_diff(self):
        """Punct diff + word diff → no match."""
        assert not _texts_match(
            "hello. world foo",
            "hello world bar",
        )


# ═══════════════════════════════════════════════════════════
# Word-boundary fallback
# ═══════════════════════════════════════════════════════════


class TestWordBoundaryFallback:
    """Local fallback never cuts source text inside a word."""

    def test_p3_game_designer_sentence_splits_before_because(self):
        """Real p3 regression: never produce `everybody's t` / `hrowing`."""
        text = (
            "and it's always tricky as a game designer because constantly everybody's throwing ideas out in on a "
            "game team, like there's no shortage of ideas ever."
        )
        words = [
            _word(" and", 88.714, 88.935),
            _word(" it's", 90.178, 90.298),
            _word(" always", 90.338, 90.539),
            _word(" tricky", 90.619, 90.9),
            _word(" as", 90.94, 91.0),
            _word(" a", 91.02, 91.04),
            _word(" game", 91.16, 91.381),
            _word(" designer", 91.421, 91.863),
            _word(" because", 91.923, 92.204),
            _word(" constantly", 92.304, 92.947),
            _word(" throwing", 93.348, 93.649),
            _word(" ideas", 93.709, 94.151),
            _word(" out", 94.351, 94.512),
            _word(" in", 94.813, 94.913),
            _word(" on", 95.094, 95.174),
            _word(" a", 95.194, 95.214),
            _word(" game", 95.255, 95.495),
            _word(" team,", 95.536, 95.756),
            _word(" like", 95.776, 95.897),
            _word(" there's", 95.937, 96.118),
            _word(" no", 96.198, 96.358),
            _word(" shortage", 96.459, 96.639),
            _word(" of", 96.7, 96.74),
            _word(" ideas", 96.94, 97.301),
            _word(" ever.", 97.462, 97.683),
        ]

        result = split_overlong_units([_seg("mu0019_u0020", words, text)], _config())

        assert [s.source_text for s in result] == [
            "and it's always tricky as a game designer",
            "because constantly everybody's throwing ideas out in on a game team, "
            "like there's no shortage of ideas ever.",
        ]
        assert not any(s.source_text.endswith("everybody's t") for s in result)
        assert not any(s.source_text.startswith("hrowing") for s in result)
        _assert_no_stranded_preposition_chunks(result)

    def test_plain_word_boundary_fallback_when_no_natural_boundary(self):
        words = [
            _word("alpha", 0.0, 0.5),
            _word("bravo", 1.0, 1.5),
            _word("charlie", 2.0, 2.5),
            _word("delta", 3.0, 3.5),
            _word("echo", 4.0, 4.5),
            _word("foxtrot", 5.0, 5.5),
            _word("golf", 6.0, 6.5),
            _word("hotel", 7.0, 7.5),
            _word("india", 8.0, 8.5),
        ]

        result = split_overlong_units([_seg("plain", words)], _config(max_duration=4.0))

        assert len(result) > 1
        assert all(s.source_text == _words_text(s.words) for s in result)
        assert " ".join(s.source_text for s in result) == _words_text(words)
        _assert_no_stranded_preposition_chunks(result)

    def test_does_not_create_middle_chunk_starting_with_preposition(self):
        text = (
            "there was a mix of veterans and of this mix of veterans and then people like me joined the team "
            "during beta"
        )
        words = [_word(word, float(i), float(i) + 0.45) for i, word in enumerate(text.split())]

        result = split_overlong_units([_seg("prep_start", words, text)], _config(max_duration=5.0))

        assert len(result) > 1
        _assert_no_stranded_preposition_chunks(result)

    def test_does_not_create_chunk_ending_with_preposition(self):
        text = "we went to the store and then we continued working on the project for several more weeks"
        words = [_word(word, float(i), float(i) + 0.5) for i, word in enumerate(text.split())]

        result = split_overlong_units([_seg("prep_end", words, text)], _config(max_duration=4.0))

        assert len(result) > 1
        _assert_no_stranded_preposition_chunks(result)

    def test_sentence_initial_preposition_is_allowed_on_first_chunk(self):
        text = (
            "at a basic level you move use abilities from your action bar follow quests and complete them quickly today"
        )
        words = [_word(word, float(i), float(i) + 0.4) for i, word in enumerate(text.split())]

        result = split_overlong_units([_seg("prep_ok", words, text)], _config(max_duration=4.0))

        assert len(result) > 1
        assert _first_word(result[0].source_text) == "at"
        _assert_no_stranded_preposition_chunks(result)

    def test_repeated_you_boundary_uses_correct_source_occurrence(self):
        source_text = (
            "but if you really want to run a game like overwatch or world of warcraft successfully, "
            "you need master level engineers who have architected the client and server in such a way "
            "that you can hot fix the game on a dime."
        )
        expected = [
            "but if you really want to run a game like overwatch or world of warcraft successfully,",
            "you need master level engineers who have architected the client and server in such a way",
            "that you can hot fix the game on a dime.",
        ]
        word_chunks = [_words_from_text(text) for text in expected]

        assert _texts_from_parent_chunks(source_text, word_chunks) == expected

    def test_repeated_and_that_boundaries_use_correct_source_occurrences(self):
        source_text = (
            "because like i actually love story in games, and i counter that i'm the anti-story guy, "
            "and what i mean by that is like a the most magical stories "
            "that i've ever heard come out of video games are player stories"
        )
        expected = [
            "because like i actually love story in games,",
            "and i counter that i'm the anti-story guy,",
            "and what i mean by that is like a the most magical stories",
            "that i've ever heard come out of video games are player stories",
        ]
        word_chunks = [_words_from_text(text) for text in expected]

        assert _texts_from_parent_chunks(source_text, word_chunks) == expected

    def test_truncated_source_text_falls_back_to_word_text(self):
        source_text = (
            "but if you really want to run a game like overwatch or world of warcraft successfully, "
            "you need master leve"
        )
        expected = [
            "but if you really want to run a game like overwatch or world of warcraft successfully,",
            "you need master level engineers who have architected the client and server",
        ]
        word_chunks = [_words_from_text(text) for text in expected]

        assert _texts_from_parent_chunks(source_text, word_chunks) == expected


class TestLlmSinglePartFallback:
    """An LLM single-part response is not a valid split."""

    def test_batch_single_part_response_falls_back_to_local_split(self, monkeypatch):
        text = (
            "and it's always tricky as a game designer because constantly everybody's throwing ideas out in on a "
            "game team, like there's no shortage of ideas ever."
        )
        words = [
            _word(word, float(i), float(i) + 0.3)
            for i, word in enumerate(
                [
                    "and",
                    "it's",
                    "always",
                    "tricky",
                    "as",
                    "a",
                    "game",
                    "designer",
                    "because",
                    "constantly",
                    "everybody's",
                    "throwing",
                    "ideas",
                    "out",
                    "in",
                    "on",
                    "a",
                    "game",
                    "team,",
                    "like",
                    "there's",
                    "no",
                    "shortage",
                    "of",
                    "ideas",
                    "ever.",
                ]
            )
        ]

        calls = {"count": 0}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def chat(self, messages, temperature):
                calls["count"] += 1
                return json.dumps({"results": [{"id": "llm_single", "parts": [text]}]}), {}

        monkeypatch.setattr(split_module, "OpenAIClient", FakeClient)

        result = split_overlong_units([_seg("llm_single", words, text)], _config(api_key="test-key"))

        assert calls["count"] == 2
        assert len(result) > 1
        assert result[0].source_text == "and it's always tricky as a game designer"

    def test_batch_prompt_receives_duration_context(self, monkeypatch):
        text = (
            "and it's always tricky as a game designer because constantly everybody's throwing ideas out in on a "
            "game team, like there's no shortage of ideas ever."
        )
        words = [_word(word, float(i), float(i) + 0.3) for i, word in enumerate(text.split())]
        captured: dict[str, str] = {}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def chat(self, messages, temperature):
                captured["prompt"] = messages[0]["content"]
                return json.dumps({"results": [{"id": "duration_ctx", "parts": [text]}]}), {}

        monkeypatch.setattr(split_module, "OpenAIClient", FakeClient)

        split_overlong_units([_seg("duration_ctx", words, text)], _config(api_key="test-key", max_duration=7.0))

        assert '"duration":' in captured["prompt"]
        assert '"target_duration": 7.0' in captured["prompt"]

    def test_batch_single_part_retries_once_and_accepts_split(self, monkeypatch):
        text = "alpha bravo charlie delta echo foxtrot golf hotel"
        words = _words_from_text(text, step=1.0)
        responses = [
            {"results": [{"id": "retry_ok", "parts": [text]}]},
            {"results": [{"id": "retry_ok", "parts": ["alpha bravo charlie delta", "echo foxtrot golf hotel"]}]},
        ]

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def chat(self, messages, temperature):
                return json.dumps(responses.pop(0)), {}

        monkeypatch.setattr(split_module, "OpenAIClient", FakeClient)

        result = split_overlong_units([_seg("retry_ok", words, text)], _config(api_key="test-key", max_duration=4.0))

        assert [segment.source_text for segment in result] == [
            "alpha bravo charlie delta",
            "echo foxtrot golf hotel",
        ]
        assert responses == []

    def test_llm_preposition_boundary_is_rejected_for_fallback(self, monkeypatch):
        text = "there was a mix of veterans and then people like me joined the team during beta"
        words = _words_from_text(text, step=0.5)
        bad_response = {
            "results": [
                {
                    "id": "bad_prep",
                    "parts": ["there was a mix", "of veterans and then people like me joined the team during beta"],
                }
            ]
        }

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def chat(self, messages, temperature):
                return json.dumps(bad_response), {}

        monkeypatch.setattr(split_module, "OpenAIClient", FakeClient)

        result = split_overlong_units([_seg("bad_prep", words, text)], _config(api_key="test-key", max_duration=4.0))

        assert [segment.source_text for segment in result] != bad_response["results"][0]["parts"]
        _assert_no_stranded_preposition_chunks(result)


class TestComposeSplitBatchPrompt:
    """Batch prompt documents the constraints that protect split quality."""

    @staticmethod
    def _prompt_text() -> str:
        return Path(__file__).resolve().parents[3].joinpath("prompts/compose_split.j2").read_text()

    def test_batch_prompt_forbids_word_internal_splits(self):
        prompt = self._prompt_text()

        assert "duration > target_duration" in prompt
        assert "2 or more natural chunks" in prompt
        assert "Never split inside a word" in prompt
        assert "everybody's t" in prompt
        assert "hrowing ideas" in prompt
        assert "returning one part means the split task failed" in prompt

    def test_batch_prompt_documents_rejection_rules(self):
        prompt = self._prompt_text()

        assert "Hard Rejection Rules" in prompt
        assert "Never start a continuation chunk" in prompt
        assert "Never split coordinated noun/verb phrases" in prompt
        assert "client and server" in prompt
        assert "but if" in prompt
        assert "Never create a chunk that is only a conjunction" in prompt


class TestBatchJsonParsing:
    """LLM batch parsing tolerates common non-JSON wrappers."""

    def test_parse_plain_json_object(self):
        assert _parse_batch_json('{"results": []}') == {"results": []}

    def test_parse_markdown_code_fence(self):
        response = '```json\n{"results": [{"id": "u1", "parts": ["a", "b"]}]}\n```'
        assert _parse_batch_json(response) == {"results": [{"id": "u1", "parts": ["a", "b"]}]}

    def test_parse_leading_commentary(self):
        response = 'Here is the JSON:\n{"results": [{"id": "u1", "parts": ["a", "b"]}]}'
        assert _parse_batch_json(response) == {"results": [{"id": "u1", "parts": ["a", "b"]}]}

    def test_reject_non_object_json(self):
        assert _parse_batch_json("[]") is None
