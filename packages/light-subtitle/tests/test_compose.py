"""Tests for compose_segments — merging fragmentary audio segments into
complete translation units."""

from light_models import Segment, Word
from light_subtitle.language import is_sentence_end
from light_subtitle.pipeline.translate.compose import compose_segments


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


def _seg(
    unit_id: str,
    start: float,
    end: float,
    text: str,
    speaker: str = "S1",
    words: list[Word] | None = None,
) -> Segment:
    if words is None:
        words = [_word(text.split()[0], start, end)]
    return Segment(
        unit_id=unit_id,
        start=start,
        end=end,
        speaker=speaker,
        source_text=text,
        words=words,
    )


# ═══════════════════════════════════════════════════════════
# Basic fragment merging
# ═══════════════════════════════════════════════════════════


class TestFragmentMerge:
    """Fragments (no sentence-ending punct) merge with the next segment."""

    def test_fragment_merged_with_complete(self):
        """Fragment followed by complete → merged."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "These are not just"),
                _seg("u0001", 1.2, 3.0, "coding assistants."),
            ]
        )
        assert len(result) == 1
        assert result[0].unit_id == "mu0000_u0001"
        assert result[0].source_text == "These are not just coding assistants."

    def test_complete_stays_separate(self):
        """Complete sentence followed by another → no merge."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.5, "That is correct."),
                _seg("u0001", 2.0, 3.5, "This is new."),
            ]
        )
        assert len(result) == 2
        assert result[0].unit_id == "mu0000_u0000"
        assert result[1].unit_id == "mu0001_u0001"

    def test_three_fragments_merged(self):
        """Three consecutive fragments → single merged unit."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 0.5, "First"),
                _seg("u0001", 0.7, 1.0, "second part"),
                _seg("u0002", 1.2, 2.0, "third part here."),
            ]
        )
        assert len(result) == 1
        assert "First second part third part here." in result[0].source_text


# ═══════════════════════════════════════════════════════════
# Short fragment auto-merge (≤ 3 words, no sentence end)
# ═══════════════════════════════════════════════════════════


class TestShortFragmentAutoMerge:
    """Very short fragments (≤ 3 words, no sentence-end) always merge."""

    def test_well_merged(self):
        """'Well,' (1 word, comma) → merge forward."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 0.3, "Well,"),
                _seg("u0001", 0.8, 2.0, "the agents are working."),
            ]
        )
        assert len(result) == 1

    def test_short_fragment_not_sentence_end(self):
        """3 words, no punct → merge despite prev being complete."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "I agree."),
                _seg("u0001", 1.5, 2.0, "So then"),
                _seg("u0002", 2.2, 4.0, "we should proceed."),
            ]
        )
        assert len(result) == 1
        assert "So then we should proceed." in result[0].source_text

    def test_short_but_sentence_end_stays(self):
        """3 words ending with . → NOT a fragment, stays separate."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "I agree."),
                _seg("u0001", 1.5, 2.5, "Right."),
                _seg("u0002", 3.0, 5.0, "Let's go."),
            ]
        )
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════
# Abbreviation handling
# ═══════════════════════════════════════════════════════════


class TestAbbreviationHandling:
    """Abbreviations like 'U.S.' should not be treated as sentence end."""

    def test_us_abbreviation_is_fragment(self):
        """'U.S.' ending a segment → fragment, merges forward."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "U.S."),
                _seg("u0001", 1.2, 2.5, "economy is growing."),
            ]
        )
        assert len(result) == 1
        assert "U.S. economy" in result[0].source_text

    def test_eg_abbreviation_is_fragment(self):
        """'e.g.' ending a segment → fragment."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "tools e.g."),
                _seg("u0001", 1.2, 3.0, "whisper and ffmpeg."),
            ]
        )
        assert len(result) == 1
        assert "e.g. whisper" in result[0].source_text

    def test_ie_abbreviation_is_fragment(self):
        """'i.e.' ending a segment → fragment."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "ASR i.e."),
                _seg("u0001", 1.2, 3.0, "speech to text."),
            ]
        )
        assert len(result) == 1

    def test_abbreviation_mid_sentence_kept_with_next(self):
        """'e.g.' followed by lowercase → abbreviation, merge."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "use tools e.g."),
                _seg("u0001", 1.2, 3.0, "whisper for ASR."),
            ]
        )
        assert len(result) == 1

    def test_abbreviation_followed_by_uppercase_still_abbrev(self):
        """'U.S.' followed by uppercase 'These' → still abbreviation."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "in the U.S."),
                _seg("u0001", 1.2, 3.0, "These are the rules."),
            ]
        )
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════
# Temporal / speaker boundaries
# ═══════════════════════════════════════════════════════════


class TestBoundaries:
    """Gap and speaker checks prevent inappropriate merges."""

    def test_large_gap_prevents_merge(self):
        """Gap > 2.0s → separate thoughts, no merge."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "first part"),
                _seg("u0001", 5.0, 6.0, "late continuation."),
            ]
        )
        assert len(result) == 2

    def test_speaker_change_prevents_merge(self):
        """Different speakers → no merge."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "I think so.", speaker="S1"),
                _seg("u0001", 1.5, 2.5, "I disagree.", speaker="S2"),
            ]
        )
        assert len(result) == 2

    def test_speaker_change_but_short_does_not_merge(self):
        """Short fragment but speaker differs → keep separate."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "I think that is correct.", speaker="S1"),
                _seg("u0001", 1.5, 1.8, "But", speaker="S2"),
                _seg("u0002", 2.0, 3.0, "we must verify.", speaker="S2"),
            ]
        )
        assert len(result) == 2  # S1 complete, S2 fragment+complete


# ═══════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════


class TestEdgeCases:
    """Empty input, single segment, duration limits."""

    def test_empty_input(self):
        """Empty list → empty output."""
        assert compose_segments([]) == []

    def test_single_segment(self):
        """Single segment → unchanged."""
        seg = _seg("u0000", 0.0, 1.0, "Hello world.")
        result = compose_segments([seg])
        assert len(result) == 1
        assert result[0].unit_id == "mu0000_u0000"
        assert result[0].source_text == "Hello world."

    def test_single_fragment_segment(self):
        """Single fragment segment → kept as-is."""
        seg = _seg("u0000", 0.0, 1.0, "incomplete thought")
        result = compose_segments([seg])
        assert len(result) == 1

    def test_all_complete_stays_separate(self):
        """All complete sentences → no merge."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "First sentence."),
                _seg("u0001", 1.5, 2.5, "Second sentence."),
                _seg("u0002", 3.0, 4.0, "Third sentence."),
            ]
        )
        assert len(result) == 3

    def test_short_fragment_auto_merges_after_complete(self):
        """Short fragment (≤3 words, no sentence-end) after a complete
        sentence auto-merges forward."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 1.0, "first part"),
                _seg("u0001", 1.2, 2.0, "second part. Done."),
                _seg("u0002", 2.5, 3.0, "next thought"),
                _seg("u0003", 3.2, 4.0, "finished here."),
            ]
        )
        # "next thought" (2 words, no sentence-end) triggers auto-merge
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════
# Duration / size limits
# ═══════════════════════════════════════════════════════════


class TestSemanticAccumulation:
    """New algorithm: accumulate fragments until complete sentence."""

    def test_fragments_merge_across_duration(self):
        """Fragments merge even with long combined duration."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 6.0, "long first part that keeps going"),
                _seg("u0001", 6.2, 12.5, "still continuing but long"),
                _seg("u0002", 12.7, 14.0, "past the limit."),
            ]
        )
        # All fragments merge into one complete sentence.
        assert len(result) == 1
        assert "past the limit." in result[0].source_text
        assert result[0].unit_id == "mu0000_u0002"

    def test_complete_sentence_stops_accumulation(self):
        """A complete sentence stops the buffer."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 3.0, "This is complete."),
                _seg("u0001", 3.5, 5.0, "Next sentence."),
            ]
        )
        assert len(result) == 2

    def test_fragment_chain_until_complete(self):
        """Multiple fragments chain until a complete sentence appears."""
        result = compose_segments(
            [
                _seg("u0000", 0.0, 2.0, "first fragment"),
                _seg("u0001", 2.2, 4.0, "second fragment"),
                _seg("u0002", 4.2, 6.0, "final complete sentence."),
            ]
        )
        assert len(result) == 1
        assert "first fragment second fragment final complete sentence." in result[0].source_text

    def test_ilya_example_fixed(self):
        """The motivating example: transition from ... scaling."""
        result = compose_segments(
            [
                _seg("u0373", 1359.951, 1361.0, "we've already witnessed a transition from"),
                _seg(
                    "u0374",
                    1361.5,
                    1363.76,
                    "one type of scale into a different type of scaling from pre-training to rl.",
                ),
            ]
        )
        assert len(result) == 1
        assert "transition from one type of scale" in result[0].source_text


# ═══════════════════════════════════════════════════════════
# Real-world scenario: Naval podcast
# ═══════════════════════════════════════════════════════════


class TestRealWorldNavalPodcast:
    """The exact example that motivated this feature."""

    def test_naval_podcast_snippet(self):
        """Mid-sentence fragments merge into complete thoughts."""
        result = compose_segments(
            [
                _seg("u0039", 113.106, 113.307, "Well,"),
                _seg("u0040", 114.209, 115.393, "the agents are really working."),
                _seg("u0041", 115.473, 116.075, "These are not just"),
                _seg("u0042", 116.716, 119.950, "coding assists now where you ask it to solve a specific problem."),
                _seg("u0043", 120.010, 121.860, "It gives you a pile of code, and then you cut and paste that"),
                _seg("u0044", 122.381, 124.908, "into your IDE, your development environment."),
                _seg(
                    "u0045",
                    125.751,
                    130.145,
                    "Rather, you open up a terminal, CLI, as I call it, the command line interface.",
                ),
            ]
        )
        assert len(result) == 4

        # 1. "Well, the agents are really working."
        assert "Well, the agents are really working." in result[0].source_text
        assert result[0].unit_id == "mu0039_u0040"

        # 2. "These are not just coding assists now where you ask it to
        #    solve a specific problem."
        assert "These are not just" in result[1].source_text
        assert "coding assists now" in result[1].source_text
        assert result[1].unit_id == "mu0041_u0042"

        # 3. "It gives you a pile of code, and then you cut and paste that
        #    into your IDE, your development environment."
        assert "It gives you a pile of code" in result[2].source_text
        assert "cut and paste that into your IDE" in result[2].source_text
        assert result[2].unit_id == "mu0043_u0044"

        # 4. "Rather, you open up a terminal..." (complete → standalone)
        assert "Rather, you open up a terminal" in result[3].source_text
        assert result[3].unit_id == "mu0045_u0045"


# ═══════════════════════════════════════════════════════════
# Discourse fillers (Hmm., Um., etc.)
# ═══════════════════════════════════════════════════════════


class TestDiscourseFillers:
    """Fillers ending with '.' must be sentence ends, not abbreviations."""

    def test_fillers_are_sentence_ends(self):
        for text in ("Hmm.", "Um.", "Uh.", "Mm.", "Er.", "Ah.", "Oh."):
            assert is_sentence_end(text), f"{text!r} should be a sentence end"

    def test_hmm_stays_separate_after_question(self):
        """Lex question + Joscha 'Hmm.' + answer must not chain-merge."""
        result = compose_segments(
            [
                _seg("u0246", 877.823, 878.68, "What do you understand as life?"),
                _seg("u0247", 878.68, 878.762, "Hmm."),
                _seg(
                    "u0248",
                    879.821,
                    882.479,
                    "Entities of sufficiently high complexity that are full of surprises.",
                ),
            ]
        )
        assert len(result) == 3
        assert result[0].unit_id == "mu0246_u0246"
        assert result[1].unit_id == "mu0247_u0247"
        assert result[1].source_text == "Hmm."
        assert result[2].unit_id == "mu0248_u0248"
        assert "Entities of sufficiently high complexity" in result[2].source_text
