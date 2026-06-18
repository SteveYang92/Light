"""English-specific text processing — Netflix §4 grammar, line-breaking, and break finding.

This is the primary language module. All English constants, helpers, and
classes live here.  Import from ``language.english``::

    from light_subtitle.language.english import EnglishBreakFinder, _greedy_fill_with_grammar
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from light_models import SubtitleCue
from light_models.punctuation import CLAUSE_PUNCT, EN_TRAILING_PUNCT, SENTENCE_ENDS

from .base import (
    BREAK_CLAUSE,
    BREAK_CONJUNCTION,
    BREAK_FALLBACK,
    BREAK_PREP_ARTICLE,
    BREAK_SENTENCE_END,
    BreakFinder,
    is_sentence_end,
)

if TYPE_CHECKING:
    from light_subtitle.config import SubtitleConfig

# ═══════════════════════════════════════════════════════════════════
# Netflix §4 word class sets (canonical versions)
# ═══════════════════════════════════════════════════════════════════

ARTICLES = {"a", "an", "the"}

SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "who", "what"}

OBJECT_PRONOUNS = {"me", "him", "her", "us", "them", "it"}

NEGATION = {
    "not",
    "never",
    "no",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "can't",
    "couldn't",
    "shouldn't",
    "wouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
}

AUXILIARIES = {
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
}

PHRASAL_ROOTS = {
    "give",
    "look",
    "take",
    "put",
    "get",
    "come",
    "go",
    "bring",
    "turn",
    "set",
    "run",
    "work",
    "find",
    "break",
    "pick",
    "carry",
    "hold",
    "keep",
    "let",
    "make",
    "show",
    "call",
    "check",
    "cut",
    "drop",
    "fall",
    "fill",
    "grow",
    "hang",
    "pull",
    "push",
    "shut",
    "sit",
    "stand",
    "think",
    "throw",
    "wake",
    "walk",
    "write",
    "build",
    "start",
    "move",
    "pass",
    "hand",
    "point",
    "send",
    "pay",
    "lay",
    "step",
    "back",
}

PHRASAL_PARTICLES = {
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "away",
    "at",
    "for",
    "over",
    "through",
    "into",
    "around",
    "about",
    "back",
    "forward",
    "aside",
    "along",
    "ahead",
    "apart",
    "together",
}

CONJUNCTIONS = {
    "and",
    "but",
    "or",
    "nor",
    "yet",
    "so",
    "because",
    "while",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "whether",
    "if",
    "although",
    "though",
    "unless",
    "until",
    "since",
    "as",
    "than",
    "whereas",
    "except",
    "however",
    "then",
    "also",
    "still",
}

PREPOSITIONS = {
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "from",
    "by",
    "about",
    "into",
    "through",
    "over",
    "under",
    "after",
    "before",
    "between",
    "during",
    "without",
    "within",
    "upon",
    "across",
    "along",
    "around",
    "behind",
    "beyond",
    "down",
    "off",
    "out",
    "up",
    "toward",
    "towards",
    "against",
    "among",
    "amongst",
    "beside",
    "besides",
    "above",
    "below",
    "near",
    "inside",
    "outside",
    "beneath",
    "underneath",
    "notwithstanding",
}

DISCOURSE_MARKERS = {
    "so",
    "now",
    "well",
    "anyway",
    "ok",
    "okay",
    "anyways",
    "anyhow",
    "look",
    "listen",
    "right",
    "alright",
}

ADJECTIVE_SUFFIXES = {
    "y",
    "ful",
    "less",
    "ive",
    "ous",
    "al",
    "able",
    "ible",
    "ent",
    "ant",
    "ic",
    "ish",
    "ary",
}

# ═══════════════════════════════════════════════════════════════════
# Netflix §4 helpers
# ═══════════════════════════════════════════════════════════════════


def _is_adjective(word: str) -> bool:
    """True if *word* likely functions as an adjective (suffix-based)."""
    w = word.lower().rstrip(EN_TRAILING_PUNCT)
    for suf in ADJECTIVE_SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 1:
            return True
    return False


def _is_phrasal_verb_break(word_a: str, word_b: str) -> bool:
    """True if splitting between *word_a* and *word_b* separates a phrasal verb."""
    root = word_a.lower().rstrip(EN_TRAILING_PUNCT)
    particle = word_b.lower().rstrip(EN_TRAILING_PUNCT)
    return root in PHRASAL_ROOTS and particle in PHRASAL_PARTICLES


def _violates_rule1(words: list[str], break_pos: int) -> bool:
    """Check if break *after* words[break_pos] violates Netflix §4."""
    if break_pos < 0 or break_pos + 1 >= len(words):
        return False
    a = words[break_pos]
    b = words[break_pos + 1]
    a_clean = a.lower().rstrip(EN_TRAILING_PUNCT)
    b_clean = b.lower().rstrip(EN_TRAILING_PUNCT)
    if a_clean in ARTICLES or _is_adjective(a):
        return True
    if (
        a
        and a[0].isupper()
        and b
        and b[0].isupper()
        and a.rstrip(EN_TRAILING_PUNCT).isalpha()
        and b.rstrip(EN_TRAILING_PUNCT).isalpha()
    ):
        return True
    if a_clean in AUXILIARIES or a_clean in SUBJECT_PRONOUNS or a_clean in NEGATION:
        return True
    if _is_phrasal_verb_break(a, b):
        return True
    if b_clean in OBJECT_PRONOUNS:
        return True
    if a.rstrip(EN_TRAILING_PUNCT).replace(",", "").replace(".", "").isdigit():
        return True
    return False


def _strip_punct(word: str) -> str:
    return word.lower().rstrip(EN_TRAILING_PUNCT)


def _is_forbidden_split(last_word: str, next_word: str) -> bool:
    """Check if splitting between *last_word* and *next_word* would create an unreadable fragment.

    Conservative version for segment boundaries — only blocks splits that
    produce clearly orphaned fragments (article alone, auxiliary alone, etc.).
    Sentence-ending punctuation always overrides this check.
    """
    a_clean = _strip_punct(last_word)
    b_clean = _strip_punct(next_word)

    if not a_clean or not b_clean:
        return False

    # Clause or sentence-ending punctuation overrides all forbidden checks.
    stripped = last_word.strip()
    if any(ch in CLAUSE_PUNCT for ch in stripped) or is_sentence_end(stripped):
        return False

    if a_clean in ARTICLES:
        return True
    if a_clean in AUXILIARIES:
        return True
    if a_clean in SUBJECT_PRONOUNS:
        return True
    if a_clean in NEGATION:
        return True
    if _is_phrasal_verb_break(last_word, next_word):
        return True
    if b_clean in OBJECT_PRONOUNS:
        return True
    if last_word.rstrip(EN_TRAILING_PUNCT).replace(",", "").replace(".", "").isdigit():
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# EnglishBreakFinder
# ═══════════════════════════════════════════════════════════════════


class EnglishBreakFinder(BreakFinder):
    """Break finder for English text.

    Forbidden positions: any split that violates Netflix §4
    (article→noun, auxiliary→verb, phrasal verb→particle, etc.).
    Scoring is word-based: higher score = better place to break
    before word *b*.
    """

    def __init__(self, words: list[str]):
        super().__init__(" ".join(words))
        self.words = words

    def _word_before(self, pos: int) -> str:
        offset = 0
        for w in self.words:
            next_offset = offset + len(w)
            if pos <= next_offset:
                return w
            offset = next_offset + 1
        return ""

    def _word_after(self, pos: int) -> str:
        offset = 0
        for w in self.words:
            if offset > pos:
                return w
            offset += len(w) + 1
        return ""

    def _word_index_after(self, pos: int) -> int:
        offset = 0
        for i, w in enumerate(self.words):
            if offset > pos:
                return i
            offset += len(w) + 1
        return len(self.words)

    def is_forbidden(self, pos: int) -> bool:
        wb = self._word_before(pos)
        wa = self._word_after(pos)
        if not wb or not wa:
            return False
        a = wb.lower().rstrip(EN_TRAILING_PUNCT)
        b = wa.lower().rstrip(EN_TRAILING_PUNCT)
        if a in ARTICLES:
            return True
        if _is_adjective(wb):
            return True
        if (
            wb
            and wb[0].isupper()
            and wa
            and wa[0].isupper()
            and wb.rstrip(EN_TRAILING_PUNCT).isalpha()
            and wa.rstrip(EN_TRAILING_PUNCT).isalpha()
        ):
            return True
        if a in AUXILIARIES:
            return True
        if a in SUBJECT_PRONOUNS:
            return True
        if a in NEGATION:
            return True
        if _is_phrasal_verb_break(wb, wa):
            return True
        if b in OBJECT_PRONOUNS:
            return True
        if wb.rstrip(EN_TRAILING_PUNCT).replace(",", "").replace(".", "").isdigit():
            return True
        return False

    def score(self, pos: int) -> int:
        score_val = BREAK_FALLBACK
        text = self.text

        if 0 <= pos < len(text):
            ch = text[pos]
            if ch in SENTENCE_ENDS:
                score_val = max(score_val, BREAK_SENTENCE_END)
            elif ch in CLAUSE_PUNCT or (pos >= 2 and text[pos - 2 : pos + 1] == "..."):
                score_val = max(score_val, BREAK_CLAUSE)

        wa = self._word_after(pos)
        if wa:
            wa_clean = wa.lower().rstrip(EN_TRAILING_PUNCT)
            if wa_clean in CONJUNCTIONS:
                score_val = max(score_val, BREAK_CONJUNCTION)
            elif wa_clean in PREPOSITIONS:
                score_val = max(score_val, BREAK_PREP_ARTICLE)
            elif wa_clean in ARTICLES:
                score_val = max(score_val, BREAK_PREP_ARTICLE - 10)

        return score_val


# ═══════════════════════════════════════════════════════════════════
# English line-breaking (subtitle layout)
# ═══════════════════════════════════════════════════════════════════


def _greedy_fill_with_grammar(words: list[str], max_chars: int) -> list[str]:
    """Fill lines greedily while respecting Netflix §4 grammar rules.

    When adding the next word would overflow, check grammar — if it
    violates 'should not separate', push the problematic pair to the next line.
    """
    if not words:
        return []
    lines = []
    start = 0
    while start < len(words):
        current = words[start]
        end = start + 1
        while end < len(words) and len(current) + 1 + len(words[end]) <= max_chars:
            current = current + " " + words[end]
            end += 1
        if end == len(words):
            lines.append(current)
            break
        if _violates_rule1(words[:end] + [words[end]], end - 1):
            current = " ".join(words[start : end - 1])
            end -= 1
        lines.append(current)
        start = end
    return lines


def _merge_lines(lines: list[str], max_chars: int) -> list[str]:
    """Merge adjacent short English lines when combined fits in one line."""
    if len(lines) <= 2:
        return lines
    merged = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and len(lines[i]) + len(lines[i + 1]) + 1 <= max_chars:
            merged.append(lines[i] + " " + lines[i + 1])
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    return merged


def _balance_english_pair(lines: list[str]) -> list[str]:
    """Move a word from line 1 to line 2 to avoid orphaned second lines."""
    if len(lines) != 2:
        return lines
    words1 = lines[0].split()
    words2 = lines[1].split()
    if len(words2) >= 3 or len(words1) < 4:
        return lines
    moved = words1[-1]
    new_line1 = " ".join(words1[:-1])
    new_line2 = moved + " " + " ".join(words2)
    return [new_line1, new_line2]


def _make_english_cue(original: SubtitleCue, lines: list[str]) -> SubtitleCue:
    """Create a single display cue from 1+ English lines."""
    return SubtitleCue(
        cue_id=original.cue_id,
        unit_id=original.unit_id,
        start=original.start,
        end=original.end,
        text="\n".join(lines),
        lang=original.lang,
        speaker=original.speaker,
        words=list(original.words) if original.words else [],
        merged_from=list(original.merged_from),
    )


def _build_english_cues(
    original: SubtitleCue,
    lines: list[str],
    max_lines: int,
    import_UnitWordIndex: type | None = None,
    import_chunk_times: type | None = None,
) -> list[SubtitleCue]:
    """Build one or more cues from English lines, splitting on max_lines.

    Multi-cue path uses lazy imports for _UnitWordIndex and _chunk_times
    to avoid circular dependencies.
    """
    if import_UnitWordIndex is None:
        from light_subtitle.pipeline.subtitle._word_index import (  # noqa: PLC0415
            _UnitWordIndex,
        )

        import_UnitWordIndex = _UnitWordIndex
    if import_chunk_times is None:
        from light_subtitle.pipeline.subtitle._word_index import (  # noqa: PLC0415
            _chunk_times,
        )

        import_chunk_times = _chunk_times

    if not lines:
        return [original]

    balanced: list[str] = []
    for i in range(0, len(lines), max_lines):
        chunk = lines[i : i + max_lines]
        if len(chunk) == max_lines:
            chunk = _balance_english_pair(chunk)
        balanced.extend(chunk)

    if len(balanced) <= max_lines:
        return [_make_english_cue(original, balanced)]

    word_idx = import_UnitWordIndex.from_words(original.words or [])
    en_cps = 25

    chunk_times_list: list[tuple[float, float]] = []
    for i in range(0, len(balanced), max_lines):
        chunk = balanced[i : i + max_lines]
        cs, ce = import_chunk_times(chunk, original, word_idx, en_cps)
        chunk_times_list.append((cs, ce))

    for ci in range(len(chunk_times_list) - 1):
        if chunk_times_list[ci][1] > chunk_times_list[ci + 1][0]:
            cs, _ = chunk_times_list[ci]
            next_cs, _ = chunk_times_list[ci + 1]
            capped_end = max(cs + 0.8, min(chunk_times_list[ci][1], next_cs))
            chunk_times_list[ci] = (cs, capped_end)

    cues: list[SubtitleCue] = []
    for ci, (chunk_start, chunk_end) in enumerate(chunk_times_list):
        i = ci * max_lines
        chunk = balanced[i : i + max_lines]
        if i + max_lines >= len(balanced):
            chunk_end = max(chunk_end, original.end)
        chunk_words = word_idx.find_words(chunk) if word_idx else []
        cue_idx = i // max_lines
        cues.append(
            SubtitleCue(
                cue_id=f"{original.cue_id}_{cue_idx}" if cue_idx > 0 else original.cue_id,
                unit_id=original.unit_id,
                start=chunk_start,
                end=chunk_end,
                text="\n".join(chunk),
                lang=original.lang,
                speaker=original.speaker,
                words=chunk_words,
                merged_from=list(original.merged_from),
            )
        )
    return cues


def split_english(cue: SubtitleCue, text: str, config: SubtitleConfig) -> list[SubtitleCue]:
    """Split English text into display-ready cues.

    Grammar-aware greedy fill, merge short adjacent lines, then build
    one or more cues.  When line count exceeds config.max_lines, lines
    are chunked into multiple cues with word-aligned timing.
    """
    max_chars = config.max_chars_per_line_en
    max_lines = config.max_lines
    lines: list[str] = []

    if "\n" in text:
        for pl in text.split("\n"):
            if not pl.strip():
                continue
            words = pl.strip().split()
            if not words:
                continue
            sub = _greedy_fill_with_grammar(words, max_chars)
            sub = _merge_lines(sub, max_chars)
            lines.extend(sub)
    else:
        words = text.split()
        if words:
            lines = _greedy_fill_with_grammar(words, max_chars)
            lines = _merge_lines(lines, max_chars)

    if lines:
        return _build_english_cues(cue, lines, max_lines)
    return [_make_english_cue(cue, [text])]


# ═══════════════════════════════════════════════════════════════════
# English cross-cue repair
# ═══════════════════════════════════════════════════════════════════


def mend_split_names(cues: list[SubtitleCue]) -> None:
    """Heal cues where a proper name was split across boundaries (e.g. "Yann" | "LeCun")."""
    if len(cues) < 2:
        return

    for i in range(len(cues) - 1):
        curr = cues[i]
        nxt = cues[i + 1]

        words_curr = curr.text.split()
        words_nxt = nxt.text.split()
        if not words_curr or not words_nxt:
            continue

        last_word = words_curr[-1].rstrip("\n")

        if (
            last_word
            and last_word[0].isupper()
            and last_word.isascii()
            and len(last_word) <= 6
            and not any(ch in EN_TRAILING_PUNCT for ch in last_word)
        ):
            first_word_next = words_nxt[0].rstrip("\n")
            if (
                first_word_next
                and first_word_next[0].isupper()
                and first_word_next.isascii()
                and not any(ch in EN_TRAILING_PUNCT for ch in first_word_next)
            ):
                words_curr[-1] = last_word + first_word_next
                curr.text = " ".join(words_curr)
                words_nxt = words_nxt[1:]
                nxt.text = " ".join(words_nxt) if words_nxt else ""
