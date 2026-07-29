"""Normalise the word stream — speaker smoothing, gap calculation, sentence-final detection."""

from __future__ import annotations

import re
from dataclasses import replace

from light_models import Word

from .lexicon import ABBREV, CLAUSE_PUNCT
from .types import NWord, nword_core, nword_tail

_INITIAL_NAME_RE = re.compile(r"^[A-Z]\.$")  # "J.", "K." — name initials
_NUMBER_RE = re.compile(r"^\d+\.(\d+)?$")  # "3.", "3.14"
_UNKNOWN_SPEAKER = "UNKNOWN"


def normalize(words: list[Word]) -> list[NWord]:
    """Normalise a word stream: fill speakers, compute gaps, classify tails."""
    words = _smooth_speakers(words)
    n = len(words)
    result: list[NWord] = []
    for i, w in enumerate(words):
        gap = words[i + 1].start - w.end if i + 1 < n else 0.0
        if gap < 0:
            gap = 0.0
        tail_punct = nword_tail(w.text)
        result.append(
            NWord(
                idx=i,
                text=w.text,
                core=nword_core(w.text),
                start=w.start,
                end=w.end,
                confidence=w.confidence,
                speaker=w.speaker or _UNKNOWN_SPEAKER,
                gap_after=gap,
                tail_punct=tail_punct,
                is_sentence_final=_is_sentence_final(w, words, i, n, gap),
                is_clause_punct=tail_punct in CLAUSE_PUNCT,
            )
        )
    return result


def _smooth_speakers(words: list[Word]) -> list[Word]:
    """Absorb speaker jitter: short runs flanked by the same speaker are merged."""
    max_runs = 3
    for _ in range(max_runs):
        changed = False
        runs: list[tuple[int, int, str]] = []  # (start, end_exclusive, speaker)
        i = 0
        while i < len(words):
            sp = words[i].speaker or _UNKNOWN_SPEAKER
            j = i + 1
            while j < len(words) and (words[j].speaker or _UNKNOWN_SPEAKER) == sp:
                j += 1
            runs.append((i, j, sp))
            i = j
        if len(runs) <= 1:
            break
        for k in range(1, len(runs) - 1):
            left_sp = runs[k - 1][2]
            right_sp = runs[k + 1][2]
            if left_sp != right_sp:
                continue
            s, e, this_sp = runs[k]
            nw = e - s
            dur = words[e - 1].end - words[s].start
            if nw <= 2 and dur < 1.0:
                for idx in range(s, e):
                    words[idx] = replace(words[idx], speaker=left_sp)
                changed = True
        if not changed:
            break
    return words


def _is_sentence_final(w: Word, words: list[Word], i: int, n: int, gap: float) -> bool:
    """True when *w* ends with . ! ? and is not an abbreviation, name initial, or number."""
    text = w.text.strip()
    if not text:
        return False
    last = text[-1]
    if last not in ".!?":
        return False
    if last in "!?":
        return True
    # '.' — apply veto conditions
    core = nword_core(text)
    if core in ABBREV:
        return False
    if _INITIAL_NAME_RE.match(text):
        return False
    if _NUMBER_RE.match(text):
        return False
    # '.': short abbreviation/acronym (all-caps, 2-5 chars) + uppercase next + small gap
    if i + 1 < n:
        next_word = words[i + 1].text.strip()
        upper_next = next_word and next_word[0].isupper() and gap < 0.50
        looks_abbrev = len(core) <= 5 and text.rstrip(".").isupper()
        if upper_next and looks_abbrev:
            return False
        if next_word and next_word[0].islower() and gap < 0.30:
            return False
    return True
