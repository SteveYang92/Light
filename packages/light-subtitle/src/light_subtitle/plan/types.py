"""NWord — normalised word used only by normalize."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NWord:
    idx: int
    text: str
    core: str
    start: float
    end: float
    confidence: float
    speaker: str
    gap_after: float
    tail_punct: str
    is_sentence_final: bool
    is_clause_punct: bool


def nword_core(text: str) -> str:
    s = text.strip().lower()
    while s and s[-1] in ".,!?;:—\"'()[]“”‘’":
        s = s[:-1]
    return s


def nword_tail(text: str) -> str:
    s = text.strip()
    tail: list[str] = []
    for ch in reversed(s):
        if ch in ".,!?;:—":
            tail.append(ch)
        else:
            break
    return "".join(reversed(tail))
