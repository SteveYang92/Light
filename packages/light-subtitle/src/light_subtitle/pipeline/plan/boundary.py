"""Shared boundary contract for cue splitting (LLM validator + fallback).

One word list, two consumers: ``planner._break_problems`` rejects LLM
breaks that strand a function word, and ``fallback._best_cut`` avoids
creating such boundaries in the first place.  Keeping the list here
guarantees the LLM contract and the deterministic insurance path agree
on what a "clean" boundary is.
"""

from __future__ import annotations

from light_models import Word

# Function words that must not end a split part: cutting right after them
# strands the word from what it attaches to ("…or if you are | …").
# Object-capable pronouns (it, me, him, us, them) are excluded: they often
# legitimately end a clause.  A trailing punctuation mark on the word marks
# a clause boundary and always exempts it.
DANGLING_TAIL_WORDS = frozenset(
    {
        # articles / determiners / possessives
        "a", "an", "the", "this", "these", "those", "each", "every", "some",
        "my", "your", "his", "our", "their", "its",
        # subject pronouns
        "i", "you", "he", "she", "we", "they", "who",
        # auxiliaries / copulas / modals
        "am", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "don't", "doesn't", "didn't",
        "have", "has", "had", "haven't", "hasn't",
        "can", "can't", "could", "couldn't", "will", "won't", "would", "wouldn't",
        "shall", "should", "shouldn't", "may", "might", "must",
        # prepositions
        "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto",
        "about", "over", "under", "between", "through", "during", "without",
        "within", "than", "as",
        # conjunctions / subordinators / relatives
        "and", "or", "but", "nor", "if", "because", "when", "while", "although",
        "though", "unless", "until", "whether", "that", "which", "whose", "where",
    }
)  # fmt: skip

CLAUSE_PUNCT_TAIL = tuple(".,!?;:…，。？！；：、")


def dangling_tail(word: Word) -> str | None:
    """Return the offending token when a part must not end on *word*."""
    token = word.text.strip()
    if not token or token.endswith(CLAUSE_PUNCT_TAIL):
        return None
    bare = token.strip("\"'()[]“”‘’").lower()
    return bare if bare in DANGLING_TAIL_WORDS else None


def ends_with_clause_punct(word: Word) -> bool:
    """True when *word* carries trailing clause/sentence punctuation."""
    return word.text.strip().endswith(CLAUSE_PUNCT_TAIL)
