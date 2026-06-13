"""Shared punctuation string constants for subtitle text processing.

All constants are plain strings (not sets/frozensets) so they work
with both ``rstrip()`` and ``any(ch in CONST for ch in text)`` patterns.
Consumers that need a set should call ``set(CONSTANT)`` locally.
"""

# ── English trailing punctuation (for rstripping word endings) ──────────
EN_TRAILING_PUNCT = ",.;!?\"'"

# ── Sentence-ending punctuation (all languages) ─────────────────────────
SENTENCE_ENDS = ".!?。！？…"

# ── Clause-level punctuation (line breaks allowed after these) ──────────
CLAUSE_PUNCT = ",;:—，、；："

# ── CJK pause/continuation punctuation ─────────────────────────────────
CJK_CLAUSE_PUNCT = "，、；："

# ── CJK sentence-ending punctuation ────────────────────────────────────
CJK_SENTENCE_ENDS = "。？！…"

# ── Full CJK punctuation set (for "is text purely CJK punct?" checks) ──
CJK_ALL_PUNCT = "，、。？！；：·…—～"

# ── CJK grammatical particles ──────────────────────────────────────────
CJK_PARTICLES = "的地得"
CJK_SENTENCE_PARTICLES = "的了吗呢吧啊么嘛呗"
