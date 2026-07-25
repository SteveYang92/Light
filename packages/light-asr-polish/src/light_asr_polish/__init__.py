"""light-asr-polish — LLM-based ASR post-processing.

Two word-level polish stages over ASR output:

- :func:`correct` — transcript correction (homophones, proper nouns,
  duplicate words, grammar word forms) with domain-context extraction.
- :func:`restore_punct` — punctuation restoration.

Both take word-level ASR output plus an ``OpenAIClient`` and return the
polished words together with token-usage data.  Debug artifacts are written
under *work_dir* when given, and skipped when it is ``None``.
"""

from __future__ import annotations

from .correct import correct
from .punct import restore_punct
from .word_segments import WordSegment, group_words_by_gap, join_word_text, merge_short_segments

__all__ = [
    "WordSegment",
    "correct",
    "group_words_by_gap",
    "join_word_text",
    "merge_short_segments",
    "restore_punct",
]
