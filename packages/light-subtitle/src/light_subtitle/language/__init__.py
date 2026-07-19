"""Language processing — shared utilities for both English and CJK.

English is the primary language; CJK support is available via explicit
import from ``language.cjk``.

Usage::

    from light_subtitle.language import is_sentence_end, detect_source_lang
    from light_subtitle.language.english import EnglishBreakFinder
    from light_subtitle.language.cjk import ChineseBreakFinder
"""

from .base import SENTENCE_END as SENTENCE_END
from .base import detect_source_lang as detect_source_lang
from .base import is_abbreviation_dot as is_abbreviation_dot
from .base import is_sentence_end as is_sentence_end
