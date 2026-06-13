"""Light QC rules package."""

from __future__ import annotations

from .acronym_split import AcronymSplit
from .badword_split import BadWordSplit
from .bilingual_balance import BilingualBalance
from .boundary_repetition import BoundaryRepetition
from .cjk_space_anomaly import CJKSpaceAnomaly
from .compound_words import CompoundWords
from .consecutive_fragments import ConsecutiveFragments
from .exit_point_precision import ExitPointPrecision
from .line_imbalance import LineImbalance
from .orphan_words import OrphanWords
from .overlapping_content import OverlappingContent
from .padding_conflict import PaddingConflict
from .punctuation_mid_line import PunctuationMidLine
from .reading_padding import ReadingPadding
from .semantic_breaks import SemanticBreaks
from .shot_change_soft import ShotChangeSoft
from .speaker_consistency import SpeakerConsistency
from .terminology_consistency import TerminologyConsistency
from .translation_quality import TranslationQuality
from .word_gap_anomaly import WordGapAnomaly

__all__ = [
    "AcronymSplit",
    "BadWordSplit",
    "BilingualBalance",
    "BoundaryRepetition",
    "CJKSpaceAnomaly",
    "CompoundWords",
    "ConsecutiveFragments",
    "ExitPointPrecision",
    "LineImbalance",
    "OrphanWords",
    "OverlappingContent",
    "PaddingConflict",
    "PunctuationMidLine",
    "ReadingPadding",
    "SemanticBreaks",
    "ShotChangeSoft",
    "SpeakerConsistency",
    "TerminologyConsistency",
    "TranslationQuality",
    "WordGapAnomaly",
]
