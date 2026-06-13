from __future__ import annotations

from light_models import QCIssue, SubtitleCue

from ..config import QCConfig
from .base import HardRule, SoftRule


class RuleEngine:
    def __init__(self, config: QCConfig):
        self.config = config
        self._hard_rules: list[HardRule] = []
        self._soft_rules: list[SoftRule] = []
        self._register()

    def _register(self):
        from . import hard, soft

        # ── Hard rules ──
        self._hard_rules = [
            hard.MaxLines(),
            hard.ChineseLineLength(),
            # hard.EnglishLineLength(),  # temporarily disabled
            hard.ReadingSpeed(),
            hard.MinDuration(),
            hard.MaxDuration(),
            hard.Overlap(),
            hard.EmptyText(),
            hard.MissingPunctuation(),
            hard.LeadingPunctuation(),
            hard.TinyCue(),
            hard.UntranslatedText(),
            hard.DuplicateTranslation(),
            hard.EntryPointAccuracy(),
            hard.EntryPointConfidence(),
            hard.TimelineGap(),
        ]

        if self.config.shot_changes:
            self._hard_rules.append(hard.ShotBoundaryHard())

        # GapFlash — always active
        self._hard_rules.append(hard.GapFlash())

        if self.config.target_lang is not None and not self.config.bilingual:
            self._hard_rules.append(hard.TimeAxisNotOverflow())
            self._hard_rules.append(hard.TranslationCompleteness())

        if self.config.bilingual:
            self._hard_rules.append(hard.BilingualMapping())
            self._hard_rules.append(hard.TranslationCompleteness())
            self._hard_rules.append(hard.CombinedReadingSpeed())
            self._hard_rules.append(hard.VisualDensity())
            self._hard_rules.append(hard.LineBalance())

        # ── Soft rules ──
        self._soft_rules = [
            soft.BadWordSplit(),
            soft.SemanticBreaks(),
            soft.OrphanWords(),
            soft.CompoundWords(),
            soft.SpeakerConsistency(),
            soft.ExitPointPrecision(),
            soft.ShotChangeSoft(),
            soft.WordGapAnomaly(),
            soft.PaddingConflict(),
            soft.CJKSpaceAnomaly(),
            soft.AcronymSplit(),
            soft.PunctuationMidLine(),
            soft.ConsecutiveFragments(),
            soft.LineImbalance(),
            soft.OverlappingContent(),
            soft.BoundaryRepetition(),
        ]

        if self.config.target_lang is not None:
            self._soft_rules.append(soft.TranslationQuality())
            self._soft_rules.append(soft.TerminologyConsistency())

        if self.config.bilingual:
            self._soft_rules.append(soft.BilingualBalance())

    def check(self, cues: dict[str, list[SubtitleCue]]) -> list[QCIssue]:
        issues = []
        for rule in self._hard_rules:
            issues.extend(rule.check(self._filter_cues(cues, rule), self.config))
        for rule in self._soft_rules:
            issues.extend(rule.check(self._filter_cues(cues, rule), self.config))
        return issues

    @staticmethod
    def _filter_cues(cues: dict[str, list[SubtitleCue]], rule: HardRule | SoftRule) -> dict[str, list[SubtitleCue]]:
        """Filter cues dict to only languages the rule applies to."""
        if rule.languages is None:
            return cues
        return {lang: cl for lang, cl in cues.items() if lang in rule.languages}
