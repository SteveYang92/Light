from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class TerminologyConsistency(SoftRule):
    """Check that glossary terms are used consistently across all cues.

    For each term in the glossary, verify that the source text only appears
    in source-language cues and the target translation only in target-language
    cues.  Flag any mismatch.
    """

    name = "TerminologyConsistency"
    default_severity = "warning"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        if not config.glossary:
            return []

        issues = []

        # Collect all cues separated by language.
        lang_cues: dict[str, list[tuple[int, SubtitleCue]]] = {}
        for lang, cue_list in _iter_cues(cues):
            lang_cues[lang] = [(i, cue) for i, cue in enumerate(cue_list)]

        # For each glossary entry, check that the source term does not appear
        # in target-language cues, and vice versa.
        for src_term, tgt_term in config.glossary.items():
            for _lang, indexed in lang_cues.items():
                for idx, cue in indexed:
                    text_lower = cue.text.replace("\n", " ").lower()

                    # Check if source term leaked into target
                    if src_term.lower() in text_lower and cue.lang != config.source_lang:
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=idx + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"源语言术语 '{src_term}' 出现在目标语言字幕中",
                                fix=f"替换为对应术语 '{tgt_term}'",
                            )
                        )

                    # Check if target term leaked into source
                    if tgt_term.lower() in text_lower and cue.lang == config.source_lang:
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=idx + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"译文术语 '{tgt_term}' 出现在源语言字幕中",
                                fix=f"使用源语言术语 '{src_term}'",
                            )
                        )

        return issues
