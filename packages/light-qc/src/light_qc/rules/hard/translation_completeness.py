from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule


class TranslationCompleteness(HardRule):
    """Scene B/C: verify that every source cue has a corresponding translation.

    Operates in two modes:

    - **unit_id mode** (pipeline): cues carry ``unit_id`` from
      ``SemanticUnit`` — exact set-difference check. Also detects
      orphan target unit_ids (LLM hallucination).
    - **time-overlap mode** (standalone SRT/VTT): when unit_ids are
      unavailable fall back to checking that every source cue overlaps
      at least one target cue in time.

    When only one language is present (e.g. standalone Scene B with a
    single translated file) the rule silently skips.
    """

    name = "TranslationCompleteness"

    # ── public entry ───────────────────────────────────────────

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        source_cues = cues.get(config.source_lang)
        if not source_cues:
            return []

        target_cues = self._find_target(cues, config)
        if target_cues is None:
            return []

        # Choose mode based on unit_id availability.
        has_unit_ids = any(c.unit_id for c in source_cues) and any(c.unit_id for c in target_cues)

        if has_unit_ids:
            return self._check_unit_ids(source_cues, target_cues)
        else:
            return self._check_time_overlap(source_cues, target_cues)

    # ── unit_id mode ───────────────────────────────────────────

    def _check_unit_ids(self, source_cues: list[SubtitleCue], target_cues: list[SubtitleCue]) -> list[QCIssue]:
        issues: list[QCIssue] = []

        source_units = {c.unit_id for c in source_cues if c.unit_id}
        target_units = {c.unit_id for c in target_cues if c.unit_id}

        missing = source_units - target_units
        orphan = target_units - source_units

        for uid in sorted(missing):
            sc = next((c for c in source_cues if c.unit_id == uid), None)
            if sc and sc.text.strip():
                preview = sc.text.replace("\n", " ")[:50]
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=None,
                        time=seconds_to_srt(sc.start),
                        detail=f"语义单元 {uid} 缺少翻译: '{preview}'",
                        fix="为该单元补充翻译或检查 LLM 是否遗漏",
                    )
                )

        for uid in sorted(orphan):
            tc = next((c for c in target_cues if c.unit_id == uid), None)
            if tc and tc.text.strip():
                preview = tc.text.replace("\n", " ")[:50]
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=None,
                        time=seconds_to_srt(tc.start),
                        detail=f"翻译单元 {uid} 没有对应的源语言单元: '{preview}'",
                        fix="移除多余的翻译或确认是否为 LLM 幻觉",
                    )
                )

        return issues

    # ── time-overlap mode ──────────────────────────────────────

    def _check_time_overlap(self, source_cues: list[SubtitleCue], target_cues: list[SubtitleCue]) -> list[QCIssue]:
        issues: list[QCIssue] = []

        for i, sc in enumerate(source_cues):
            if not sc.text.strip():
                continue

            covered = any(max(sc.start, tc.start) < min(sc.end, tc.end) - 0.05 for tc in target_cues)

            if not covered:
                preview = sc.text.replace("\n", " ")[:50]
                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=i + 1,
                        time=seconds_to_srt(sc.start),
                        detail=f"源语言字幕缺少翻译覆盖: '{preview}'",
                        fix="为该条字幕补充翻译",
                    )
                )

        return issues

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _find_target(cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[SubtitleCue] | None:
        """Locate the target-language cue list.

        Tries ``config.target_lang`` first, then falls back to any
        non-source language key.
        """
        # Try explicit target_lang
        if config.target_lang and config.target_lang in cues:
            return cues[config.target_lang]

        # Fallback: first key that isn't the source language
        for lang, cue_list in cues.items():
            if lang != config.source_lang:
                return cue_list

        return None
