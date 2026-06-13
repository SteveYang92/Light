import time

from light_models import QCIssue, QCReport, SubtitleCue

from .config import QCConfig
from .rules.registry import RuleEngine


def run_qc(cues: dict[str, list[SubtitleCue]], config: QCConfig) -> QCReport:
    issues: list[QCIssue] = []

    # ── Step 0: Transcript alignment ─────────────────────────
    if config.transcript_path:
        from .alignment import (
            align_words_to_cues,
            build_coverage_issues,
            load_transcript,
        )

        words = load_transcript(config.transcript_path)
        total_transcript_words = len(words)
        uncovered = align_words_to_cues(cues, words, config.alignment_tolerance)
        issues.extend(build_coverage_issues(uncovered, total_transcript_words, config.word_coverage_min))

    # ── Step 1: Rule engine ─────────────────────────────────
    rules = RuleEngine(config)
    issues.extend(rules.check(cues))
    # NOTE: EntryPointAccuracy / ExitPointPrecision / WordGapAnomaly
    # etc. automatically fire when cue.words is populated by alignment.

    llm_new = 0
    if config.llm_enabled and config.llm_api_key:
        t0 = time.time()
        from .llm_qc import run_llm_qc

        llm_issues = run_llm_qc(cues, config)
        elapsed = time.time() - t0

        # Dedup: rule engine results take priority over LLM QC.
        seen: set[tuple[str, int | None]] = set()
        for i in issues:
            seen.add((i.rule, i.cue_id))

        for li in llm_issues:
            key = (li.rule, li.cue_id)
            if key not in seen:
                li.detail = li.detail + " [LLM]"
                issues.append(li)
                seen.add(key)
                llm_new += 1

        if llm_new:
            print(f"  LLM QC: {llm_new} new issue(s) in {elapsed:.1f}s")
        else:
            print(f"  LLM QC: no new issues (all overlapped with rule engine) [{elapsed:.1f}s]")

    return _summarize(cues, config, issues)


def _summarize(cues: dict[str, list[SubtitleCue]], config: QCConfig, issues: list[QCIssue]) -> QCReport:
    total = sum(len(c) for c in cues.values())
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    suggestions = sum(1 for i in issues if i.severity == "suggestion")

    issues.sort(key=lambda i: {"error": 0, "warning": 1, "suggestion": 2}[i.severity])

    return QCReport(
        total_cues=total,
        errors=errors,
        warnings=warnings,
        suggestions=suggestions,
        passed=errors == 0,
        bilingual=config.bilingual,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        issues=issues,
    )
