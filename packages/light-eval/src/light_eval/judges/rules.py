"""Rule judges — deterministic per-step metrics, no LLM involved."""

from __future__ import annotations

from collections import Counter

from light_models import word_from_dict
from light_models.cue_utils import covered_time_window, effective_unit_ids
from light_subtitle import artifacts
from light_subtitle.config import PlanConfig
from light_subtitle.plan.boundary import dangling_tail, effective_tail
from light_subtitle.translate.translate import covered_unit_ids
from light_text import is_cjk

from ..loader import Fixture
from ..models import EvalCase, ProblemTypeStats, StepOutput

_TIME_TOLERANCE_S = 0.05
_DEFAULT_TARGET_LANG_RATIO_THRESHOLD = 0.6


def judge_for_step(step: str) -> PlanRulesJudge | TranslateRulesJudge:
    if step == "plan":
        return PlanRulesJudge()
    if step == "translate":
        return TranslateRulesJudge()
    raise ValueError(f"no rule judge for step: {step}")


# ── plan ────────────────────────────────────────────────────────────────────


class PlanRulesJudge:
    """Rule metrics for planned cue units (``plan.json`` output)."""

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[ProblemTypeStats]:
        if output.skipped or output.error:
            return []
        units = output.output
        return [
            _word_coverage(fixture, units),
            _duration_violations(case, units),
            _dangling_tails(units),
            _empty_units(units),
        ]


def _word_key(word_dict: dict) -> tuple[str, float, float]:
    return (word_dict["text"].strip(), round(word_dict["start"], 3), round(word_dict["end"], 3))


def _word_coverage(fixture: Fixture, units: list[dict]) -> ProblemTypeStats:
    input_words = Counter(
        (w.text.strip(), round(w.start, 3), round(w.end, 3)) for seg in fixture.segments for w in seg.words
    )
    output_words = Counter(_word_key(w) for unit in units for w in unit.get("words", []))
    missing = input_words - output_words
    total = sum(input_words.values())
    covered = total - sum(missing.values())
    evidence = [f'"{text}" x{n}' for (text, _, _), n in sorted(missing.items())][:20]
    return ProblemTypeStats(
        problem_type="word_coverage",
        error_count=0 if not missing else 1,
        passed=not missing,
        detail=f"{covered}/{total} input words covered",
        evidence=evidence,
    )


def _duration_violations(case: EvalCase, units: list[dict]) -> ProblemTypeStats:
    min_dur = float(case.params.get("min_duration", PlanConfig.min_duration))
    max_dur = float(case.params.get("max_duration", PlanConfig.max_duration))
    soft_max = max_dur * 1.15
    violations = []
    for unit in units:
        dur = unit["end"] - unit["start"]
        if dur < min_dur or dur > soft_max:
            violations.append(f"{unit['unit_id']} ({dur:.2f}s)")
    return ProblemTypeStats(
        problem_type="duration_violations",
        error_count=len(violations),
        passed=not violations,
        detail=f"{len(violations)} unit(s) outside [{min_dur}, {soft_max:.2f}]s",
        evidence=violations,
    )


def _dangling_tails(units: list[dict]) -> ProblemTypeStats:
    offenders = []
    for unit in units:
        words = [word_from_dict(w) for w in unit.get("words", [])]
        tail = effective_tail(words)
        if tail is None:
            continue
        bad = dangling_tail(tail)
        if bad is not None:
            offenders.append(f'{unit["unit_id"]} (ends on "{bad}")')
    return ProblemTypeStats(
        problem_type="dangling_tails",
        error_count=len(offenders),
        passed=not offenders,
        detail=f"{len(offenders)} unit(s) end on a function word",
        evidence=offenders,
    )


def _empty_units(units: list[dict]) -> ProblemTypeStats:
    empty = [unit["unit_id"] for unit in units if not unit.get("text", "").strip()]
    return ProblemTypeStats(
        problem_type="empty_units",
        error_count=len(empty),
        passed=not empty,
        detail=f"{len(empty)} empty unit(s)",
        evidence=empty,
    )


# ── translate ───────────────────────────────────────────────────────────────


class TranslateRulesJudge:
    """Rule metrics for translated cues (``raw.json``-schema output)."""

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[ProblemTypeStats]:
        if output.skipped or output.error:
            return []
        cues = [artifacts.cue_from_dict(raw) for raw in output.output]
        return [
            _unit_coverage(fixture, cues),
            _empty_translations(cues),
            _target_lang_ratio(case, cues),
            _source_fidelity(fixture, cues),
        ]


def _unit_coverage(fixture: Fixture, cues) -> ProblemTypeStats:
    input_ids = {seg.unit_id for seg in fixture.segments}
    covered = covered_unit_ids(cues) & input_ids
    missing = sorted(input_ids - covered)
    return ProblemTypeStats(
        problem_type="unit_coverage",
        error_count=0 if not missing else 1,
        passed=not missing,
        detail=f"{len(covered)}/{len(input_ids)} units covered",
        evidence=missing[:20],
    )


def _empty_translations(cues) -> ProblemTypeStats:
    empty = [cue.unit_id for cue in cues if not cue.text.strip()]
    return ProblemTypeStats(
        problem_type="empty_translations",
        error_count=len(empty),
        passed=not empty,
        detail=f"{len(empty)} empty translation(s)",
        evidence=empty,
    )


def _target_lang_ratio(case: EvalCase, cues) -> ProblemTypeStats:
    target_lang = str(case.params.get("target_lang", "zh"))
    if target_lang != "zh":
        return ProblemTypeStats(
            problem_type="target_lang_ratio",
            passed=True,
            detail=f"target_lang={target_lang}: zh-only",
        )
    chars = [ch for cue in cues for ch in cue.text if not ch.isspace()]
    cjk = sum(1 for ch in chars if is_cjk(ch))
    ratio = cjk / len(chars) if chars else 0.0
    threshold = float(case.params.get("target_lang_ratio_threshold", _DEFAULT_TARGET_LANG_RATIO_THRESHOLD))
    return ProblemTypeStats(
        problem_type="target_lang_ratio",
        error_count=0 if ratio >= threshold else 1,
        passed=ratio >= threshold,
        detail=f"{cjk}/{len(chars)} non-space chars are CJK (threshold {threshold})",
    )


def _source_fidelity(fixture: Fixture, cues) -> ProblemTypeStats:
    input_by_id = {seg.unit_id: seg for seg in fixture.segments}
    input_order = {seg.unit_id: i for i, seg in enumerate(fixture.segments)}
    unit_times = {seg.unit_id: (seg.start, seg.end) for seg in fixture.segments}
    problems: list[str] = []
    for cue in cues:
        ids = effective_unit_ids(cue)
        unknown = sorted(ids - set(input_by_id))
        if unknown:
            problems.append(f"{cue.cue_id}: unknown unit ids {unknown}")
            continue
        chain = [cue.unit_id, *cue.merged_from]
        if [input_order[u] for u in chain] != sorted(input_order[u] for u in chain):
            problems.append(f"{cue.cue_id}: merge chain out of input order")
            continue
        window = covered_time_window(cue, unit_times)
        timing_ok = (
            window is not None
            and abs(cue.start - window[0]) <= _TIME_TOLERANCE_S
            and abs(cue.end - window[1]) <= _TIME_TOLERANCE_S
        )
        if not timing_ok:
            problems.append(f"{cue.cue_id}: timing mismatch vs covered units")
    n_ok = len(cues) - len(problems)
    return ProblemTypeStats(
        problem_type="source_fidelity",
        error_count=len(problems),
        passed=not problems,
        detail=f"{n_ok}/{len(cues)} cues consistent with input units",
        evidence=problems[:20],
    )
