"""Rule judges — deterministic per-step metrics, no LLM involved.

plan dimensions:
- ``word_coverage``: output unit words must cover 100% of input words.
- ``duration_violations``: units shorter than ``min_duration`` or longer
  than ``max_duration`` (case params, else ``PlanConfig`` defaults).
- ``dangling_tails``: units ending on a function word (reuses
  :func:`light_subtitle.plan.boundary.dangling_tail`).
- ``empty_units``: units with blank text.

translate dimensions:
- ``unit_coverage``: share of input units covered by output cues
  (``merged_from`` chains count, via ``covered_unit_ids``).
- ``empty_translations``: cues with blank text.
- ``target_lang_ratio``: CJK character share of cue text (zh only).
- ``source_fidelity``: per-cue consistency with the input units — known
  unit ids, chain order, and timing window (via ``covered_time_window``).
"""

from __future__ import annotations

from collections import Counter

from light_models import Word, word_from_dict
from light_models.cue_utils import covered_time_window, effective_unit_ids
from light_subtitle import artifacts
from light_subtitle.config import PlanConfig
from light_subtitle.plan.boundary import dangling_tail
from light_subtitle.translate.translate import covered_unit_ids
from light_text import is_cjk

from ..loader import Fixture
from ..models import DimensionScore, EvalCase, StepOutput

_TIME_TOLERANCE_S = 0.05
_DEFAULT_MIN_TARGET_LANG_RATIO = 0.9


# ── Entry point ─────────────────────────────────────────────────────────────


def judge_for_step(step: str) -> PlanRulesJudge | TranslateRulesJudge:
    """Return the rule judge for *step*."""
    if step == "plan":
        return PlanRulesJudge()
    if step == "translate":
        return TranslateRulesJudge()
    raise ValueError(f"no rule judge for step: {step}")


# ── plan ────────────────────────────────────────────────────────────────────


class PlanRulesJudge:
    """Rule metrics for planned cue units (``plan.json`` output)."""

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[DimensionScore]:
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


def _word_coverage(fixture: Fixture, units: list[dict]) -> DimensionScore:
    """Share of input words covered by the planned units (100% to pass)."""
    input_words = Counter(
        (w.text.strip(), round(w.start, 3), round(w.end, 3)) for seg in fixture.segments for w in seg.words
    )
    output_words = Counter(_word_key(w) for unit in units for w in unit.get("words", []))
    missing = input_words - output_words
    total = sum(input_words.values())
    covered = total - sum(missing.values())
    ratio = covered / total if total else 1.0
    evidence = [f'"{text}" x{n}' for (text, _, _), n in sorted(missing.items())][:20]
    return DimensionScore(
        dimension="word_coverage",
        score=ratio,
        passed=not missing,
        detail=f"{covered}/{total} input words covered",
        evidence=evidence,
    )


def _duration_violations(case: EvalCase, units: list[dict]) -> DimensionScore:
    """Count units outside [min_duration, max_duration] (0 to pass)."""
    min_dur = float(case.params.get("min_duration", PlanConfig.min_duration))
    max_dur = float(case.params.get("max_duration", PlanConfig.max_duration))
    violations = []
    for unit in units:
        dur = unit["end"] - unit["start"]
        if dur < min_dur or dur > max_dur:
            violations.append(f"{unit['unit_id']} ({dur:.2f}s)")
    return DimensionScore(
        dimension="duration_violations",
        score=float(len(violations)),
        passed=not violations,
        detail=f"{len(violations)} unit(s) outside [{min_dur}, {max_dur}]s",
        evidence=violations,
    )


def _dangling_tails(units: list[dict]) -> DimensionScore:
    """Count units whose last word is a stranded function word (0 to pass)."""
    offenders = []
    for unit in units:
        words = unit.get("words", [])
        if not words:
            continue
        last: Word = word_from_dict(words[-1])
        bad = dangling_tail(last)
        if bad is not None:
            offenders.append(f'{unit["unit_id"]} (ends on "{bad}")')
    return DimensionScore(
        dimension="dangling_tails",
        score=float(len(offenders)),
        passed=not offenders,
        detail=f"{len(offenders)} unit(s) end on a function word",
        evidence=offenders,
    )


def _empty_units(units: list[dict]) -> DimensionScore:
    """Count units with blank text (0 to pass)."""
    empty = [unit["unit_id"] for unit in units if not unit.get("text", "").strip()]
    return DimensionScore(
        dimension="empty_units",
        score=float(len(empty)),
        passed=not empty,
        detail=f"{len(empty)} empty unit(s)",
        evidence=empty,
    )


# ── translate ───────────────────────────────────────────────────────────────


class TranslateRulesJudge:
    """Rule metrics for translated cues (``raw.json``-schema output)."""

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[DimensionScore]:
        if output.skipped or output.error:
            return []
        cues = [artifacts.cue_from_dict(raw) for raw in output.output]
        return [
            _unit_coverage(fixture, cues),
            _empty_translations(cues),
            _target_lang_ratio(case, cues),
            _source_fidelity(fixture, cues),
        ]


def _unit_coverage(fixture: Fixture, cues) -> DimensionScore:
    """Share of input unit ids covered by cues incl. merge chains (100% to pass)."""
    input_ids = {seg.unit_id for seg in fixture.segments}
    covered = covered_unit_ids(cues) & input_ids
    missing = sorted(input_ids - covered)
    ratio = len(covered) / len(input_ids) if input_ids else 1.0
    return DimensionScore(
        dimension="unit_coverage",
        score=ratio,
        passed=not missing,
        detail=f"{len(covered)}/{len(input_ids)} units covered",
        evidence=missing[:20],
    )


def _empty_translations(cues) -> DimensionScore:
    """Count cues with blank target text (0 to pass)."""
    empty = [cue.unit_id for cue in cues if not cue.text.strip()]
    return DimensionScore(
        dimension="empty_translations",
        score=float(len(empty)),
        passed=not empty,
        detail=f"{len(empty)} empty translation(s)",
        evidence=empty,
    )


def _target_lang_ratio(case: EvalCase, cues) -> DimensionScore:
    """CJK share of cue text for ``target_lang=zh`` (threshold to pass)."""
    target_lang = str(case.params.get("target_lang", "zh"))
    if target_lang != "zh":
        return DimensionScore(
            dimension="target_lang_ratio",
            score=1.0,
            passed=True,
            detail=f"target_lang={target_lang}: ratio check implemented for zh only",
        )
    chars = [ch for cue in cues for ch in cue.text if not ch.isspace()]
    cjk = sum(1 for ch in chars if is_cjk(ch))
    ratio = cjk / len(chars) if chars else 0.0
    threshold = float(case.params.get("min_target_lang_ratio", _DEFAULT_MIN_TARGET_LANG_RATIO))
    return DimensionScore(
        dimension="target_lang_ratio",
        score=ratio,
        passed=ratio >= threshold,
        detail=f"{cjk}/{len(chars)} non-space chars are CJK (threshold {threshold})",
    )


def _source_fidelity(fixture: Fixture, cues) -> DimensionScore:
    """Per-cue consistency with input units: known ids, ordered chains, timing."""
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
    ratio = n_ok / len(cues) if cues else 1.0
    return DimensionScore(
        dimension="source_fidelity",
        score=ratio,
        passed=not problems,
        detail=f"{n_ok}/{len(cues)} cues consistent with input units",
        evidence=problems[:20],
    )
