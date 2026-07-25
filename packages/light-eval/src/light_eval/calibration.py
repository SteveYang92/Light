"""LLM-judge calibration against human annotations.

Pairs each case's human ``Annotation`` (1-5 per dimension) with the
:class:`~light_eval.judges.llm.LLMJudge` scores for the same case, then
reports per-dimension MAE and ±1-point agreement rate.  The judge counts
as *trustworthy* when the pooled ±1 agreement rate reaches 0.8.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .models import Annotation, DimensionScore

TRUST_THRESHOLD = 0.8  # pooled ±1 agreement rate at which the judge counts as calibrated


# ── Report models ───────────────────────────────────────────────────────────


@dataclass
class DimensionCalibration:
    """Calibration stats for one rubric dimension."""

    dimension: str
    n_pairs: int
    mae: float
    within_one_rate: float  # share of pairs with |human - judge| <= 1

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "n_pairs": self.n_pairs,
            "mae": round(self.mae, 4),
            "within_one_rate": round(self.within_one_rate, 4),
        }


@dataclass
class CalibrationReport:
    """Aggregate calibration verdict across cases and dimensions."""

    n_cases: int = 0
    n_pairs: int = 0
    overall_within_one_rate: float = 0.0
    overall_mae: float = 0.0
    trustworthy: bool = False
    dimensions: list[DimensionCalibration] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "n_pairs": self.n_pairs,
            "overall_within_one_rate": round(self.overall_within_one_rate, 4),
            "overall_mae": round(self.overall_mae, 4),
            "trustworthy": self.trustworthy,
            "trust_threshold": TRUST_THRESHOLD,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }

    def to_text(self) -> str:
        """Console-friendly multi-line summary."""
        lines = [f"LLM judge calibration — {self.n_cases} case(s), {self.n_pairs} score pair(s)"]
        for dim in self.dimensions:
            lines.append(
                f"  {dim.dimension}: pairs={dim.n_pairs}  MAE={dim.mae:.2f}  ±1 agreement={dim.within_one_rate:.1%}"
            )
        verdict = "TRUSTWORTHY" if self.trustworthy else "NOT TRUSTWORTHY"
        lines.append(
            f"overall: MAE={self.overall_mae:.2f}  ±1 agreement={self.overall_within_one_rate:.1%}"
            f" (threshold {TRUST_THRESHOLD:.0%}) → {verdict}"
        )
        return "\n".join(lines)


# ── Calibration ─────────────────────────────────────────────────────────────


def calibrate(pairs: Iterable[tuple[Annotation, Sequence[DimensionScore]]]) -> CalibrationReport:
    """Compare human annotations with judge scores case by case.

    *pairs* yields ``(annotation, judge_scores)`` per case.  Only dimensions
    present on both sides enter the stats.
    """
    per_dimension: dict[str, list[tuple[int, float]]] = {}
    n_cases = 0
    for annotation, scores in pairs:
        n_cases += 1
        judge_by_dim = {s.dimension: s.score for s in scores}
        for dimension, human in annotation.dimensions.items():
            if dimension not in judge_by_dim:
                continue
            per_dimension.setdefault(dimension, []).append((human, judge_by_dim[dimension]))

    all_pairs = [pair for pairs_ in per_dimension.values() for pair in pairs_]
    dimensions = [_stats(dimension, dim_pairs) for dimension, dim_pairs in sorted(per_dimension.items())]
    within_one_rate = _within_one_rate(all_pairs)
    return CalibrationReport(
        n_cases=n_cases,
        n_pairs=len(all_pairs),
        overall_within_one_rate=within_one_rate,
        overall_mae=_mae(all_pairs),
        trustworthy=bool(all_pairs) and within_one_rate >= TRUST_THRESHOLD,
        dimensions=dimensions,
    )


# ── Stats helpers ───────────────────────────────────────────────────────────


def _stats(dimension: str, pairs: list[tuple[int, float]]) -> DimensionCalibration:
    return DimensionCalibration(
        dimension=dimension,
        n_pairs=len(pairs),
        mae=_mae(pairs),
        within_one_rate=_within_one_rate(pairs),
    )


def _mae(pairs: list[tuple[int, float]]) -> float:
    """Mean absolute error between human and judge scores."""
    if not pairs:
        return 0.0
    return sum(abs(human - judge) for human, judge in pairs) / len(pairs)


def _within_one_rate(pairs: list[tuple[int, float]]) -> float:
    """Share of pairs where human and judge scores differ by at most 1."""
    if not pairs:
        return 0.0
    return sum(1 for human, judge in pairs if abs(human - judge) <= 1) / len(pairs)
