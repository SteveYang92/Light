"""Calibration tests — human annotations vs judge scores."""

from __future__ import annotations

from light_eval.calibration import TRUST_THRESHOLD, calibrate
from light_eval.models import Annotation, DimensionScore


def _ann(**dimensions: int) -> Annotation:
    return Annotation(dimensions=dimensions)


def _judge(**dimensions: float) -> list[DimensionScore]:
    return [DimensionScore(dimension=d, score=s, passed=s >= 4) for d, s in dimensions.items()]


# ── MAE & ±1 agreement ─────────────────────────────────────────────────────


def test_perfect_agreement_is_trustworthy() -> None:
    report = calibrate(
        [
            (_ann(faithfulness=5, naturalness=4), _judge(faithfulness=5, naturalness=4)),
            (_ann(faithfulness=3, naturalness=5), _judge(faithfulness=3, naturalness=5)),
        ]
    )
    assert report.n_cases == 2
    assert report.n_pairs == 4
    assert report.overall_mae == 0.0
    assert report.overall_within_one_rate == 1.0
    assert report.trustworthy

    dims = {d.dimension: d for d in report.dimensions}
    assert dims["faithfulness"].n_pairs == 2
    assert dims["faithfulness"].mae == 0.0
    assert dims["faithfulness"].within_one_rate == 1.0


def test_mae_and_within_one_rate_computed_per_dimension() -> None:
    report = calibrate(
        [
            (_ann(faithfulness=5, naturalness=4), _judge(faithfulness=3, naturalness=3)),  # off by 2 / 1
            (_ann(faithfulness=4, naturalness=4), _judge(faithfulness=4, naturalness=5)),  # off by 0 / 1
        ]
    )
    dims = {d.dimension: d for d in report.dimensions}
    assert dims["faithfulness"].mae == 1.0  # (2 + 0) / 2
    assert dims["faithfulness"].within_one_rate == 0.5
    assert dims["naturalness"].mae == 1.0  # (1 + 1) / 2
    assert dims["naturalness"].within_one_rate == 1.0


def test_threshold_boundary_at_point_eight() -> None:
    # 4 of 5 pairs within ±1 → exactly 0.8 → trustworthy (>= threshold)
    report = calibrate(
        [
            (_ann(faithfulness=5), _judge(faithfulness=4)),
            (_ann(faithfulness=4), _judge(faithfulness=4)),
            (_ann(faithfulness=3), _judge(faithfulness=3)),
            (_ann(faithfulness=2), _judge(faithfulness=3)),
            (_ann(faithfulness=5), _judge(faithfulness=2)),  # off by 3
        ]
    )
    assert report.overall_within_one_rate == 0.8
    assert report.trustworthy is (0.8 >= TRUST_THRESHOLD)

    # 3 of 5 within ±1 → 0.6 → not trustworthy
    report = calibrate(
        [
            (_ann(faithfulness=5), _judge(faithfulness=4)),
            (_ann(faithfulness=4), _judge(faithfulness=4)),
            (_ann(faithfulness=3), _judge(faithfulness=3)),
            (_ann(faithfulness=5), _judge(faithfulness=1)),
            (_ann(faithfulness=5), _judge(faithfulness=2)),
        ]
    )
    assert report.overall_within_one_rate == 0.6
    assert not report.trustworthy


# ── Pair selection & edge cases ─────────────────────────────────────────────


def test_only_shared_dimensions_enter_stats() -> None:
    report = calibrate(
        [
            # boundary_quality human-only, terminology judge-only → both dropped
            (_ann(boundary_quality=4, faithfulness=5), _judge(faithfulness=5, terminology=2)),
        ]
    )
    assert report.n_pairs == 1
    assert [d.dimension for d in report.dimensions] == ["faithfulness"]


def test_no_shared_pairs_is_not_trustworthy() -> None:
    report = calibrate([(_ann(faithfulness=5), _judge(naturalness=5))])
    assert report.n_pairs == 0
    assert report.overall_within_one_rate == 0.0
    assert not report.trustworthy
    assert report.dimensions == []


def test_empty_input_yields_empty_report() -> None:
    report = calibrate([])
    assert report.n_cases == 0
    assert report.n_pairs == 0
    assert not report.trustworthy


# ── Serialization ───────────────────────────────────────────────────────────


def test_to_dict_and_to_text() -> None:
    report = calibrate([(_ann(faithfulness=5), _judge(faithfulness=4))])
    data = report.to_dict()
    assert data["n_cases"] == 1
    assert data["trustworthy"] is True
    assert data["trust_threshold"] == TRUST_THRESHOLD
    assert data["dimensions"][0]["dimension"] == "faithfulness"
    assert data["dimensions"][0]["mae"] == 1.0

    text = report.to_text()
    assert "faithfulness" in text
    assert "MAE=1.00" in text
    assert "TRUSTWORTHY" in text
