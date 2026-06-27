"""Tests for the token usage tracking module."""

from __future__ import annotations

import json
from pathlib import Path

from light_subtitle.usage.models import CostSummary
from light_subtitle.usage.pricing import compute_cost, merge_cost
from light_subtitle.usage.report import merge_reports
from light_subtitle.usage.tracker import (
    UsageTracker,
    merge_token_usage,
    parse_api_usage,
    save_step_usage,
    usage_delta,
)


class _FakeUsage:
    def model_dump(self, *, exclude_none: bool = False) -> dict:
        return {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens": 40,
            "total_tokens": 140,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }


def test_parse_api_usage_includes_cache_buckets() -> None:
    parsed = parse_api_usage(_FakeUsage())
    assert parsed["prompt_cache_hit_tokens"] == 80
    assert parsed["prompt_cache_miss_tokens"] == 20
    assert parsed["reasoning_tokens"] == 5


def test_merge_token_usage_accumulates() -> None:
    total: dict = {}
    merge_token_usage(total, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    merge_token_usage(total, {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    assert total["prompt_tokens"] == 13
    assert total["total_tokens"] == 20
    assert total["calls"] == 2


def test_usage_delta() -> None:
    before = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "calls": 1}
    after = {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37, "calls": 3}
    delta = usage_delta(before, after)
    assert delta["prompt_tokens"] == 15
    assert delta["calls"] == 2


def test_compute_cost_api_direct() -> None:
    cost = compute_cost({"cost_usd": 0.0123}, "deepseek-v4-flash")
    assert cost.total_usd == 0.0123
    assert cost.source == "api"


def test_compute_cost_api_buckets() -> None:
    usage = {
        "prompt_cache_hit_tokens": 1_000_000,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
    }
    cost = compute_cost(usage, "deepseek-v4-flash")
    assert cost.source == "api_buckets"
    assert cost.total_usd is not None
    assert abs(cost.total_usd - 0.0028) < 1e-6


def test_compute_cost_fallback_flat() -> None:
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    cost = compute_cost(usage, "deepseek-v4-flash")
    assert cost.source == "fallback"
    assert cost.total_usd is not None


def test_compute_cost_unknown_model() -> None:
    cost = compute_cost({"prompt_tokens": 100}, "unknown-model")
    assert cost.total_usd is None
    assert cost.source == "unknown"


def test_tracker_load_from_artifacts(tmp_path: Path) -> None:
    punct_dir = tmp_path / "punct_restore"
    punct_dir.mkdir()
    save_step_usage(
        punct_dir / "usage.json",
        {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60, "calls": 2},
    )

    tracker = UsageTracker(model="deepseek-v4-flash")
    tracker.load_from_dir(tmp_path)
    report = tracker.build_report()
    assert report.steps["punct"]["total_tokens"] == 60


def test_merge_reports() -> None:
    from light_subtitle.usage.models import UsageReport

    a = UsageReport(model="deepseek-v4-flash", steps={"punct": {"prompt_tokens": 10, "total_tokens": 10, "calls": 1}})
    b = UsageReport(model="deepseek-v4-flash", steps={"context": {"prompt_tokens": 20, "total_tokens": 20, "calls": 1}})
    merged = merge_reports([a, b])
    assert merged.steps["punct"]["prompt_tokens"] == 10
    assert merged.steps["context"]["prompt_tokens"] == 20


def test_merge_cost_prefers_api_source() -> None:
    total = CostSummary(source="fallback", total_usd=1.0)
    merge_cost(total, CostSummary(source="api", total_usd=0.5))
    assert total.source == "api"
    assert total.total_usd == 1.5


def test_tracker_load_translate_breakdown_no_double_count(tmp_path: Path) -> None:
    """Resume must not sum breakdown sub-steps and top-level translate totals twice."""
    tx_dir = tmp_path / "translations"
    tx_dir.mkdir()
    save_step_usage(
        tx_dir / "usage.json",
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "calls": 3,
            "breakdown": {
                "translate.translate": {
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "total_tokens": 120,
                    "calls": 2,
                },
                "translate.retry": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                    "calls": 1,
                },
            },
        },
    )

    tracker = UsageTracker(model="deepseek-v4-flash")
    tracker.load_from_dir(tmp_path)
    report = tracker.build_report()
    assert report.totals["prompt_tokens"] == 100
    assert report.totals["total_tokens"] == 150
    assert report.totals["calls"] == 3
    assert "translate" not in report.steps
    tracker = UsageTracker(model="deepseek-v4-flash")
    tracker.record("context", {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    path = tracker.save_report(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["steps"]["context"]["total_tokens"] == 120
    assert "cost" in data
