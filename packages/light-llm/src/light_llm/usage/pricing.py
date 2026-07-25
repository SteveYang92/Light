"""Cost estimation — API direct cost first, then bucketed token fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import API_COST_KEYS, CostSummary

_FALLBACK_PATH = Path(__file__).with_name("pricing_fallback.yaml")
_FALLBACK_CACHE: dict[str, dict[str, float]] | None = None


def _load_fallback_rates() -> dict[str, dict[str, float]]:
    global _FALLBACK_CACHE
    if _FALLBACK_CACHE is not None:
        return _FALLBACK_CACHE
    if not _FALLBACK_PATH.exists():
        _FALLBACK_CACHE = {}
        return _FALLBACK_CACHE
    with open(_FALLBACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _FALLBACK_CACHE = dict(data.get("models") or {})
    return _FALLBACK_CACHE


def _per_million(tokens: int, rate: float) -> float:
    return tokens * rate / 1_000_000


def _extract_direct_cost(usage: dict[str, Any]) -> float | None:
    """Return API-reported cost when present."""
    for key in API_COST_KEYS:
        val = usage.get(key)
        if val is not None:
            return float(val)
    return None


def _has_cache_buckets(usage: dict[str, Any]) -> bool:
    return "prompt_cache_hit_tokens" in usage or "prompt_cache_miss_tokens" in usage


def compute_cost(usage: dict[str, Any], model: str) -> CostSummary:
    """Compute cost for a single usage dict using the API-first priority chain."""
    direct = _extract_direct_cost(usage)
    if direct is not None:
        return CostSummary(total_usd=direct, source="api")

    rates = _load_fallback_rates().get(model)
    if not rates:
        return CostSummary(source="unknown")

    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int(usage.get("reasoning_tokens") or 0)
    output_tokens = completion + reasoning

    if _has_cache_buckets(usage):
        hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        miss = int(usage.get("prompt_cache_miss_tokens") or 0)
        cache_hit_usd = _per_million(hit, float(rates["cache_hit_per_1m"]))
        cache_miss_usd = _per_million(miss, float(rates["cache_miss_per_1m"]))
        output_usd = _per_million(output_tokens, float(rates["output_per_1m"]))
        total = cache_hit_usd + cache_miss_usd + output_usd
        return CostSummary(
            total_usd=total,
            cache_hit_usd=cache_hit_usd,
            cache_miss_usd=cache_miss_usd,
            output_usd=output_usd,
            source="api_buckets",
        )

    prompt = int(usage.get("prompt_tokens") or 0)
    # Flat fallback: treat all prompt tokens as cache-miss input.
    cache_miss_usd = _per_million(prompt, float(rates["cache_miss_per_1m"]))
    output_usd = _per_million(output_tokens, float(rates["output_per_1m"]))
    total = cache_miss_usd + output_usd
    return CostSummary(
        total_usd=total,
        cache_miss_usd=cache_miss_usd,
        output_usd=output_usd,
        source="fallback",
    )


def merge_cost(total: CostSummary, addition: CostSummary) -> None:
    """Accumulate *addition* into *total*, preserving the strongest source label."""
    if addition.total_usd is None:
        return

    def _add(a: float | None, b: float | None) -> float | None:
        if b is None:
            return a
        return (a or 0.0) + b

    total.total_usd = _add(total.total_usd, addition.total_usd)
    total.cache_hit_usd = _add(total.cache_hit_usd, addition.cache_hit_usd)
    total.cache_miss_usd = _add(total.cache_miss_usd, addition.cache_miss_usd)
    total.output_usd = _add(total.output_usd, addition.output_usd)

    if total.source == "unknown":
        total.source = addition.source
    elif addition.source == "api":
        total.source = "api"
    elif addition.source == "api_buckets" and total.source not in ("api",):
        total.source = "api_buckets"
    elif addition.source == "fallback" and total.source == "unknown":
        total.source = "fallback"
