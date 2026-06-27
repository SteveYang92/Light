"""Data models for pipeline token usage reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

USAGE_REPORT_VERSION = 1

# Integer token fields merged across LLM calls.
TOKEN_INT_KEYS = (
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "calls",
)

# Direct cost fields from API usage objects (USD).
API_COST_KEYS = ("cost_usd", "total_cost", "cost")


@dataclass
class CostSummary:
    """Aggregated cost for a usage report."""

    total_usd: float | None = None
    cache_hit_usd: float | None = None
    cache_miss_usd: float | None = None
    output_usd: float | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_usd": self.total_usd,
            "cache_hit_usd": self.cache_hit_usd,
            "cache_miss_usd": self.cache_miss_usd,
            "output_usd": self.output_usd,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostSummary:
        return cls(
            total_usd=data.get("total_usd"),
            cache_hit_usd=data.get("cache_hit_usd"),
            cache_miss_usd=data.get("cache_miss_usd"),
            output_usd=data.get("output_usd"),
            source=str(data.get("source", "unknown")),
        )


@dataclass
class UsageReport:
    """Pipeline-level token and cost summary."""

    version: int = USAGE_REPORT_VERSION
    model: str = ""
    currency: str = "USD"
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)
    cost: CostSummary = field(default_factory=CostSummary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model": self.model,
            "currency": self.currency,
            "steps": self.steps,
            "totals": self.totals,
            "cost": self.cost.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageReport:
        cost_raw = data.get("cost") or {}
        return cls(
            version=int(data.get("version", USAGE_REPORT_VERSION)),
            model=str(data.get("model", "")),
            currency=str(data.get("currency", "USD")),
            steps=dict(data.get("steps") or {}),
            totals=dict(data.get("totals") or {}),
            cost=CostSummary.from_dict(cost_raw if isinstance(cost_raw, dict) else {}),
        )
