"""Usage accumulation, API parsing, and per-step persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TOKEN_INT_KEYS, CostSummary, UsageReport
from .pricing import compute_cost, merge_cost

USAGE_REPORT_FILENAME = "usage_report.json"

# Step artifact paths relative to output_dir.
STEP_USAGE_ARTIFACTS: dict[str, str] = {
    "correct": "transcript_correct/usage.json",
    "punct": "punct_restore/usage.json",
    "context": "context/usage.json",
    "translate.compose_split": "compose/usage.json",
    "translate": "translations/usage.json",
    "annotate": "annotations/usage.json",
}


def parse_api_usage(usage_obj: object | None) -> dict[str, Any]:
    """Serialize ``response.usage`` from the OpenAI SDK into a plain dict."""
    if usage_obj is None:
        return {}

    if hasattr(usage_obj, "model_dump"):
        raw = usage_obj.model_dump(exclude_none=True)
    elif isinstance(usage_obj, dict):
        raw = dict(usage_obj)
    else:
        raw = {}

    result: dict[str, Any] = {}
    token_keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    for key in token_keys:
        if key in raw and raw[key] is not None:
            result[key] = int(raw[key])

    details = raw.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        result["reasoning_tokens"] = int(details["reasoning_tokens"])

    for key in ("cost_usd", "total_cost", "cost"):
        if key in raw and raw[key] is not None:
            result[key] = float(raw[key])

    return result


def merge_token_usage(total: dict[str, Any], usage: dict[str, Any] | None, *, calls: int = 1) -> None:
    """Accumulate token counts (and API cost fields) from one LLM call into *total*."""
    if not usage:
        if calls:
            total["calls"] = int(total.get("calls", 0)) + calls
        return

    for key in TOKEN_INT_KEYS:
        if key == "calls":
            continue
        if key in usage and usage[key] is not None:
            total[key] = int(total.get(key, 0)) + int(usage[key])

    for key in ("cost_usd", "total_cost", "cost"):
        if key in usage and usage[key] is not None:
            total[key] = float(total.get(key, 0.0)) + float(usage[key])

    total["calls"] = int(total.get("calls", 0)) + calls


def format_token_usage(usage: dict[str, Any] | None) -> str:
    """Format token usage for log output."""
    if not usage:
        return "tokens: ?"
    parts = [
        f"tokens: {usage.get('total_tokens', '?')}",
        f"prompt: {usage.get('prompt_tokens', '?')}",
        f"completion: {usage.get('completion_tokens', '?')}",
    ]
    if usage.get("prompt_cache_hit_tokens") is not None:
        parts.append(f"cache_hit: {usage['prompt_cache_hit_tokens']}")
    if usage.get("prompt_cache_miss_tokens") is not None:
        parts.append(f"cache_miss: {usage['prompt_cache_miss_tokens']}")
    return " ".join(parts)


def usage_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Return token fields added between *before* and *after* usage snapshots."""
    if not after:
        return {}
    before = before or {}
    delta: dict[str, Any] = {}
    for key in TOKEN_INT_KEYS:
        if key == "calls":
            continue
        diff = int(after.get(key, 0)) - int(before.get(key, 0))
        if diff:
            delta[key] = diff
    call_diff = int(after.get("calls", 0)) - int(before.get("calls", 0))
    if call_diff:
        delta["calls"] = call_diff
    return delta


def save_step_usage(path: str | Path, data: dict[str, Any]) -> None:
    """Write a step usage artifact as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_step_usage() -> dict[str, Any]:
    return dict.fromkeys(TOKEN_INT_KEYS, 0)


def _merge_step_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    merge_token_usage(target, source, calls=int(source.get("calls", 0)) or 0)


class UsageTracker:
    """Collect per-step token usage and build a pipeline report."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._steps: dict[str, dict[str, Any]] = {}

    def record(self, step_id: str, usage: dict[str, Any] | None, *, calls: int = 1) -> None:
        """Record usage for *step_id* (accumulates across multiple calls)."""
        if step_id not in self._steps:
            self._steps[step_id] = _empty_step_usage()
        merge_token_usage(self._steps[step_id], usage, calls=calls)

    def record_breakdown(self, breakdown: dict[str, dict[str, Any]]) -> None:
        """Record multiple sub-steps at once."""
        for step_id, usage in breakdown.items():
            if usage:
                self.record(step_id, usage, calls=int(usage.get("calls", 1)))

    def load_from_dir(self, output_dir: str | Path) -> None:
        """Hydrate tracker from per-step usage artifacts (resume support)."""
        output_dir = Path(output_dir)
        for step_id, rel_path in STEP_USAGE_ARTIFACTS.items():
            path = output_dir / rel_path
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            breakdown = data.get("breakdown")
            if isinstance(breakdown, dict):
                self.record_breakdown(breakdown)
            else:
                self.record(step_id, data, calls=int(data.get("calls", 0)))

    def build_report(self) -> UsageReport:
        """Build the final usage report with totals and cost."""
        totals = _empty_step_usage()
        cost = CostSummary()

        for step_usage in self._steps.values():
            _merge_step_usage(totals, step_usage)
            step_cost = compute_cost(step_usage, self.model)
            merge_cost(cost, step_cost)

        if cost.total_usd is None and totals.get("calls", 0) == 0:
            cost.source = "unknown"

        return UsageReport(
            model=self.model,
            steps=dict(self._steps),
            totals=totals,
            cost=cost,
        )

    def format_summary(self) -> str:
        """One-line summary for pipeline completion log."""
        report = self.build_report()
        totals = report.totals
        lines = [
            "── Token usage ──",
            f"  Total: {format_token_usage(totals)}",
            f"  LLM calls: {totals.get('calls', 0)}",
        ]
        if report.steps:
            for step_id, usage in sorted(report.steps.items()):
                if usage.get("total_tokens", 0) or usage.get("calls", 0):
                    lines.append(f"  {step_id}: {format_token_usage(usage)}")
        cost = report.cost
        if cost.total_usd is not None:
            lines.append(f"  Est. cost: ${cost.total_usd:.4f} USD ({cost.source})")
        return "\n".join(lines)

    def save_report(self, output_dir: str | Path) -> Path:
        """Write ``usage_report.json`` under *output_dir*."""
        path = Path(output_dir) / USAGE_REPORT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.build_report().to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def merge(cls, trackers: list[UsageTracker]) -> UsageTracker:
        """Merge multiple segment trackers (long-video merge)."""
        if not trackers:
            return cls(model="")
        merged = cls(model=trackers[0].model)
        for tracker in trackers:
            for step_id, usage in tracker._steps.items():
                merged.record(step_id, usage, calls=int(usage.get("calls", 0)))
        return merged
