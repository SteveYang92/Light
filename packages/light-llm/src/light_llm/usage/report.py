"""Load, merge, and persist pipeline usage reports."""

from __future__ import annotations

import json
from pathlib import Path

from .models import UsageReport
from .tracker import USAGE_REPORT_FILENAME, UsageTracker

__all__ = ["USAGE_REPORT_FILENAME", "load_usage_from_dir", "merge_reports", "write_usage_report"]


def write_usage_report(report: UsageReport, path: str | Path) -> None:
    """Write *report* to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_usage_from_dir(output_dir: str | Path) -> UsageReport | None:
    """Load ``usage_report.json`` if present."""
    path = Path(output_dir) / USAGE_REPORT_FILENAME
    if not path.exists():
        return None
    return UsageReport.from_dict(json.loads(path.read_text(encoding="utf-8")))


def merge_reports(reports: list[UsageReport]) -> UsageReport:
    """Merge segment-level reports into one."""
    if not reports:
        return UsageReport()
    trackers = []
    for report in reports:
        tracker = UsageTracker(model=report.model or reports[0].model)
        tracker._steps = dict(report.steps)
        trackers.append(tracker)
    merged_tracker = UsageTracker.merge(trackers)
    return merged_tracker.build_report()
