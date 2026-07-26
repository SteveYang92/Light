"""Report rendering — JSON (for agents) lives on ``EvalReport`` itself;
this module renders the single-file HTML report (for humans)."""

from __future__ import annotations

import html
from pathlib import Path

from .models import EvalReport

_PASS_MARK = '<span class="pass">PASS</span>'
_FAIL_MARK = '<span class="fail">FAIL</span>'

_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>light-eval report — {step}</title>
<style>
  body {{ font-family: -apple-system, "Helvetica Neue", sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; font-size: 0.9rem; }}
  th {{ background: #f5f5f5; }}
  .pass {{ color: #1a7f37; font-weight: 600; }}
  .fail {{ color: #cf222e; font-weight: 600; }}
  .err {{ color: #9a6700; font-weight: 600; }}
  .evidence {{ color: #666; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>light-eval report — step: {step}</h1>
<div class="meta">created_at: {created_at} · cases: {n_cases} · passed: {n_passed} ·
errored: {n_errored} · skipped: {n_skipped}</div>
{dim_table}
{case_tables}
</body>
</html>
"""


def to_html(report: EvalReport) -> str:
    """Render *report* as a standalone HTML page."""
    agg = report.aggregate()
    return _PAGE.format(
        step=html.escape(report.step),
        created_at=html.escape(report.created_at),
        n_cases=agg["n_cases"],
        n_passed=agg["n_passed"],
        n_errored=agg["n_errored"],
        n_skipped=agg["n_skipped"],
        dim_table=_dimension_table(agg["dimensions"]),
        case_tables="\n".join(_case_table(result) for result in report.cases),
    )


def save_html(report: EvalReport, path: str | Path) -> Path:
    """Write the HTML report to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_html(report), encoding="utf-8")
    return path


def _dimension_table(dimensions: dict[str, dict[str, int]]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(dim)}</td><td>{counts['passed']}/{counts['total']}</td></tr>"
        for dim, counts in sorted(dimensions.items())
    )
    return f"<table><tr><th>dimension</th><th>passed</th></tr>{rows}</table>"


def _case_table(result) -> str:
    case = result.case
    if result.output.error:
        status = f'<span class="err">ERROR</span> {html.escape(result.output.error)}'
    elif result.output.skipped:
        status = '<span class="err">SKIPPED</span> (no LLM client)'
    else:
        status = _PASS_MARK if result.passed else _FAIL_MARK
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(score.dimension)}</td>"
        f"<td>{_PASS_MARK if score.passed else _FAIL_MARK}</td>"
        f"<td>{score.score:.4g}</td>"
        f"<td>{html.escape(score.detail)}"
        + (
            f'<div class="evidence">must_fix ×{sum(1 for i in score.issues if i.get("severity") != "minor")}</div>'
            if any(i.get("severity") != "minor" for i in score.issues)
            else ""
        )
        + (f'<div class="evidence">{html.escape("; ".join(score.evidence))}</div>' if score.evidence else "")
        + "</td></tr>"
        for score in result.scores
    )
    header = f'<h2>{html.escape(case.name)} <span class="meta">[{html.escape(case.kind)}] {status}</span></h2>'
    if not result.scores:
        return header
    return header + f"<table><tr><th>dimension</th><th>verdict</th><th>score</th><th>detail</th></tr>{rows}</table>"
