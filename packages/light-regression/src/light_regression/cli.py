# ruff: noqa: B008

from pathlib import Path

import typer

from .checker import RegressionChecker
from .dashboard import DashboardGenerator
from .history import HistoryManager
from .models import TestCase
from .runner import RegressionRunner

app = typer.Typer(help="light-subtitle regression testing")


@app.command()
def run(
    case_file: Path = typer.Argument(..., help="Path to case.yaml"),
    snapshots_dir: Path = typer.Option(Path("tests/regression/snapshots"), help="Snapshots directory"),
    keep_subtitle: bool = typer.Option(False, help="Keep generated subtitle"),
):
    """Run a regression test case."""
    case = TestCase.from_yaml(case_file)
    runner = RegressionRunner(snapshots_dir)
    record, diff = runner.run(case, keep_subtitle)

    typer.echo(f"Run ID: {record.run_id}")
    typer.echo(
        f"Duration: {record.duration_sec:.1f}s  "
        f"Errors: {record.report.get('errors', 0)}  "
        f"Warnings: {record.report.get('warnings', 0)}  "
        f"Suggestions: {record.report.get('suggestions', 0)}"
    )

    if diff.degraded:
        typer.secho("\n⚠️  REGRESSION DETECTED", fg=typer.colors.RED, bold=True)
        for reason in diff.reasons:
            typer.secho(f"  • {reason}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    else:
        typer.secho("\n✓ PASS", fg=typer.colors.GREEN, bold=True)


@app.command()
def dashboard(
    snapshots_dir: Path = typer.Option(Path("tests/regression/snapshots")),
    output: Path = typer.Option(Path("regression_dashboard.html")),
):
    """Generate HTML dashboard."""
    generator = DashboardGenerator()
    generator.generate(snapshots_dir, output)
    typer.secho(f"✓ Dashboard: {output.absolute()}", fg=typer.colors.GREEN)


@app.command()
def diff(
    case_file: Path = typer.Argument(..., help="Path to case.yaml"),
    run_a: str = typer.Argument(..., help="First run ID"),
    run_b: str = typer.Argument(..., help="Second run ID"),
    snapshots_dir: Path = typer.Option(Path("tests/regression/snapshots")),
):
    """Compare two runs."""
    case = TestCase.from_yaml(case_file)
    history = HistoryManager(snapshots_dir)

    record_a = history.get_run(case.name, run_a)
    record_b = history.get_run(case.name, run_b)

    checker = RegressionChecker()
    d = checker.compare(record_a, record_b, case.thresholds)

    import json

    typer.echo(
        json.dumps(
            {
                "baseline_run_id": d.baseline_run_id,
                "current_run_id": d.current_run_id,
                "degraded": d.degraded,
                "errors_delta": d.errors_delta,
                "warnings_delta": d.warnings_delta,
                "suggestions_delta": d.suggestions_delta,
                "reasons": d.reasons,
                "rule_changes": d.rule_changes,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
