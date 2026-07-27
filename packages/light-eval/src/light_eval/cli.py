# ruff: noqa: B008
"""light-eval CLI — run eval suites against the subtitle pipeline steps.

``run`` executes a suite (rule judges + optional LLM judge) and persists
per-case outputs; ``harvest`` lists harvestable candidates from real
pipeline runs; ``serve`` launches the eval workbench web UI.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import loader
from .judges.rules import judge_for_step
from .models import VALID_STEPS, CaseResult, EvalReport
from .report import save_html
from .runner import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, build_llm_client, run_case

app = typer.Typer(help="Subtitle self-improvement eval framework.")


@app.command()
def run(
    suite_dir: Path = typer.Argument(..., help="Suite root containing <step>/<case_name>/ dirs"),
    step: str | None = typer.Option(None, "--step", help=f"Restrict to one step: {', '.join(VALID_STEPS)}"),
    llm_base_url: str = typer.Option(DEFAULT_LLM_BASE_URL, "--llm-base-url"),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, "--llm-model"),
    llm_api_key: str = typer.Option("", "--llm-api-key", help="Falls back to env DEEPSEEK_API_KEY"),
    work_dir: Path | None = typer.Option(
        None, "--work-dir", help="Where step artifacts are written (default: in-case)"
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Report output path"),
    format: str = typer.Option("json", "-f", "--format", help="Report format: json | html"),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Also score with the LLM judge (needs API key)"),
) -> None:
    """Run a case suite through one pipeline step and score it with rule judges."""
    from .judges.llm import LLMJudge

    if step is not None and step not in VALID_STEPS:
        raise typer.BadParameter(f"--step must be one of: {', '.join(VALID_STEPS)}")
    if format not in ("json", "html"):
        raise typer.BadParameter("--format must be json or html")

    cases = loader.discover_cases(suite_dir, step=step)
    if not cases:
        typer.echo(f"No cases found under {suite_dir}")
        raise typer.Exit(code=1)

    llm = build_llm_client(base_url=llm_base_url, model=llm_model, api_key=llm_api_key)
    if llm is None:
        typer.echo("No LLM API key — plan uses deterministic fallback, translate cases are skipped")
    llm_judge = LLMJudge(llm) if judge and llm is not None else None

    report = EvalReport(step=step or "all")
    for case in cases:
        fixture = loader.load_fixture(case)
        step_output = run_case(case, fixture, llm=llm, work_dir=work_dir)
        # Persist per-case output so `calibrate` can re-judge without re-running.
        import json

        out_path = case.case_dir / ".eval_run" / "output.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(step_output.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        scores = judge_for_step(case.step).score(case, fixture, step_output)
        if llm_judge is not None:
            scores += llm_judge.score(case, fixture, step_output)
        result = CaseResult(case=case, scores=scores, output=step_output)
        report.cases.append(result)
        if step_output.error:
            status = "ERROR"
        elif step_output.skipped:
            status = "SKIPPED"
        else:
            status = "PASS" if result.passed else "FAIL"
        typer.echo(f"[{status}] {case.step}/{case.name}")

    agg = report.aggregate()
    typer.echo(f"\n{cases_summary(agg)}")

    if output is not None:
        saved = save_html(report, output) if format == "html" else report.save(output)
        typer.echo(f"Report written to {saved}")
    elif format == "json":
        typer.echo(report.to_json())


def cases_summary(agg: dict) -> str:
    """One-line aggregate summary for the console."""
    dims = ", ".join(
        f"{pt} {counts['passed']}/{counts['total']}" for pt, counts in sorted(agg["problem_types"].items())
    )
    return (
        f"cases: {agg['n_passed']}/{agg['n_cases']} passed"
        f" (errored: {agg['n_errored']}, skipped: {agg['n_skipped']})" + (f" · {dims}" if dims else "")
    )


@app.command()
def harvest(
    output_dir: Path = typer.Argument(Path("output"), help="Run output directory to scan"),
    as_json: bool = typer.Option(False, "--json", help="Print candidates as JSON instead of a table"),
) -> None:
    """List harvestable eval candidates from real pipeline runs."""
    from .harvest import format_table, scan_candidates

    candidates = scan_candidates(output_dir)
    if as_json:
        import json

        typer.echo(json.dumps([c.to_dict() for c in candidates], ensure_ascii=False, indent=2))
        return
    if not candidates:
        typer.echo(f"No candidates found under {output_dir}")
        return
    typer.echo(format_table(candidates))
    typer.echo(f"\n{len(candidates)} candidate(s) — use `light-eval serve` to build cases interactively")


@app.command()
def serve(
    suite_dir: Path = typer.Option(Path("tests/eval"), "--suite-dir", help="Case suite root (<step>/<case_name>/)"),
    output_dir: list[Path] = typer.Option(
        [], "--output-dir", help="Run output directories scanned for candidates (can repeat)"
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8788, "--port"),
) -> None:
    """Launch the eval workbench web UI (candidates / cases / annotation)."""
    import uvicorn

    from .serve import create_app

    suite_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = [str(d) for d in output_dir] if output_dir else []
    typer.echo(f"eval workbench: http://{host}:{port}  (suite: {suite_dir}, candidate dirs: {output_dirs or 'none'})")
    uvicorn.run(create_app(suite_dir=suite_dir, output_dirs=output_dirs), host=host, port=port)
