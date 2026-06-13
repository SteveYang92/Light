# ruff: noqa: B008

import os

import typer

from .config import QCConfig
from .engine import run_qc
from .input import load
from .output import to_console, to_html, to_json

app = typer.Typer()


@app.command()
def check(
    input_path: list[str] = typer.Option(..., "-i", "--input", help="Input subtitle files"),
    bilingual: bool = typer.Option(False, "--bilingual", help="Bilingual mode (both source and target subtitles)"),
    source_lang: str = typer.Option("auto", "--source-lang"),
    target_lang: str = typer.Option("auto", "--target-lang"),
    llm: bool = typer.Option(False, "--llm", help="Enable LLM QC"),
    llm_base_url: str = typer.Option("https://api.openai.com/v1", "--llm-base-url"),
    llm_model: str = typer.Option("gpt-4o", "--llm-model"),
    llm_api_key: str = typer.Option("", "--llm-api-key"),
    output: str = typer.Option("", "-o", "--output", help="Output path for report"),
    format: str = typer.Option("console", "-f", "--format", help="Output format: console, json, html"),
    max_lines: int = typer.Option(2, "--max-lines"),
    max_lines_zh: int = typer.Option(1, "--max-lines-zh", help="Max lines per cue for Chinese"),
    max_chars_zh: int = typer.Option(40, "--max-chars-zh"),
    max_chars_en: int = typer.Option(42, "--max-chars-en"),
    cps_limit: int = typer.Option(9, "--cps-limit"),
    cps_limit_en: int = typer.Option(25, "--cps-limit-en"),
    min_duration: float = typer.Option(0.8, "--min-duration"),
    max_duration: float = typer.Option(7.0, "--max-duration"),
    min_gap: float = typer.Option(0.1, "--min-gap"),
    glossary: str = typer.Option("", "--glossary", help="Path to YAML glossary"),
    fps: float = typer.Option(25.0, "--fps", help="Video frame rate for tolerance calculation"),
    entry_tolerance_frames: int = typer.Option(
        3, "--entry-tolerance-frames", help="Max frame offset for entry point accuracy"
    ),
    # ── Transcript alignment ──
    transcript: str = typer.Option("", "-t", "--transcript", help="Path to transcript.json (light-transcript.v1)"),
    alignment_tolerance: float = typer.Option(
        0.12,
        "--alignment-tolerance",
        help="Word→cue alignment tolerance in seconds (default 2 frames @ 25fps)",  # noqa: E501
    ),
    word_coverage_min: float = typer.Option(0.95, "--word-coverage-min", help="Minimum transcript word coverage ratio"),
):
    glossary_dict = {}
    if glossary:
        import yaml

        with open(glossary) as f:
            glossary_dict = yaml.safe_load(f) or {}

    # Fallback API key: CLI arg → DEEPSEEK_API_KEY → OPENAI_API_KEY
    api_key = llm_api_key or os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    config = QCConfig(
        bilingual=bilingual,
        source_lang=source_lang,
        target_lang=target_lang if target_lang != "auto" else None,
        llm_enabled=llm,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=api_key,
        max_lines=max_lines,
        max_lines_zh=max_lines_zh,
        max_chars_per_line_zh=max_chars_zh,
        max_chars_per_line_en=max_chars_en,
        cps_limit=cps_limit,
        cps_limit_en=cps_limit_en,
        min_duration=min_duration,
        max_duration=max_duration,
        min_gap=min_gap,
        fps=fps,
        entry_tolerance_frames=entry_tolerance_frames,
        glossary=glossary_dict,
        transcript_path=transcript if transcript else None,
        alignment_tolerance=alignment_tolerance,
        word_coverage_min=word_coverage_min,
    )

    cues = load(input_path)
    report = run_qc(cues, config)

    if format == "json":
        out_path = output or "qc_report.json"
        to_json(report, out_path)
        typer.echo(f"Report written to {out_path}")
    elif format == "html":
        out_path = output or "qc_report.html"
        to_html(report, out_path, cues)
        typer.echo(f"Report written to {out_path}")
    else:
        to_console(report)
        if output:
            to_json(report, output)
            typer.echo(f"Report also written to {output}")


def main():
    app()
