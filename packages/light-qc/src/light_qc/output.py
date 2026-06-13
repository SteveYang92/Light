import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from light_models import QCReport, SubtitleCue

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HTML_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def to_json(report: QCReport, path: str | None = None) -> str:
    data = {
        "total_cues": report.total_cues,
        "errors": report.errors,
        "warnings": report.warnings,
        "suggestions": report.suggestions,
        "passed": report.passed,
        "bilingual": report.bilingual,
        "source_lang": report.source_lang,
        "target_lang": report.target_lang,
        "issues": [
            {
                "severity": i.severity,
                "category": i.category,
                "rule": i.rule,
                "cue_id": i.cue_id,
                "time": i.time,
                "detail": i.detail,
                "fix": i.fix,
            }
            for i in report.issues
        ],
    }
    content = json.dumps(data, indent=2, ensure_ascii=False)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return content


def to_console(report: QCReport) -> None:
    status = "PASSED" if report.passed else "FAILED"
    status_icon = "✓" if report.passed else "✗"

    print(f"\n{'=' * 60}")
    print(f"  QC Report [{status_icon} {status}]")
    print(f"{'=' * 60}")
    print(f"  Bilingual: {report.bilingual}")
    print(f"  Source: {report.source_lang}  →  Target: {report.target_lang or 'N/A'}")
    print(f"  Total cues: {report.total_cues}")
    print(f"  Errors: {report.errors}  Warnings: {report.warnings}  Suggestions: {report.suggestions}")
    print(f"{'=' * 60}\n")

    if report.issues:
        for issue in report.issues:
            sev_icon = {"error": "✗", "warning": "⚠", "suggestion": "○"}[issue.severity]
            print(f"  [{sev_icon} {issue.severity.upper()}] {issue.rule}")
            if issue.cue_id is not None:
                print(f"    Cue #{issue.cue_id}  @ {issue.time}")
            print(f"    {issue.detail}")
            if issue.fix:
                print(f"    Fix: {issue.fix}")
            print()
    else:
        print("  No issues found.\n")


def to_html(report: QCReport, path: str, cues: dict[str, list[SubtitleCue]] | None = None) -> None:
    """Generate filtered HTML report with expandable cue detail.

    When *cues* is provided, clicking an issue row expands to show the
    current cue (and the previous / next cue) for easy context review.
    Cue data is indexed by 1-based position within the first language list.
    """
    issues_json = json.dumps(
        [
            {
                "severity": i.severity,
                "rule": i.rule,
                "cue_id": i.cue_id,
                "time": i.time or "",
                "detail": i.detail,
                "fix": i.fix or "",
            }
            for i in report.issues
        ],
        ensure_ascii=False,
    )

    rule_names = sorted({i.rule for i in report.issues})
    rules_json = json.dumps(rule_names, ensure_ascii=False)

    # Build cue lookup: 1-based index → {text, start, end}
    cue_texts_json = "null"
    if cues:
        # Use the first language's cue list (most common case)
        for cue_list in cues.values():
            cue_texts = [{"text": c.text, "start": _safe_time(c.start), "end": _safe_time(c.end)} for c in cue_list]
            cue_texts_json = json.dumps(cue_texts, ensure_ascii=False)
            break

    html = _HTML_ENV.get_template("qc_report.html").render(
        passed=report.passed,
        passed_text="PASSED" if report.passed else "FAILED",
        passed_class="failed" if not report.passed else "passed",
        errors=report.errors,
        warnings=report.warnings,
        suggestions=report.suggestions,
        bilingual=report.bilingual,
        source_lang=report.source_lang,
        target_lang=report.target_lang or "N/A",
        total_cues=report.total_cues,
        rules_json=rules_json,
        issues_json=issues_json,
        cue_texts_json=cue_texts_json,
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _safe_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
