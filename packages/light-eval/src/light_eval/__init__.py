"""light-eval — 字幕自改进评估框架。

针对字幕管线单步骤（plan / translate）的离线评估：
case 发现与加载（loader）、真实能力包 runner、rule judge（judges.rules）、
LLM defect judge（judges.llm）、JSON/HTML 报告（report）、typer CLI（cli）。
"""

from __future__ import annotations

from .models import Annotation, CaseResult, EvalCase, EvalReport, ProblemTypeStats, StepOutput

__all__ = [
    "Annotation",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "ProblemTypeStats",
    "StepOutput",
]
