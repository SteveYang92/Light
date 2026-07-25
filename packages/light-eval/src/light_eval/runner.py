"""Step runner — execute one pipeline step on one case via the real
capability-package APIs (no mocks).

- ``plan``: :func:`light_subtitle.plan.run` — ``llm=None`` is supported and
  takes the deterministic fallback, so rule metrics stay meaningful without
  an LLM.  Units are persisted to ``<work_dir>/<case>/plan/plan.json``.
- ``translate``: :func:`light_subtitle.translate.translate.run` — with
  ``llm=None`` the case is marked ``skipped`` (empty output, no error).

Exceptions are captured into ``StepOutput.error`` so one bad case never
aborts the whole suite.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from light_llm.client import OpenAIClient
from light_models import Segment, SubtitleCue, word_to_dict
from light_subtitle import artifacts
from light_subtitle import plan as plan_pipeline
from light_subtitle.config import PlanConfig, TranslateConfig
from light_subtitle.translate import translate as translate_step

from .loader import Fixture
from .models import EvalCase, StepOutput

DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"


# ── LLM client ──────────────────────────────────────────────────────────────


def build_llm_client(
    base_url: str = DEFAULT_LLM_BASE_URL,
    model: str = DEFAULT_LLM_MODEL,
    api_key: str = "",
) -> OpenAIClient | None:
    """Build an ``OpenAIClient``; None when no API key is available.

    *api_key* falls back to ``DEEPSEEK_API_KEY`` (mirrors the CLI config).
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return None
    return OpenAIClient(base_url=base_url, api_key=key, model=model)


# ── Serialization (judge- and report-facing) ────────────────────────────────


def unit_to_dict(unit: Segment) -> dict:
    """Serialize a plan unit, keeping word timing for boundary metrics."""
    return {
        "unit_id": unit.unit_id,
        "start": unit.start,
        "end": unit.end,
        "text": unit.source_text,
        "speaker": unit.speaker,
        "words": [word_to_dict(w) for w in unit.words],
    }


def cue_to_dict(cue: SubtitleCue) -> dict:
    """Serialize a translated cue (reuses the pipeline raw.json schema)."""
    return artifacts.cue_to_raw_dict(cue)


# ── Runner ──────────────────────────────────────────────────────────────────


def run_case(
    case: EvalCase,
    fixture: Fixture,
    *,
    llm: OpenAIClient | None = None,
    work_dir: str | Path | None = None,
) -> StepOutput:
    """Run the case's step on *fixture*; capture output, usage, timing, errors."""
    started = time.perf_counter()
    try:
        if case.step == "plan":
            output, usage = _run_plan(case, fixture, llm=llm, work_dir=work_dir)
        elif case.step == "translate":
            result = _run_translate(case, fixture, llm=llm)
            if result is None:
                return StepOutput(
                    case=case.name,
                    duration_s=time.perf_counter() - started,
                    skipped=True,
                )
            output, usage = result
        else:
            raise ValueError(f"unsupported step: {case.step}")
    except Exception as exc:  # one bad case must not abort the suite
        return StepOutput(
            case=case.name,
            duration_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return StepOutput(case=case.name, output=output, usage=usage, duration_s=time.perf_counter() - started)


def _run_plan(
    case: EvalCase,
    fixture: Fixture,
    *,
    llm: OpenAIClient | None,
    work_dir: str | Path | None,
) -> tuple[list[dict], dict | None]:
    config = PlanConfig(
        max_duration=float(case.params.get("max_duration", PlanConfig.max_duration)),
        min_duration=float(case.params.get("min_duration", PlanConfig.min_duration)),
    )
    plan_dir = Path(work_dir or case.case_dir / ".eval_run") / case.name / "plan"
    units, usage = plan_pipeline.run(fixture.segments, config, plan_dir, llm=llm)
    return [unit_to_dict(u) for u in units], usage


def _run_translate(
    case: EvalCase,
    fixture: Fixture,
    *,
    llm: OpenAIClient | None,
) -> tuple[list[dict], dict | None] | None:
    """Translate plan units; None when no LLM client is available (skipped)."""
    if llm is None:
        return None
    config = TranslateConfig(target_lang=str(case.params.get("target_lang", "zh")))
    cues, usage = translate_step.run(
        fixture.segments,
        config,
        tx_dir=None,
        llm=llm,
        glossary=fixture.glossary,
        content_summary=fixture.summary,
    )
    return [cue_to_dict(c) for c in cues], usage
