"""Runtime step plan and resume resolution (built from step_registry)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import SubtitleConfig
from .run_state import RunState, RunStatus
from .step_registry import StepDefinition, build_enabled_definitions


@dataclass(frozen=True)
class PlanStep:
    """One enabled step in an execution plan."""

    id: str
    definition: StepDefinition
    required_artifacts: tuple[Path, ...] = ()


def build_step_plan(config: SubtitleConfig) -> list[PlanStep]:
    """Build ordered step list from the declarative registry."""
    return [
        PlanStep(
            id=definition.id.value,
            definition=definition,
            required_artifacts=definition.artifacts(config),
        )
        for definition in build_enabled_definitions(config)
    ]


def resolve_start_index(
    plan: list[PlanStep],
    config: SubtitleConfig,
    run_state: RunState | None,
) -> int:
    """Return index of the first step to execute."""
    if config.resume_from:
        for i, step in enumerate(plan):
            if step.id == config.resume_from:
                return i
        valid = ", ".join(s.id for s in plan)
        raise ValueError(f"Unknown step {config.resume_from!r}. Valid steps: {valid}")

    if config.resume and run_state is not None:
        target = run_state.failed_step or run_state.current_step
        if run_state.status == RunStatus.INTERRUPTED and run_state.current_step:
            target = run_state.current_step
        if target:
            for i, step in enumerate(plan):
                if step.id == target:
                    return i
        completed = set(run_state.completed_steps)
        for i, step in enumerate(plan):
            if step.id not in completed:
                return i
        return len(plan)  # all steps completed — nothing to do

    if config.resume and run_state is None:
        raise FileNotFoundError("No pipeline_run.json found. Use --resume-from STEP to resume from artifacts.")

    return 0


def validate_artifacts(step: PlanStep) -> None:
    """Ensure required artifacts exist before resuming at *step*."""
    missing = [p for p in step.required_artifacts if p and not p.exists()]
    if missing:
        paths = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Cannot resume at {step.id}: missing artifacts: {paths}")


def list_step_ids(config: SubtitleConfig) -> list[str]:
    return [s.id for s in build_step_plan(config)]
