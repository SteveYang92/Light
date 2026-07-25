"""Pipeline run state tracking — persisted to pipeline_run.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

RUN_STATE_FILENAME = "pipeline_run.json"
RUN_STATE_VERSION = 1


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class StepRecord:
    status: str = "pending"
    completed_at: str | None = None
    error: str | None = None


@dataclass
class RunState:
    version: int = RUN_STATE_VERSION
    status: str = RunStatus.RUNNING
    input_path: str = ""
    output_dir: str = ""
    started_at: str = ""
    updated_at: str = ""
    current_step: str | None = None
    failed_step: str | None = None
    error: str | None = None
    completed_steps: list[str] = field(default_factory=list)
    steps: dict[str, StepRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = {k: asdict(v) for k, v in self.steps.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        steps_raw = data.pop("steps", {})
        steps = {k: StepRecord(**v) for k, v in steps_raw.items()}
        return cls(steps=steps, **data)


class RunStateManager:
    """Read/write pipeline_run.json in the output directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / RUN_STATE_FILENAME
        self._state: RunState | None = None
        self._current_step: str | None = None

    def load(self) -> RunState | None:
        if not self.path.exists():
            return None
        with open(self.path, encoding="utf-8") as f:
            self._state = RunState.from_dict(json.load(f))
        return self._state

    @property
    def state(self) -> RunState | None:
        return self._state

    def begin(self, input_path: str, output_dir: str) -> None:
        now = _now_iso()
        self._state = RunState(
            input_path=input_path,
            output_dir=output_dir,
            started_at=now,
            updated_at=now,
            status=RunStatus.RUNNING,
        )
        self._flush()

    def mark_running(self, step_id: str) -> None:
        if self._state is None:
            return
        self._current_step = step_id
        self._state.current_step = step_id
        self._state.status = RunStatus.RUNNING
        self._state.failed_step = None
        self._state.error = None
        rec = self._state.steps.setdefault(step_id, StepRecord())
        rec.status = "running"
        self._state.updated_at = _now_iso()
        self._flush()

    def mark_completed(self, step_id: str) -> None:
        if self._state is None:
            return
        if step_id not in self._state.completed_steps:
            self._state.completed_steps.append(step_id)
        rec = self._state.steps.setdefault(step_id, StepRecord())
        rec.status = "completed"
        rec.completed_at = _now_iso()
        rec.error = None
        self._state.updated_at = _now_iso()
        self._flush()

    def mark_failed(self, step_id: str, exc: BaseException) -> None:
        if self._state is None:
            return
        self._state.status = RunStatus.FAILED
        self._state.failed_step = step_id
        self._state.error = str(exc)
        rec = self._state.steps.setdefault(step_id, StepRecord())
        rec.status = "failed"
        rec.error = str(exc)
        self._state.updated_at = _now_iso()
        self._flush()

    def mark_interrupted(self, step_id: str | None = None) -> None:
        if self._state is None:
            return
        self._state.status = RunStatus.INTERRUPTED
        if step_id:
            self._state.failed_step = step_id
            rec = self._state.steps.setdefault(step_id, StepRecord())
            rec.status = "interrupted"
        self._state.updated_at = _now_iso()
        self._flush()

    def mark_run_completed(self) -> None:
        if self._state is None:
            return
        self._state.status = RunStatus.COMPLETED
        self._state.current_step = None
        self._state.updated_at = _now_iso()
        self._flush()

    @property
    def current_step(self) -> str | None:
        return self._current_step

    def _flush(self) -> None:
        if self._state is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, indent=2, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
