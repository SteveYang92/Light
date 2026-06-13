"""Pipeline orchestration — ASR → segmentation → translation → formatting → export."""

from __future__ import annotations

import signal
from collections.abc import Callable
from dataclasses import dataclass, field

from light_models import Segment, SubtitleCue, Word

from . import logger
from .config import SubtitleConfig
from .pipeline.asr import AsrContext
from .pipeline.translate.context import TranslateContext
from .run_state import RunStateManager
from .state_hydrate import hydrate_state
from .step_plan import build_step_plan, resolve_start_index, validate_artifacts

# ── Progress callback type ─────────────────────────────

ProgressCallback = Callable[[str, float, str], None] | None

# ── Shared state passed through pipeline phases ─────────


@dataclass
class PipelineState:
    """Mutable state accumulated during pipeline execution."""

    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    source_lang: str = "en"
    raw_source_cues: list[SubtitleCue] = field(default_factory=list)
    translated_cues: list[SubtitleCue] = field(default_factory=list)
    translation_usage: dict | None = None
    annotations: dict[str, str] = field(default_factory=dict)
    auto_glossary: dict[str, str] = field(default_factory=dict)
    merged_glossary: dict[str, str] = field(default_factory=dict)
    content_summary: dict | None = None


# ── Orchestrator ────────────────────────────────────────


class Orchestrator:
    """Orchestrate the full subtitle pipeline from ASR to export."""

    def __init__(self, config: SubtitleConfig, progress_callback: ProgressCallback = None):
        self.config = config
        self.state = PipelineState()
        self.asr_ctx = AsrContext()
        self.tx_ctx = TranslateContext()
        self._progress = progress_callback or (lambda _s, _p, _m: None)
        self._formatted_source: list[SubtitleCue] | None = None
        self._formatted_target: list[SubtitleCue] | None = None
        self._state_mgr: RunStateManager | None = None

    def run(self) -> None:
        logger.init(self.config.output_dir)

        mode = (
            "bilingual"
            if self.config.bilingual
            else f"translate→{self.config.target_lang}"
            if self.config.target_lang
            else "source-only"
        )
        logger.info(f"[{mode}] Processing {self.config.input_path}")

        self._state_mgr = RunStateManager(self.config.output_dir)
        plan = build_step_plan(self.config)
        run_state = self._state_mgr.load() if (self.config.resume or self.config.resume_from) else None
        start_idx = resolve_start_index(plan, self.config, run_state)

        if not self.config.resume and not self.config.resume_from:
            self._state_mgr.begin(self.config.input_path, self.config.output_dir)
        elif start_idx > 0:
            validate_artifacts(plan[start_idx])
            hydrate_state(self, plan, start_idx)
            if run_state is None:
                self._state_mgr.begin(self.config.input_path, self.config.output_dir)

        self._install_interrupt_handler()

        remaining = plan[start_idx:]
        logger.info(f"── Steps ({len(remaining)} to run) ──")

        for step in remaining:
            self._state_mgr.mark_running(step.id)
            definition = step.definition
            logger.info(f"▶ {step.id}...")
            try:
                if definition.progress_start is not None:
                    definition.progress_start(self)
                definition.run(self)
                if definition.progress_end is not None:
                    definition.progress_end(self)
            except Exception as exc:
                self._state_mgr.mark_failed(step.id, exc)
                raise
            self._state_mgr.mark_completed(step.id)
            logger.info(f"  ✓ {step.id} done")

        self._state_mgr.mark_run_completed()
        logger.info("Done.")

    def _install_interrupt_handler(self) -> None:
        def _handler(_signum: int, _frame: object) -> None:
            if self._state_mgr is not None:
                self._state_mgr.mark_interrupted(self._state_mgr.current_step)
            raise SystemExit(130)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
