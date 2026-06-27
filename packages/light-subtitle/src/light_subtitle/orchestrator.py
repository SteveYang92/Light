"""Pipeline orchestration — ASR → segmentation → translation → formatting → export."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from light_models import Segment, SubtitleCue, Word

from . import logger
from .config import SubtitleConfig
from .pipeline.asr import AsrContext
from .pipeline.translate.context import TranslateContext
from .run_state import RunStateManager
from .state_hydrate import hydrate_state
from .step_plan import build_step_plan, resolve_start_index, validate_artifacts
from .step_registry import ASR_STEP_IDS
from .usage.tracker import UsageTracker

# ── Progress callback type ─────────────────────────────

ProgressCallback = Callable[[str, float, str], None] | None

# ── Shared state passed through pipeline phases ─────────


@dataclass
class PipelineState:
    """Mutable state accumulated during pipeline execution."""

    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    # Composed translation units — shared between English source formatting
    # and the translate pipeline.  Built by the compose step from
    # ``segments`` (raw pause-based units) so both tracks share the same
    # ``unit_id`` graph (``m…`` / ``mu…_N``) for bilingual alignment.
    composed_segments: list[Segment] = field(default_factory=list)
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

    def __init__(
        self,
        config: SubtitleConfig,
        progress_callback: ProgressCallback = None,
        on_asr_complete: Callable[[], None] | None = None,
        shutdown_event: threading.Event | None = None,
    ):
        self.config = config
        self.state = PipelineState()
        self.asr_ctx = AsrContext()
        self.tx_ctx = TranslateContext()
        self._progress = progress_callback or (lambda _s, _p, _m: None)
        self._formatted_source: list[SubtitleCue] | None = None
        self._formatted_target: list[SubtitleCue] | None = None
        self._state_mgr: RunStateManager | None = None
        self._on_asr_complete = on_asr_complete or (lambda: None)
        self._shutdown = shutdown_event or threading.Event()  # never-set sentinel

    @property
    def _seg_tag(self) -> str:
        """Segment label extracted from output_dir (e.g. 'seg1', 'chunk_2')."""
        name = Path(self.config.output_dir).name
        if name.startswith(".seg"):
            return name[1:]
        if name.startswith("chunk_"):
            return name
        return ""

    def run(self) -> None:
        logger.init(self.config.output_dir)

        tag = f"[{self._seg_tag}] " if self._seg_tag else ""

        mode = (
            "bilingual"
            if self.config.bilingual
            else f"translate→{self.config.target_lang}"
            if self.config.target_lang
            else "source-only"
        )
        logger.info(f"{tag}[{mode}] Processing {self.config.input_path}")

        self.usage_tracker = UsageTracker(model=self.config.llm_model)
        self._state_mgr = RunStateManager(self.config.output_dir)
        plan = build_step_plan(self.config)
        run_state = self._state_mgr.load() if (self.config.resume or self.config.resume_from) else None
        start_idx = resolve_start_index(plan, self.config, run_state)

        if not self.config.resume and not self.config.resume_from:
            self._state_mgr.begin(self.config.input_path, self.config.output_dir)
        elif start_idx > 0:
            if start_idx < len(plan):
                validate_artifacts(plan[start_idx])
            hydrate_state(self, plan, start_idx)
            if run_state is None:
                self._state_mgr.begin(self.config.input_path, self.config.output_dir)
            self.usage_tracker.load_from_dir(self.config.output_dir)

        self._install_interrupt_handler()

        asr_values = {e.value for e in ASR_STEP_IDS}
        remaining = plan[start_idx:]

        # Pre-fire: if ASR is already done (resume from post-ASR, or all
        # steps complete), signal immediately so the next segment can start
        # its ASR without waiting for an ASR→post-ASR boundary transition
        # that will never happen.
        if not remaining or remaining[0].id not in asr_values:
            self._on_asr_complete()
            if not remaining:
                logger.info(f"{tag}Done (already complete).")
                self._finalize_usage_report(tag)
                return

        logger.info(f"{tag}── Steps ({len(remaining)} to run) ──")

        for i, step in enumerate(remaining):
            if self._shutdown.is_set():
                self._state_mgr.mark_interrupted(self._state_mgr.current_step)
                logger.info(f"{tag}  Interrupted.")
                return

            self._state_mgr.mark_running(step.id)
            definition = step.definition
            logger.info(f"{tag}▶ {step.id}...")
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
            logger.info(f"{tag}  ✓ {step.id} done")

            # Fire callback at the ASR → post-ASR boundary
            if step.id in asr_values:
                next_step = remaining[i + 1] if i + 1 < len(remaining) else None
                if next_step is None or next_step.id not in asr_values:
                    self._on_asr_complete()

        self._state_mgr.mark_run_completed()
        self._finalize_usage_report(tag)
        logger.info(f"{tag}Done.")

    def _finalize_usage_report(self, tag: str) -> None:
        """Write usage_report.json and log a summary."""
        if not self.usage_tracker._steps:
            self.usage_tracker.load_from_dir(self.config.output_dir)
        if not self.usage_tracker._steps:
            return
        self.usage_tracker.save_report(self.config.output_dir)
        for line in self.usage_tracker.format_summary().splitlines():
            logger.info(f"{tag}{line}")

    def _install_interrupt_handler(self) -> None:
        """Install SIGINT/SIGTERM handlers (main thread only)."""
        if threading.current_thread() is not threading.main_thread():
            return

        def _handler(_signum: int, _frame: object) -> None:
            if self._state_mgr is not None:
                self._state_mgr.mark_interrupted(self._state_mgr.current_step)
            raise SystemExit(130)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
