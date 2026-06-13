"""Tests for pipeline resume state and step planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from light_subtitle.config import AsrEngine, SubtitleConfig
from light_subtitle.run_state import RunStateManager, RunStatus
from light_subtitle.step_plan import build_step_plan, resolve_start_index, validate_artifacts


def _config(**kwargs) -> SubtitleConfig:
    defaults = {
        "input_path": "test.wav",
        "output_dir": "./output",
        "target_lang": None,
    }
    defaults.update(kwargs)
    return SubtitleConfig(**defaults)


def test_build_step_plan_source_only():
    plan = build_step_plan(_config())
    ids = [s.id for s in plan]
    assert ids[0] == "asr.extract"
    assert "translate.compose" not in ids
    assert ids[-2:] == ["subtitle", "export"]


def test_build_step_plan_whisper_cpp_includes_align():
    plan = build_step_plan(_config(asr=AsrEngine.WHISPER_CPP))
    ids = [s.id for s in plan]
    assert "asr.align" in ids


def test_build_step_plan_whisperx_skips_align():
    plan = build_step_plan(_config(asr=AsrEngine.WHISPERX))
    ids = [s.id for s in plan]
    assert "asr.align" not in ids


def test_resolve_resume_from(tmp_path: Path):
    config = _config(output_dir=str(tmp_path), resume_from="segment")
    plan = build_step_plan(config)
    idx = resolve_start_index(plan, config, None)
    assert plan[idx].id == "segment"


def test_resolve_resume_from_unknown_step(tmp_path: Path):
    config = _config(output_dir=str(tmp_path), resume_from="bogus")
    plan = build_step_plan(config)
    with pytest.raises(ValueError, match="Unknown step"):
        resolve_start_index(plan, config, None)


def test_run_state_manager_lifecycle(tmp_path: Path):
    mgr = RunStateManager(tmp_path)
    mgr.begin("in.wav", str(tmp_path))
    mgr.mark_running("asr.extract")
    mgr.mark_completed("asr.extract")
    mgr.mark_failed("asr.transcribe", RuntimeError("boom"))

    state = mgr.load()
    assert state is not None
    assert state.status == RunStatus.FAILED
    assert state.failed_step == "asr.transcribe"
    assert "asr.extract" in state.completed_steps


def test_validate_artifacts_missing(tmp_path: Path):
    config = _config(output_dir=str(tmp_path), resume_from="correct")
    plan = build_step_plan(config)
    step = next(s for s in plan if s.id == "correct")
    with pytest.raises(FileNotFoundError):
        validate_artifacts(step)


def test_validate_artifacts_present(tmp_path: Path):
    config = _config(output_dir=str(tmp_path), resume_from="correct")
    (tmp_path / "transcript.json").write_text(
        json.dumps({"words": [{"text": " hi", "start": 0, "end": 1, "confidence": 0.9, "speaker": None}]})
    )
    plan = build_step_plan(config)
    step = next(s for s in plan if s.id == "correct")
    validate_artifacts(step)


def test_hydrate_words_from_transcript(tmp_path: Path):
    from light_subtitle.state_hydrate import hydrate_pipeline_state

    words_data = {"words": [{"text": " hi", "start": 0.0, "end": 0.5, "confidence": 0.9, "speaker": None}]}
    (tmp_path / "transcript.json").write_text(json.dumps(words_data))

    class _State:
        words: list = []
        segments: list = []
        raw_source_cues: list = []
        source_lang: str = "en"
        auto_glossary: dict = {}
        merged_glossary: dict = {}
        content_summary = None
        translated_cues: list = []
        translation_usage = None

    state = _State()
    config = _config(output_dir=str(tmp_path))
    hydrate_pipeline_state(state, config, "correct")
    assert len(state.words) == 1
    assert state.words[0].text == " hi"


def test_step_registry_unique_ids():
    from light_subtitle.step_registry import build_step_definitions

    ids = [d.id for d in build_step_definitions(_config())]
    assert len(ids) == len(set(ids))


def test_step_registry_hydrate_before_segment():
    from light_subtitle.step_registry import StepId, build_step_definitions

    defs = {d.id: d for d in build_step_definitions(_config())}
    assert defs[StepId.ASR_EXTRACT].hydrate is not None
    assert defs[StepId.CORRECT].hydrate is not None
    assert defs[StepId.SEGMENT].hydrate is not None
    assert defs[StepId.TRANSLATE_COMPOSE].hydrate is not None


def test_hydrate_state_replays_plan_prefix(tmp_path: Path):
    from light_subtitle.orchestrator import PipelineState
    from light_subtitle.state_hydrate import hydrate_state
    from light_subtitle.step_plan import build_step_plan
    from light_subtitle.step_registry import StepId

    words_data = {"words": [{"text": " hi", "start": 0.0, "end": 0.5, "confidence": 0.9, "speaker": None}]}
    (tmp_path / "transcript.json").write_text(json.dumps(words_data))

    config = _config(output_dir=str(tmp_path), resume_from="correct")
    plan = build_step_plan(config)
    start_idx = resolve_start_index(plan, config, None)

    class _Orch:
        state = PipelineState()
        asr_ctx = type("ctx", (), {"audio_path": "", "words": []})()
        tx_ctx = type("ctx", (), {"translation_segments": [], "translated_cues": [], "usage": None})()

    orch = _Orch()
    orch.config = config
    hydrate_state(orch, plan, start_idx)
    assert len(orch.state.words) == 1
    assert plan[start_idx].id == StepId.CORRECT.value
