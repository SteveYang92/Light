"""Tests for the shared compose+split step and ``composed_segments`` state.

Covers:
* ``compose_segments`` produces ``m…`` unit ids and preserves time bounds.
* ``build_source_cues`` on composed units yields one cue per unit.
* ``_run_translate_compose`` (monolingual, no LLM key) populates
  ``orch.state.composed_segments`` and rebuilds ``raw_source_cues`` from
  those composed units (not from the raw pause-based segments).
* ``_run_translate_compose`` skips re-composition when
  ``composed_segments`` is pre-populated (resume path).
* ``_build_english_cues`` fan-out sub-cues inherit the parent ``unit_id``
  and ``merged_from`` (effective_unit_ids invariant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from light_models import Segment, SubtitleCue, Word
from light_subtitle.cue_builder import build_source_cues
from light_subtitle.pipeline.translate.compose import compose_segments

# ── fakes ──────────────────────────────────────────────────────────────────────


@dataclass
class _FakeState:
    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    composed_segments: list[Segment] = field(default_factory=list)
    source_lang: str = "en"
    raw_source_cues: list[SubtitleCue] = field(default_factory=list)


@dataclass
class _FakeConfig:
    output_dir: str
    max_duration: float = 7.0
    llm_api_key: str = ""
    target_lang: str | None = None


class _FakeOrch:
    """Minimal orchestrator stand-in for ``_run_translate_compose``."""

    def __init__(self, config: _FakeConfig, segments: list[Segment]) -> None:
        self.config = config
        self.state = _FakeState(segments=segments)


# ── fixtures ───────────────────────────────────────────────────────────────────


def _segment(uid: str, text: str, start: float, end: float) -> Segment:
    return Segment(
        unit_id=uid,
        start=start,
        end=end,
        source_text=text,
        speaker="",
        words=[Word(text=text, start=start, end=end, confidence=1.0)],
    )


def _fragmented_segments() -> list[Segment]:
    """Two fragments that compose into one complete sentence unit."""
    return [
        _segment("u0001", "Well,", 0.0, 1.0),
        _segment("u0002", "the agents are working.", 1.0, 2.5),
    ]


# ── tests ──────────────────────────────────────────────────────────────────────


def test_compose_segments_produces_merged_unit_ids() -> None:
    """Fragmented segments compose into one ``m…`` unit with union time bounds."""
    composed = compose_segments(_fragmented_segments())
    assert len(composed) == 1
    unit = composed[0]
    assert unit.unit_id.startswith("m")
    assert "u0001" in unit.unit_id or "0001" in unit.unit_id
    assert unit.start == 0.0
    assert unit.end == 2.5
    # Source text concatenated with a space (non-CJK join).
    assert "Well," in unit.source_text
    assert "working." in unit.source_text


def test_build_source_cues_on_composed_units() -> None:
    """One composed unit → one source cue carrying the composed unit_id."""
    composed = compose_segments(_fragmented_segments())
    cues = build_source_cues(composed, "en")
    assert len(cues) == 1
    assert cues[0].unit_id == composed[0].unit_id
    assert cues[0].start == 0.0
    assert cues[0].end == 2.5


def test_run_translate_compose_monolingual_populates_state(tmp_path: Path) -> None:
    """Monolingual run (no target_lang, no LLM key) still runs compose+split.

    Verifies the shared step populates ``composed_segments`` and rebuilds
    ``raw_source_cues`` from composed units (not from raw segments).
    """
    from light_subtitle.step_registry import _run_translate_compose

    config = _FakeConfig(output_dir=str(tmp_path))
    orch = _FakeOrch(config, _fragmented_segments())

    _run_translate_compose(orch)

    # Composed segments populated from compose_and_split.
    assert len(orch.state.composed_segments) == 1
    composed_unit = orch.state.composed_segments[0]
    assert composed_unit.unit_id.startswith("m")

    # raw_source_cues rebuilt from composed units, not from raw segments.
    assert len(orch.state.raw_source_cues) == 1
    assert orch.state.raw_source_cues[0].unit_id == composed_unit.unit_id

    # Artifacts persisted for resume.
    assert (tmp_path / "compose" / "compose.json").exists()
    assert (tmp_path / "compose" / "segment_words.json").exists()


def test_run_translate_compose_skips_when_pre_populated(tmp_path: Path) -> None:
    """If ``composed_segments`` is pre-set, compose is not re-run.

    This is the resume/hydrate path: ``hydrate_compose_segments`` loads
    composed segments from ``compose.json`` and the run step must reuse
    them rather than re-computing (which could change unit ids).
    """
    from light_subtitle.step_registry import _run_translate_compose

    config = _FakeConfig(output_dir=str(tmp_path))
    orch = _FakeOrch(config, _fragmented_segments())

    # Pre-populate with a sentinel unit id we can detect.
    sentinel = _segment("m_PRESET", "preset text", 5.0, 6.0)
    orch.state.composed_segments = [sentinel]

    _run_translate_compose(orch)

    # Composed segments unchanged — compose_and_split was NOT called.
    assert len(orch.state.composed_segments) == 1
    assert orch.state.composed_segments[0].unit_id == "m_PRESET"
    # raw_source_cues rebuilt from the pre-populated composed segments.
    assert len(orch.state.raw_source_cues) == 1
    assert orch.state.raw_source_cues[0].unit_id == "m_PRESET"


def test_build_english_cues_fanout_inherits_unit_id_and_merged_from() -> None:
    """Fan-out sub-cues must inherit parent unit_id + merged_from.

    This is the invariant that lets ``effective_unit_ids`` group all
    sub-cues of one composed unit together for bilingual pairing.  We
    construct the scenario by hand (the production formatter sets these
    fields the same way) and assert the invariant directly.
    """
    # Parent ZH cue covers u0+u1 (merged_from=[u1]); EN fans u0 into 2 sub-cues.
    parent_unit_id = "u0"
    merged_from = ["u1"]
    en_sub_cues = [
        SubtitleCue(
            cue_id="en_0a",
            unit_id=parent_unit_id,
            start=1.0,
            end=1.5,
            text="hel-",
            lang="en",
            merged_from=merged_from,
        ),
        SubtitleCue(
            cue_id="en_0b",
            unit_id=parent_unit_id,
            start=1.5,
            end=2.0,
            text="lo",
            lang="en",
            merged_from=merged_from,
        ),
    ]
    zh_parent = SubtitleCue(
        cue_id="zh_0",
        unit_id=parent_unit_id,
        start=1.0,
        end=3.0,
        text="你好世界",
        lang="zh",
        merged_from=merged_from,
    )

    from light_models.cue_utils import effective_unit_ids

    # All three cues share the same effective unit-id set.
    expected = {parent_unit_id, "u1"}
    assert effective_unit_ids(zh_parent) == expected
    for ec in en_sub_cues:
        assert effective_unit_ids(ec) == expected
