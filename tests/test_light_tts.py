from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from light_tts.assemble import PlacedSegment, assemble_timeline
from light_tts.audio_io import concat_with_crossfade, read_wav, trim_edge_silence, write_wav
from light_tts.config import AlignMode, EngineMode, MixMode, TtsConfig
from light_tts.cues_loader import Cue, find_cues_json, load_cues, resolve_cues_path, resolve_dub_cues_path
from light_tts.dub import _preview_timeline_duration, run_dub, run_mix_only
from light_tts.indextts_runtime import (
    INDEXTTS2_SAMPLE_RATE,
    INDEXTTS15_SAMPLE_RATE,
    resolve_ref_audio_path,
    resolve_torch_compile,
    variant_spec,
)
from light_tts.merge_turns import SpeakerTurn, merge_speaker_turns
from light_tts.mix import find_video
from light_tts.speaker_map import build_indextts_speaker_map, build_speaker_voice_map, voice_for_cue
from light_tts.sync import compute_subtitle_aligned_start, compute_turn_placed_start, fit_budget, fit_duration

FIXTURES = Path(__file__).parent / "fixtures"
CUES_JSON = FIXTURES / "tts_cues.json"


def test_merge_speaker_turns_groups_consecutive_speakers() -> None:
    cues = [
        Cue("a", 0.0, 1.0, "你好", "SPEAKER_00", "zh"),
        Cue("b", 1.1, 2.0, "世界", "SPEAKER_00", "zh"),
        Cue("c", 2.1, 3.0, "嗯", "SPEAKER_01", "zh"),
    ]
    turns = merge_speaker_turns(cues, speaker_gap_s=0.08)
    assert len(turns) == 2
    assert turns[0].text == "你好世界。"
    assert turns[0].cue_ids == ("a", "b")
    assert turns[0].slot_end == pytest.approx(2.02, abs=0.01)
    assert turns[1].speaker == "SPEAKER_01"


def test_merge_speaker_turns_alternating_keeps_count() -> None:
    """Fixture cues alternate speakers — one turn per display cue."""
    cues = load_cues(CUES_JSON, lang="zh")
    turns = merge_speaker_turns(cues)
    assert len(turns) == len(cues)


def test_merge_speaker_turns_splits_long_monologue() -> None:
    cues = [Cue(f"c{i:02d}", float(i * 10), float(i * 10 + 9), "你好", "", "zh") for i in range(20)]
    turns = merge_speaker_turns(cues, max_turn_duration_s=90.0)
    assert len(turns) > 1
    assert all(t.slot_duration <= 91.0 for t in turns[:-1])


def test_merge_speaker_turns_splits_by_qwen_chars() -> None:
    cues = [Cue(f"c{i:02d}", float(i), float(i + 0.8), "这是一小段中文", "", "zh") for i in range(10)]
    turns = merge_speaker_turns(cues, max_turn_duration_s=None, max_turn_chars=30, min_turn_chars=10)
    assert len(turns) > 1
    assert all(len(t.text) <= 40 for t in turns)
    assert all(t.text.endswith("。") for t in turns)


def test_resolve_cues_path_accepts_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    cues_file = run_dir / "cues.json"
    cues_file.write_text(CUES_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    assert resolve_cues_path(run_dir) == cues_file.resolve()


def test_find_cues_json_helpful_when_parent_has_many_runs(tmp_path: Path) -> None:
    root = tmp_path / "output"
    for name in ("run_a", "run_b"):
        seg = root / name / ".seg1"
        seg.mkdir(parents=True)
        (seg / "cues.json").write_text('{"cues": []}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Point to one pipeline run"):
        find_cues_json(root)


def test_load_cues_filters_lang() -> None:
    cues = load_cues(CUES_JSON, lang="zh")
    assert len(cues) == 3
    assert all(c.lang == "zh" for c in cues)


def test_resolve_dub_cues_path_requires_raw_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "cues.json").write_text(
        json.dumps({"cues": [{"cue_id": "en1", "start": 0, "end": 1, "text": "hi", "lang": "en"}]}),
        encoding="utf-8",
    )
    raw = run_dir / "translations" / "raw.json"
    raw.parent.mkdir()
    raw.write_text(
        json.dumps({"cues": [{"cue_id": "zh1", "start": 0, "end": 1, "text": "你好。", "lang": "zh"}]}),
        encoding="utf-8",
    )
    assert resolve_dub_cues_path(run_dir, lang="zh") == raw.resolve()
    with pytest.raises(ValueError, match="lang='en'"):
        resolve_dub_cues_path(run_dir, lang="en")


def test_build_indextts_speaker_map() -> None:
    cues = [
        Cue("a", 0.0, 1.0, "你好", "SPEAKER_00", "zh"),
        Cue("b", 1.0, 2.0, "世界", "SPEAKER_00", "zh"),
        Cue("c", 2.0, 3.0, "嗯", "SPEAKER_01", "zh"),
    ]
    mapping = build_indextts_speaker_map(cues)
    assert mapping == {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"}


def test_resolve_ref_audio_path_prefers_speaker_refs(tmp_path: Path) -> None:
    ref = tmp_path / "dan.wav"
    ref.write_bytes(b"x")
    cfg = TtsConfig(
        output_dir=str(tmp_path / "out"),
        indextts_speaker_refs={"SPEAKER_00": str(ref)},
    )
    assert resolve_ref_audio_path(cfg, "SPEAKER_00") == ref.resolve()


def test_resolve_torch_compile_disabled_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Mps:
        def is_available(self) -> bool:
            return True

    class _Backends:
        mps = _Mps()

    fake_torch = type("Torch", (), {"backends": _Backends()})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_torch_compile(True) is False


def test_config_indextts2_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "indextts2.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "engine: indextts2",
                "ref_audio: /tmp/ref.wav",
                "emotion: calm",
                "indextts_chunk_chars: 120",
            ]
        ),
        encoding="utf-8",
    )
    cfg = TtsConfig.from_yaml(cfg_path, output_dir=str(tmp_path))
    assert cfg.engine_mode == EngineMode.INDEXTTS2
    assert cfg.indextts_resolved_version == "2.0"
    assert cfg.indextts_ref_audio == "/tmp/ref.wav"
    assert cfg.indextts_chunk_chars == 120


def test_config_indextts15_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "indextts15.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "engine: indextts15",
                "ref_audio: /tmp/ref15.wav",
            ]
        ),
        encoding="utf-8",
    )
    cfg = TtsConfig.from_yaml(cfg_path, output_dir=str(tmp_path))
    assert cfg.engine_mode == EngineMode.INDEXTTS15
    assert cfg.indextts_resolved_version == "1.5"
    assert cfg.is_official_indextts
    assert not cfg.indextts_supports_emotion
    assert cfg.indextts_ref_audio == "/tmp/ref15.wav"


def test_config_indextts_version_field(tmp_path: Path) -> None:
    cfg_path = tmp_path / "indextts.yaml"
    cfg_path.write_text("engine: indextts\nindextts_version: '1.5'\n", encoding="utf-8")
    cfg = TtsConfig.from_yaml(cfg_path, output_dir=str(tmp_path))
    assert cfg.engine_mode == EngineMode.INDEXTTS2
    assert cfg.indextts_resolved_version == "1.5"


def test_resolve_indextts_yaml_prefers_unified_name(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "indextts.yaml").write_text("engine: indextts2\n", encoding="utf-8")
    (out_dir / "indextts2.yaml").write_text("engine: indextts15\n", encoding="utf-8")
    cfg = TtsConfig(output_dir=str(out_dir), engine_mode=EngineMode.INDEXTTS2)
    assert cfg.resolve_indextts_yaml_path() == out_dir / "indextts.yaml"


def test_is_official_indextts_chunk_settings() -> None:
    v2 = TtsConfig(output_dir="/tmp", engine_mode=EngineMode.INDEXTTS2, indextts_chunk_chars=99)
    v15 = TtsConfig(output_dir="/tmp", engine_mode=EngineMode.INDEXTTS15, indextts_chunk_chars=88)
    qwen = TtsConfig(output_dir="/tmp", engine_mode=EngineMode.MLX, qwen_chunk_chars=77)
    assert v2.chunk_chars() == 99
    assert v15.chunk_chars() == 88
    assert qwen.chunk_chars() == 77


def test_config_indextts15_fast_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "indextts.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "engine: indextts15",
                "indextts_use_fast: false",
                "indextts_max_text_tokens_per_segment: 80",
                "indextts_segments_bucket_max_size: 2",
            ]
        ),
        encoding="utf-8",
    )
    cfg = TtsConfig.from_yaml(cfg_path, output_dir=str(tmp_path))
    assert cfg.indextts_use_fast is False
    assert cfg.indextts_max_text_tokens_per_segment == 80
    assert cfg.indextts_segments_bucket_max_size == 2


def test_indextts15_uses_infer_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    from light_tts.engine.indextts import OfficialIndexTTSEngine

    calls: list[str] = []

    class _FakeTts:
        def infer(self, **_kwargs: object) -> None:
            calls.append("infer")

        def infer_fast(self, **_kwargs: object) -> None:
            calls.append("infer_fast")

    config = TtsConfig(
        output_dir="/tmp/out",
        engine_mode=EngineMode.INDEXTTS15,
        indextts_use_fast=True,
        indextts_max_text_tokens_per_segment=100,
    )
    engine = OfficialIndexTTSEngine.__new__(OfficialIndexTTSEngine)
    engine._config = config
    engine._version = "1.5"
    engine._tts = _FakeTts()
    engine._infer_v15(Path("/tmp/ref.wav"), "你好世界", Path("/tmp/out.wav"))
    assert calls == ["infer_fast"]

    config.indextts_use_fast = False
    engine._infer_v15(Path("/tmp/ref.wav"), "你好世界", Path("/tmp/out.wav"))
    assert calls == ["infer_fast", "infer"]


def test_speaker_map_auto_assign() -> None:
    cues = load_cues(CUES_JSON, lang="zh")
    config = TtsConfig(output_dir="/tmp/out", auto_assign=True, speakers={"SPEAKER_00": "Vivian"})
    mapping = build_speaker_voice_map(cues, config)
    assert mapping["SPEAKER_00"] == "Vivian"
    assert mapping["SPEAKER_01"] != mapping["SPEAKER_00"]


def test_voice_for_cue_default() -> None:
    cues = load_cues(CUES_JSON, lang="zh")
    mapping = {"SPEAKER_00": "Vivian"}
    assert voice_for_cue(cues[0], mapping, "Serena") == "Vivian"
    assert voice_for_cue(cues[1], mapping, "Serena") == "Serena"


def test_config_yaml_reads_qwen_settings(tmp_path: Path) -> None:
    voices = tmp_path / "voices.yaml"
    voices.write_text(
        "\n".join(
            [
                "default_voice: Uncle_Fu",
                "temperature: 0.25",
                "qwen_chunk_chars: 80",
                "qwen_max_tokens_min: 600",
                "qwen_max_tokens_max: 1200",
                "top_p: 0.9",
                "qwen_seed: 1234",
            ]
        ),
        encoding="utf-8",
    )
    cfg = TtsConfig.from_yaml(voices, output_dir=str(tmp_path))
    assert cfg.default_voice == "Uncle_Fu"
    assert cfg.temperature == 0.25
    assert cfg.qwen_chunk_chars == 80
    assert cfg.qwen_max_tokens_min == 600
    assert cfg.qwen_max_tokens_max == 1200
    assert cfg.top_p == 0.9
    assert cfg.qwen_seed == 1234


def test_config_defaults_qwen_seed_unset() -> None:
    cfg = TtsConfig(output_dir="/tmp/out")
    assert cfg.qwen_seed is None


def test_max_tokens_for_duration_caps_runaway() -> None:
    from light_tts.synthesize import _max_tokens_for_duration

    assert _max_tokens_for_duration(0.0, min_tokens=512, max_tokens=2048) == 512
    assert _max_tokens_for_duration(40.0, min_tokens=512, max_tokens=2048) < 2049
    assert _max_tokens_for_duration(500.0, min_tokens=512, max_tokens=2048) == 2048


def test_is_monologue_single_speaker() -> None:
    from light_tts.synthesize import _is_monologue

    cues = [Cue("a", 0.0, 1.0, "你好", "", "zh"), Cue("b", 1.0, 2.0, "世界", "", "zh")]
    assert _is_monologue(cues)
    cues2 = [Cue("a", 0.0, 1.0, "你好", "SPEAKER_00", "zh"), Cue("b", 1.0, 2.0, "嗯", "SPEAKER_01", "zh")]
    assert not _is_monologue(cues2)


def test_trim_edge_silence_removes_gaps() -> None:
    sr = 24000
    silent = np.zeros(int(0.5 * sr), dtype=np.float32)
    speech = np.ones(int(0.3 * sr), dtype=np.float32) * 0.2
    samples = np.concatenate([silent, speech, silent])
    trimmed = trim_edge_silence(samples, sr, pad_ms=10.0)
    assert len(trimmed) < len(samples)
    assert len(trimmed) / sr == pytest.approx(0.32, abs=0.05)


def test_concat_with_crossfade_joins_without_gap() -> None:
    sr = 24000
    a = np.ones(int(0.2 * sr), dtype=np.float32) * 0.2
    b = np.ones(int(0.2 * sr), dtype=np.float32) * 0.3
    joined = concat_with_crossfade([a, b], sr, crossfade_ms=20.0)
    assert len(joined) / sr == pytest.approx(0.38, abs=0.03)


def test_fit_duration_pads_short() -> None:
    sr = 24000
    samples = np.ones(int(0.5 * sr), dtype=np.float32) * 0.1
    result = fit_duration(samples, sr, target_duration=2.0)
    assert len(result.samples) == int(2.0 * sr)
    assert result.atempo == 1.0


def test_fit_duration_skips_pad_when_disabled() -> None:
    sr = 24000
    samples = np.ones(int(0.5 * sr), dtype=np.float32) * 0.1
    result = fit_duration(samples, sr, target_duration=2.0, pad_to_target=False)
    assert len(result.samples) == int(0.5 * sr)


def test_place_turn_by_cues_aligns_each_display_cue() -> None:
    from light_tts.synthesize import _place_turn_by_cues

    cues = [
        Cue(cue_id="a", start=0.6, end=4.9, text="一", speaker="__default__", lang="zh"),
        Cue(cue_id="b", start=11.4, end=13.9, text="二", speaker="__default__", lang="zh"),
    ]
    turn = SpeakerTurn(
        turn_id="turn_0000",
        speaker="__default__",
        start=0.6,
        slot_end=13.9,
        text="一。二。",
        lang="zh",
        cue_ids=("a", "b"),
    )
    sr = 22050
    samples = np.ones(sr * 2, dtype=np.float32)
    config = TtsConfig(output_dir=".", subtitle_aligned=True, speech_offset=0.05)
    placed, _ = _place_turn_by_cues(turn, {c.cue_id: c for c in cues}, samples, sr, config, atempo_max=2.2)
    assert len(placed) == 2
    assert placed[0].start == pytest.approx(0.65, abs=0.01)
    assert placed[1].start == pytest.approx(11.45, abs=0.01)
    assert placed[0].start + len(placed[0].samples) / sr <= placed[1].start + 0.001


def test_compute_subtitle_aligned_start_waits_for_cue_time() -> None:
    start = compute_subtitle_aligned_start(22.0, 11.1, speaker_gap_s=0.08)
    assert start == pytest.approx(22.0, abs=0.01)


def test_compute_subtitle_aligned_start_pushes_after_overrun() -> None:
    start = compute_subtitle_aligned_start(10.0, 12.0, speaker_gap_s=0.08)
    assert start == pytest.approx(12.08, abs=0.01)


def test_compute_turn_placed_start_compresses_long_source_pause() -> None:
    start = compute_turn_placed_start(
        10.0,
        7.0,
        speaker_gap_s=0.08,
        max_inter_speaker_pause_s=0.75,
    )
    assert start == pytest.approx(7.75, abs=0.01)


def test_compute_turn_placed_start_keeps_short_gap() -> None:
    start = compute_turn_placed_start(
        7.5,
        7.0,
        speaker_gap_s=0.08,
        max_inter_speaker_pause_s=0.75,
    )
    assert start == pytest.approx(7.5, abs=0.01)


def test_fit_duration_strict_cap_false_keeps_tail() -> None:
    sr = 24000
    samples = np.ones(int(2.0 * sr), dtype=np.float32) * 0.1
    result = fit_duration(
        samples,
        sr,
        target_duration=1.0,
        max_duration=1.0,
        atempo_max=1.28,
        strict_cap=False,
        pad_to_target=False,
    )
    assert not result.trimmed
    assert len(result.samples) / sr > 1.5


def test_fit_duration_speeds_up_long() -> None:
    sr = 24000
    samples = np.ones(int(3.0 * sr), dtype=np.float32) * 0.1
    result = fit_duration(samples, sr, target_duration=2.0, max_duration=2.0, atempo_max=1.28, allow_trim=True)
    assert result.atempo > 1.0
    assert len(result.samples) / sr <= 2.05


def test_fit_duration_same_speaker_may_overflow() -> None:
    sr = 24000
    samples = np.ones(int(1.85 * sr), dtype=np.float32) * 0.1
    result = fit_duration(
        samples,
        sr,
        target_duration=1.0,
        max_duration=1.5,
        atempo_max=1.28,
    )
    assert not result.trimmed
    assert len(result.samples) / sr <= 1.51
    assert result.overflow_s > 0.0


def test_fit_duration_cross_speaker_trims_to_cap() -> None:
    sr = 24000
    samples = np.ones(int(2.0 * sr), dtype=np.float32) * 0.1
    result = fit_duration(
        samples,
        sr,
        target_duration=1.0,
        max_duration=1.0,
        atempo_max=1.42,
    )
    assert result.trimmed
    assert len(result.samples) / sr <= 1.01


def test_fit_budget_cross_speaker_no_overflow() -> None:
    target, max_dur, allow_overflow, atempo = fit_budget(
        0.0,
        1.0,
        speech_offset=0.05,
        next_start=1.15,
        next_speaker="SPEAKER_01",
        cue_speaker="SPEAKER_00",
        speaker_gap_s=0.08,
        allow_trim=False,
    )
    assert target == 1.0
    assert allow_overflow is False
    assert atempo == 1.42
    assert max_dur <= 1.05


def test_assemble_no_overlap_gaps() -> None:
    sr = 24000
    segments = [
        PlacedSegment(start=1.0, samples=np.ones(int(0.5 * sr), dtype=np.float32), sample_rate=sr),
        PlacedSegment(start=4.0, samples=np.ones(int(0.5 * sr), dtype=np.float32), sample_rate=sr),
    ]
    timeline = assemble_timeline(segments, total_duration=6.0, sample_rate=sr, crossfade_ms=50)
    assert len(timeline) == int(6.0 * sr)
    assert timeline[int(0.5 * sr)] == 0.0
    assert timeline[int(1.1 * sr)] > 0.0


def test_preview_timeline_duration_uses_generated_audio_end() -> None:
    sr = 24000
    placed = [
        PlacedSegment(start=10.0, samples=np.ones(int(2.0 * sr), dtype=np.float32), sample_rate=sr),
        PlacedSegment(start=20.0, samples=np.ones(int(3.0 * sr), dtype=np.float32), sample_rate=sr),
    ]
    assert _preview_timeline_duration(placed, fallback_duration=180.0) == pytest.approx(24.0)


def test_run_dub_indextts15_mock_engine(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    ref = out_dir / "tts" / "ref.wav"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"RIFF")
    (out_dir / "translations" / "raw.json").parent.mkdir(parents=True)
    (out_dir / "translations" / "raw.json").write_text(CUES_JSON.read_text(encoding="utf-8"), encoding="utf-8")

    config = TtsConfig(
        output_dir=str(out_dir),
        lang="zh",
        engine_mode=EngineMode.MOCK,
        mix_mode=MixMode.REPLACE,
        preview=True,
        preview_duration_s=10.0,
        indextts_chunk_chars=40,
    )
    config.engine_mode = EngineMode.INDEXTTS15

    from light_tts import synthesize as synth_mod
    from light_tts.engine import create_engine

    engine = create_engine(TtsConfig(output_dir=str(out_dir), engine_mode=EngineMode.MOCK))
    cues = load_cues(out_dir / "translations" / "raw.json", lang="zh")
    turns = merge_speaker_turns(cues, max_turn_duration_s=None, max_turn_chars=40, min_turn_chars=10)
    speaker_map, placed, sr, _placed_turns = synth_mod.synthesize_turns(
        turns,
        cues,
        config,
        segments_dir=out_dir / "tts" / "preview" / "segments",
        engine=engine,
    )
    assert speaker_map
    assert placed
    assert sr == 24000


def test_run_dub_indextts2_mock_engine(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    ref = out_dir / "tts" / "ref.wav"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"RIFF")
    (out_dir / "translations" / "raw.json").parent.mkdir(parents=True)
    (out_dir / "translations" / "raw.json").write_text(CUES_JSON.read_text(encoding="utf-8"), encoding="utf-8")

    config = TtsConfig(
        output_dir=str(out_dir),
        lang="zh",
        engine_mode=EngineMode.MOCK,
        mix_mode=MixMode.REPLACE,
        preview=True,
        preview_duration_s=10.0,
        indextts_chunk_chars=40,
    )
    config.engine_mode = EngineMode.INDEXTTS2

    from light_tts import synthesize as synth_mod
    from light_tts.engine import create_engine

    engine = create_engine(TtsConfig(output_dir=str(out_dir), engine_mode=EngineMode.MOCK))
    cues = load_cues(out_dir / "translations" / "raw.json", lang="zh")
    turns = merge_speaker_turns(cues, max_turn_duration_s=None, max_turn_chars=40, min_turn_chars=10)
    speaker_map, placed, sr, _placed_turns = synth_mod.synthesize_turns(
        turns,
        cues,
        config,
        segments_dir=out_dir / "tts" / "preview" / "segments",
        engine=engine,
    )
    assert speaker_map
    assert placed
    assert sr == 24000


def test_indextts2_engine_sample_rate() -> None:
    from light_tts.engine.indextts import IndexTTS2Engine, OfficialIndexTTSEngine

    assert IndexTTS2Engine.sample_rate == INDEXTTS2_SAMPLE_RATE == 22050
    assert OfficialIndexTTSEngine is IndexTTS2Engine
    assert variant_spec("1.5").sample_rate == INDEXTTS15_SAMPLE_RATE == 24000
    assert variant_spec("2.0").sample_rate == INDEXTTS2_SAMPLE_RATE == 22050


def test_resume_skips_cached_segment_with_wrong_sample_rate(tmp_path: Path) -> None:
    from light_tts import synthesize as synth_mod
    from light_tts.engine import create_engine

    config = TtsConfig(output_dir=str(tmp_path), engine_mode=EngineMode.MOCK, resume=True)
    engine = create_engine(config)
    turn = SpeakerTurn(
        turn_id="t001",
        speaker="SPEAKER_00",
        start=0.0,
        slot_end=2.0,
        text="测试",
        lang="zh",
        cue_ids=("c1",),
    )
    out_path = tmp_path / "t001.wav"
    write_wav(out_path, np.zeros(22050, dtype=np.float32), 22050)

    samples, sr, stat = synth_mod._synthesize_turn(
        turn,
        tts_engine=engine,
        voice="Ryan",
        language="Chinese",
        instruct=None,
        config=config,
        monologue=False,
        out_path=out_path,
        natural_target_s=1.0,
        max_tokens=100,
    )
    assert sr == 24000
    assert stat["status"] == "ok"
    assert len(samples) > 0


def test_resume_regenerates_when_cached_sr_mismatch_indextts15(tmp_path: Path) -> None:
    from light_tts import synthesize as synth_mod
    from light_tts.engine import create_engine

    config = TtsConfig(output_dir=str(tmp_path), engine_mode=EngineMode.INDEXTTS15, resume=True)
    engine = create_engine(TtsConfig(output_dir=str(tmp_path), engine_mode=EngineMode.MOCK))
    turn = SpeakerTurn(
        turn_id="t001",
        speaker="SPEAKER_00",
        start=0.0,
        slot_end=2.0,
        text="测试",
        lang="zh",
        cue_ids=("c1",),
    )
    out_path = tmp_path / "t001.wav"
    write_wav(out_path, np.zeros(22050, dtype=np.float32), 22050)

    samples, sr, stat = synth_mod._synthesize_turn(
        turn,
        tts_engine=engine,
        voice="SPEAKER_00",
        language="Chinese",
        instruct=None,
        config=config,
        monologue=False,
        out_path=out_path,
        natural_target_s=1.0,
        max_tokens=100,
    )
    assert sr == 24000
    assert stat["status"] == "ok"
    assert len(samples) > 0


def _write_raw_json(out_dir: Path) -> Path:
    raw = out_dir / "translations" / "raw.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(CUES_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    return raw


def test_run_dub_mock_engine(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    raw = _write_raw_json(out_dir)

    config = TtsConfig(
        output_dir=str(out_dir),
        lang="zh",
        engine_mode=EngineMode.MOCK,
        mix_mode=MixMode.REPLACE,
        max_cues=3,
    )

    from light_tts import synthesize as synth_mod
    from light_tts.engine import create_engine

    engine = create_engine(config)
    speaker_map, placed, sr = synth_mod.synthesize_cues(
        load_cues(raw, lang="zh", max_cues=3),
        config,
        segments_dir=out_dir / "tts" / "segments",
        engine=engine,
    )
    assert "SPEAKER_00" in speaker_map
    assert len(placed) == 3
    assert sr == 24000

    dub_path = run_dub(config, skip_mix=True)
    assert dub_path.name == "dub.wav"
    assert (out_dir / "tts" / "voice_map.json").is_file()
    assert (out_dir / "tts" / "turns.json").is_file()
    run_meta = json.loads((out_dir / "tts" / "tts_run.json").read_text())
    assert run_meta["mode"] == "speaker_turns"
    voice_map = json.loads((out_dir / "tts" / "voice_map.json").read_text())
    assert voice_map["speakers"]["SPEAKER_00"]


def test_run_dub_preview_writes_preview_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    _write_raw_json(out_dir)

    config = TtsConfig(
        output_dir=str(out_dir),
        lang="zh",
        engine_mode=EngineMode.MOCK,
        mix_mode=MixMode.REPLACE,
        preview=True,
        preview_duration_s=5.0,
        qwen_chunk_chars=40,
    )
    dub_path = run_dub(config, skip_mix=True)
    assert dub_path == out_dir / "tts" / "preview" / "dub.wav"
    assert (out_dir / "tts" / "preview" / "chunks.json").is_file()
    run_meta = json.loads((out_dir / "tts" / "preview" / "tts_run.json").read_text())
    assert run_meta["config"]["preview"] is True


def test_find_video_prefers_video_webm(tmp_path: Path) -> None:
    (tmp_path / "other.mp4").write_bytes(b"x")
    webm = tmp_path / "video.webm"
    webm.write_bytes(b"x")
    assert find_video(tmp_path, None) == webm


def test_run_mix_only_requires_dub_wav(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "video.webm").write_bytes(b"x")
    config = TtsConfig(output_dir=str(out_dir), mix_only=True, mix_mode=MixMode.DUCK)
    with pytest.raises(FileNotFoundError, match="dub.wav"):
        run_mix_only(config)


def test_run_mix_only_calls_mix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    tts_dir = out_dir / "tts"
    tts_dir.mkdir(parents=True)
    write_wav(tts_dir / "dub.wav", np.zeros(1000, dtype=np.float32), 22050)
    (out_dir / "video.webm").write_bytes(b"x")
    calls: list[tuple] = []

    def fake_mix(video_path, dub_wav, output_path, *, mode, duck_db) -> None:
        calls.append((video_path, dub_wav, output_path, mode, duck_db))

    monkeypatch.setattr("light_tts.dub.mix_dub", fake_mix)
    config = TtsConfig(output_dir=str(out_dir), mix_only=True, mix_mode=MixMode.DUCK)
    out = run_mix_only(config)
    assert out == out_dir / "video_dub.mp4"
    assert len(calls) == 1
    assert calls[0][0].name == "video.webm"
    assert calls[0][1].name == "dub.wav"


def test_merge_dub_timeline_trims_overlap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from light_tts.episode_merge import merge_dub_timeline

    monkeypatch.setattr(
        "light_tts.episode_merge.compute_segment_offsets",
        lambda *args, **kwargs: [0.0, 15.0],
    )

    episode = tmp_path / "episode"
    seg1 = episode / ".seg1"
    seg2 = episode / ".seg2"
    for seg in (seg1, seg2):
        (seg / "tts").mkdir(parents=True)
        (seg / "video.webm").write_bytes(b"x")
    sr = 22050
    write_wav(seg1 / "tts" / "dub.wav", np.ones(sr * 30, dtype=np.float32), sr)
    write_wav(seg2 / "tts" / "dub.wav", np.full(sr * 30, 2.0, dtype=np.float32), sr)
    (episode / "split_points.json").write_text(
        json.dumps({"split_points": [0.0, 20.0, 40.0], "overlap": 5.0}),
        encoding="utf-8",
    )
    out_path, out_sr = merge_dub_timeline(episode)
    assert out_sr == sr
    merged, _ = read_wav(out_path)
    assert np.allclose(merged[: sr * 20], 1.0)
    assert np.allclose(merged[sr * 20 : sr * 40], 2.0)


def test_retime_turn_cues_partitions_interval() -> None:
    from light_tts.subtitle_retime import retime_turn_cues

    cues = [
        Cue(cue_id="a", start=0.0, end=3.0, text="一", speaker="__default__", lang="zh"),
        Cue(cue_id="b", start=3.0, end=6.0, text="二二", speaker="__default__", lang="zh"),
        Cue(cue_id="c", start=6.0, end=10.0, text="三三三", speaker="__default__", lang="zh"),
    ]
    turn = SpeakerTurn(
        turn_id="turn_0000",
        speaker="__default__",
        start=0.0,
        slot_end=10.0,
        text="一二三",
        lang="zh",
        cue_ids=("a", "b", "c"),
    )
    retimed = retime_turn_cues(turn, {c.cue_id: c for c in cues}, audio_start=5.0, audio_duration=10.0)
    assert len(retimed) == 3
    assert retimed[0].start == pytest.approx(5.0)
    assert retimed[-1].end == pytest.approx(15.0)
    for left, right in zip(retimed, retimed[1:], strict=False):
        assert left.end == pytest.approx(right.start)
        assert left.end <= right.start + 0.001


def test_turn_retime_placement_no_atempo(tmp_path: Path) -> None:
    from light_tts.synthesize import reassemble_turns_from_segments

    sr = 22050
    cues = [Cue("a", 0.0, 2.0, "测试", "__default__", "zh")]
    turn = SpeakerTurn(
        turn_id="turn_0000",
        speaker="__default__",
        start=0.0,
        slot_end=2.0,
        text="测试",
        lang="zh",
        cue_ids=("a",),
    )
    raw = np.ones(sr * 3, dtype=np.float32)
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    write_wav(segments_dir / "turn_0000.wav", raw, sr)
    config = TtsConfig(output_dir=str(tmp_path), align_mode=AlignMode.TURN_RETIME, speech_offset=0.05)
    placed, _, _ = reassemble_turns_from_segments([turn], cues, config, segments_dir=segments_dir)
    assert len(placed) == 1
    assert len(placed[0].samples) == len(raw)


def test_turn_retime_no_overlap_between_turns(tmp_path: Path) -> None:
    from light_tts.synthesize import reassemble_turns_from_segments

    sr = 22050
    cues = [
        Cue("a", 0.0, 2.0, "一", "__default__", "zh"),
        Cue("b", 5.0, 7.0, "二", "__default__", "zh"),
    ]
    turns = [
        SpeakerTurn("turn_0000", "__default__", 0.0, 2.0, "一", "zh", ("a",)),
        SpeakerTurn("turn_0001", "__default__", 5.0, 7.0, "二", "zh", ("b",)),
    ]
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    write_wav(segments_dir / "turn_0000.wav", np.ones(sr * 4, dtype=np.float32), sr)
    write_wav(segments_dir / "turn_0001.wav", np.ones(sr, dtype=np.float32), sr)
    config = TtsConfig(
        output_dir=str(tmp_path),
        align_mode=AlignMode.TURN_RETIME,
        speech_offset=0.05,
        speaker_gap_s=0.08,
    )
    placed, _, _ = reassemble_turns_from_segments(turns, cues, config, segments_dir=segments_dir)
    assert len(placed) == 2
    end0 = placed[0].start + len(placed[0].samples) / sr
    assert placed[1].start >= end0 + config.speaker_gap_s - 0.001
