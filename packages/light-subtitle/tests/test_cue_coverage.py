from light_models import Segment, SubtitleCue, Word, covered_source_text
from light_subtitle.pipeline.translate.evaluate import evaluate_translations


def test_evaluate_uses_covered_source_for_merged_cue(monkeypatch):
    """Merged cues should pair with concatenated source text, not head only."""
    segments = [
        Segment(
            unit_id="u0",
            start=1.0,
            end=2.0,
            source_text="First",
            speaker="",
            words=[Word(text="First", start=1.0, end=2.0, confidence=0.9)],
        ),
        Segment(
            unit_id="u1",
            start=2.5,
            end=4.0,
            source_text="second",
            speaker="",
            words=[Word(text="second", start=2.5, end=4.0, confidence=0.9)],
        ),
    ]
    cues = [
        SubtitleCue(
            cue_id="t1",
            unit_id="u0",
            start=1.0,
            end=4.0,
            text="第一第二",
            lang="zh",
            merged_from=["u1"],
        ),
    ]

    captured: list[str] = []

    def fake_evaluate_batch(pairs, config, batch_num):
        captured.extend(src for _, src in pairs)
        return []

    monkeypatch.setattr(
        "light_subtitle.pipeline.translate.evaluate._evaluate_batch",
        fake_evaluate_batch,
    )

    from light_subtitle.config import SubtitleConfig

    config = SubtitleConfig(
        input_path="x",
        evaluate_enabled=True,
        llm_api_key="test-key",
    )
    evaluate_translations(cues, segments, config)
    assert captured == ["First second"]


def test_covered_source_text_matches_segment_order():
    cue = SubtitleCue(
        cue_id="t1",
        unit_id="u0",
        start=1.0,
        end=4.0,
        text="x",
        lang="zh",
        merged_from=["u1", "u2"],
    )
    source_map = {
        "u0": "Alpha",
        "u1": "beta",
        "u2": "gamma",
    }
    assert covered_source_text(cue, source_map) == "Alpha beta gamma"
