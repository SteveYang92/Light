from light_models import Segment, SubtitleCue, Word


def test_word_creation():
    w = Word(text="hello", start=1.0, end=2.0, confidence=0.95)
    assert w.text == "hello"
    assert w.start == 1.0
    assert w.end == 2.0
    assert w.confidence == 0.95
    assert w.speaker is None


def test_word_with_speaker():
    w = Word(text="hello", start=1.0, end=2.0, confidence=0.95, speaker="S1")
    assert w.speaker == "S1"


def test_segment():
    w1 = Word(text="你好", start=1.0, end=1.5, confidence=0.9, speaker="S1")
    w2 = Word(text="世界", start=1.6, end=2.0, confidence=0.95, speaker="S1")
    seg = Segment(
        unit_id="u001",
        start=1.0,
        end=2.0,
        speaker="S1",
        source_text="你好世界",
        words=[w1, w2],
        source_cue_ids=["c1", "c2"],
    )
    assert seg.unit_id == "u001"
    assert len(seg.words) == 2
    assert seg.source_cue_ids == ["c1", "c2"]


def test_subtitle_cue():
    cue = SubtitleCue(
        cue_id="c001",
        unit_id="u001",
        start=1.0,
        end=3.0,
        text="第一行\n第二行",
        lang="zh",
        speaker="S1",
    )
    assert cue.cue_id == "c001"
    assert cue.lang == "zh"
    assert cue.qc == {}


def test_subtitle_cue_qc():
    cue = SubtitleCue(
        cue_id="c001",
        unit_id="u001",
        start=1.0,
        end=3.0,
        text="test",
        lang="zh",
        qc={"reading_speed_ok": True, "line_length_ok": True},
    )
    assert cue.qc["reading_speed_ok"] is True
