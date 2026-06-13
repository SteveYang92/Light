from light_models import seconds_to_ass, seconds_to_srt, seconds_to_vtt, srt_to_seconds


def test_seconds_to_srt_zero():
    assert seconds_to_srt(0) == "00:00:00,000"


def test_seconds_to_srt_simple():
    assert seconds_to_srt(1.5) == "00:00:01,500"
    assert seconds_to_srt(61.0) == "00:01:01,000"
    assert seconds_to_srt(3661.0) == "01:01:01,000"


def test_seconds_to_srt_millis():
    assert seconds_to_srt(0.001) == "00:00:00,001"
    assert seconds_to_srt(0.999) == "00:00:00,999"


def test_srt_to_seconds():
    assert srt_to_seconds("00:00:01,500") == 1.5
    assert srt_to_seconds("00:01:01,000") == 61.0
    assert srt_to_seconds("01:01:01,000") == 3661.0


def test_roundtrip():
    for s in [0, 0.5, 1.5, 61.0, 3661.5, 123.456]:
        assert abs(srt_to_seconds(seconds_to_srt(s)) - s) < 0.001


def test_seconds_to_vtt():
    assert seconds_to_vtt(1.5) == "00:00:01.500"
    assert seconds_to_vtt(61.0) == "00:01:01.000"


def test_seconds_to_ass():
    assert seconds_to_ass(1.5) == "0:00:01.50"
    assert seconds_to_ass(61.0) == "0:01:01.00"
