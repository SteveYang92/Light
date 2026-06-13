import tempfile
from pathlib import Path

from light_qc.input import parse_srt, parse_vtt

SRT_CONTENT = """1
00:00:01,000 --> 00:00:04,000
欢迎来到节目。

2
00:00:04,200 --> 00:00:08,500
今天讨论 AI。

3
00:00:09,000 --> 00:00:12,000
这是第一行
这是第二行
"""

VTT_CONTENT = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
欢迎来到节目。

2
00:00:04.200 --> 00:00:08.500
今天讨论 AI。

3
00:00:09.000 --> 00:00:12.000
这是第一行
这是第二行
"""


def test_parse_srt():
    cues = parse_srt(_write_temp(SRT_CONTENT, ".srt"))
    assert len(cues) == 3
    assert cues[0].start == 1.0
    assert cues[0].end == 4.0
    assert "欢迎来到节目" in cues[0].text
    assert cues[2].text == "这是第一行\n这是第二行"


def test_parse_vtt():
    cues = parse_vtt(_write_temp(VTT_CONTENT, ".vtt"))
    assert len(cues) == 3
    assert cues[0].start == 1.0
    assert cues[0].end == 4.0


def test_detect_lang():
    cues = parse_srt(_write_temp(SRT_CONTENT, ".srt"))
    zh_cues = [c for c in cues if c.lang == "zh"]
    assert len(zh_cues) == 3


def _write_temp(content: str, suffix: str) -> str:
    path = Path(tempfile.gettempdir()) / f"test_light{suffix}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return str(path)
