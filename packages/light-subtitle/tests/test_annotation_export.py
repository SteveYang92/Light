from light_models import SubtitleCue
from light_subtitle.pipeline.export import (
    export_annotation_vtt,
    export_vtt,
    format_annotation_display,
    strip_annotation_marker,
)


def test_strip_annotation_marker_removes_leading_markers() -> None:
    assert strip_annotation_marker("RL训练：强化学习") == "RL训练：强化学习"
    assert strip_annotation_marker("※ RL训练：强化学习") == "RL训练：强化学习"
    assert strip_annotation_marker("※ ※ /played：MMORPG命令") == "/played：MMORPG命令"
    assert strip_annotation_marker("  ※  ※  术语：解释  ") == "术语：解释"


def test_format_annotation_display_keeps_single_marker() -> None:
    assert format_annotation_display("RL训练：强化学习") == "※ RL训练：强化学习"
    assert format_annotation_display("※ RL训练：强化学习") == "※ RL训练：强化学习"
    assert format_annotation_display("※ ※ /played：MMORPG命令") == "※ /played：MMORPG命令"
    assert format_annotation_display("") == ""


def test_export_vtt_converts_ass_line_breaks(tmp_path) -> None:
    output = tmp_path / "zh.vtt"
    cues = [SubtitleCue(cue_id="c1", unit_id="u1", start=0, end=1, text="第一行\\N第二行", lang="zh")]

    export_vtt(cues, str(output))

    assert "第一行\n第二行" in output.read_text(encoding="utf-8")
    assert "\\N" not in output.read_text(encoding="utf-8")


def test_export_annotation_vtt_converts_ass_line_breaks(tmp_path) -> None:
    output = tmp_path / "annotations.vtt"
    cues = [SubtitleCue(cue_id="c1", unit_id="u1", start=0, end=1, text="正文", lang="zh")]

    export_annotation_vtt(cues, {"u1": "术语\\N解释"}, str(output))

    content = output.read_text(encoding="utf-8")
    assert "※ 术语\n解释" in content
    assert "\\N" not in content
