import json
import re
from pathlib import Path

from light_models import SubtitleCue, srt_to_seconds


def parse_srt(path: str) -> list[SubtitleCue]:
    cues = []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
        r"((?:.+\n?)+)",
        re.MULTILINE,
    )

    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start_tc = match.group(2)
        end_tc = match.group(3)
        text = match.group(4).strip()
        start = srt_to_seconds(start_tc)
        end = srt_to_seconds(end_tc)
        cues.append(
            SubtitleCue(
                cue_id=f"srt_{idx:04d}",
                unit_id="",
                start=start,
                end=end,
                text=text,
                lang=_detect_lang(text),
            )
        )
    return cues


def parse_vtt(path: str) -> list[SubtitleCue]:
    cues = []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"WEBVTT.*?\n\n", "", content, flags=re.DOTALL)

    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n"
        r"((?:.+\n?)+?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}|\n*$)",
        re.MULTILINE,
    )

    for i, match in enumerate(pattern.finditer(content), 1):
        start_tc = match.group(1).replace(".", ",")
        end_tc = match.group(2).replace(".", ",")
        text = match.group(3).strip()
        start = srt_to_seconds(start_tc)
        end = srt_to_seconds(end_tc)
        cues.append(
            SubtitleCue(
                cue_id=f"vtt_{i:04d}",
                unit_id="",
                start=start,
                end=end,
                text=text,
                lang=_detect_lang(text),
            )
        )
    return cues


def parse_ass(path: str) -> list[SubtitleCue]:
    cues = []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"Dialogue:\s*\d+,\s*"
        r"(\d+:\d{2}:\d{2}\.\d{2}),\s*"
        r"(\d+:\d{2}:\d{2}\.\d{2}),"
        r"(?:[^,]*,){7}"
        r"(.*)",
        re.MULTILINE,
    )

    for i, match in enumerate(pattern.finditer(content), 1):
        start_tc = match.group(1).replace(".", ",") + "0"
        end_tc = match.group(2).replace(".", ",") + "0"
        text = match.group(3).strip()
        text = text.replace("\\N", "\n").replace("\\n", "\n")
        text = re.sub(r"\{[^}]*\}", "", text)
        start = srt_to_seconds(start_tc)
        end = srt_to_seconds(end_tc)
        cues.append(
            SubtitleCue(
                cue_id=f"ass_{i:04d}",
                unit_id="",
                start=start,
                end=end,
                text=text,
                lang=_detect_lang(text),
            )
        )
    return cues


def parse_json(path: str) -> list[SubtitleCue]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cues = []
    for item in data.get("cues", data.get("cues", [])):
        if isinstance(item, dict):
            cues.append(
                SubtitleCue(
                    cue_id=item.get("cue_id", item.get("id", "")),
                    unit_id=item.get("unit_id", ""),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                    text=item.get("text", item.get("zh_text", item.get("en_text", ""))),
                    lang=item.get("lang", item.get("language", _detect_lang(item.get("text", "")))),
                    speaker=item.get("speaker", ""),
                    qc=item.get("qc", {}),
                )
            )
    return cues


def load(paths: list[str], default_lang: str = "auto") -> dict[str, list[SubtitleCue]]:
    result: dict[str, list[SubtitleCue]] = {}
    for p in paths:
        path = Path(p)
        suffix = path.suffix.lower()
        if suffix == ".srt":
            cues = parse_srt(str(path))
        elif suffix == ".vtt":
            cues = parse_vtt(str(path))
        elif suffix == ".ass" or suffix == ".ssa":
            cues = parse_ass(str(path))
        elif suffix == ".json":
            cues = parse_json(str(path))
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        lang = _detect_lang_from_cues(cues)
        if lang == "unknown" and default_lang != "auto":
            for cue in cues:
                cue.lang = default_lang
            lang = default_lang
        result.setdefault(lang, []).extend(cues)

    if not result:
        result["unknown"] = []

    return result


def parse_single(path: str) -> list[SubtitleCue]:
    loaded = load([path])
    for cues in loaded.values():
        return cues
    return []


def _detect_lang(text: str) -> str:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def _detect_lang_from_cues(cues: list[SubtitleCue]) -> str:
    zh_count = sum(1 for c in cues if c.lang == "zh")
    en_count = sum(1 for c in cues if c.lang == "en")
    return "zh" if zh_count > en_count else "en"
