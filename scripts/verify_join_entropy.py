"""Targeted verification of the join pass on the real entropy artifacts.

Reuses the existing pipeline outputs (no pipeline re-run): loads
plan/plan.json + plan/segment_words.json + translations/raw.json, runs
ONLY the join pass (one small LLM call set), and reports:

- before/after for the 9 review-issue spots (matched by ZH text)
- three-caps stats (max chars, >5s cues, over-CPS cues) before vs after
- dangling-flavour scan counts before vs after

Usage: uv run python scripts/verify_join_entropy.py
"""

from __future__ import annotations

import json
from pathlib import Path

from light_models import Segment, SubtitleCue, Word
from light_subtitle.config import SubtitleConfig
from light_subtitle.pipeline.translate.join import join_cues, save_joined_units

OUT = Path("output/Reinventing_Entropy_Compression_is_Intelligence_Part_1")
SCRATCH = Path("/tmp/join_verify")

REVIEW_SPOTS = [
    ("#3", "所以你自然会"),
    ("#6", "但指令"),
    ("#7", "看看你是如何不可避免地"),
    ("#4", "这里 我们可以把"),
    ("#1", "你已经看到了"),
    ("#9", "小型GPT"),
    ("#5", "更长的字符串后面"),
    ("#2", "然后 另外"),
    ("#8", "一方面是因为"),
]

ZH_DANGLE_END = (
    "的", "地", "得", "把", "将", "与", "所", "和", "或", "但", "而", "是", "会", "能", "要",
    "让", "对", "为", "从", "向", "用", "被", "想", "像", "看", "到", "成", "于", "之",
)


def load() -> tuple[list[SubtitleCue], list[Segment]]:
    plan = json.load(open(OUT / "plan/plan.json"))["units"]
    seg_words = json.load(open(OUT / "plan/segment_words.json"))
    units = [
        Segment(
            unit_id=u["unit_id"],
            start=u["start"],
            end=u["end"],
            speaker=u["speaker"],
            source_text=u["text"],
            words=[Word(**w) for w in seg_words.get(u["unit_id"], [])],
        )
        for u in plan
    ]
    by_unit = {u.unit_id: u for u in units}
    raw = json.load(open(OUT / "translations/raw.json"))
    cues = []
    for c in raw:
        ws = []
        for uid in [c["unit_id"], *c.get("merged_from", [])]:
            u = by_unit.get(uid)
            if u:
                ws.extend(u.words)
        cues.append(
            SubtitleCue(
                cue_id=c["cue_id"],
                unit_id=c["unit_id"],
                start=c["start"],
                end=c["end"],
                text=c["text"],
                lang="zh",
                merged_from=c.get("merged_from", []),
                words=ws,
            )
        )
    return cues, units


def stats(cues: list[SubtitleCue]) -> dict:
    def nchars(c: SubtitleCue) -> int:
        return len(c.text.replace("\n", "").replace(" ", ""))

    max_chars = max(nchars(c) for c in cues)
    over_dur = sum(1 for c in cues if c.end - c.start > 5.0)
    over_cps = sum(1 for c in cues if c.end > c.start and nchars(c) / (c.end - c.start) > 9)
    flash = sum(1 for c in cues if c.end - c.start < 1.0 and not c.text.rstrip().endswith(("。", "！", "？", "…")))
    dangle = sum(1 for c in cues if c.text.rstrip("。！？…，、；：").endswith(ZH_DANGLE_END))
    return {
        "cues": len(cues),
        "max_chars": max_chars,
        ">5s": over_dur,
        "over_cps": over_cps,
        "flash": flash,
        "dangle": dangle,
    }


def neighbourhood(cues: list[SubtitleCue], i: int) -> list[str]:
    out = []
    for j in range(max(0, i - 1), min(len(cues), i + 3)):
        c = cues[j]
        out.append(f"    [{j}] ({c.end - c.start:.1f}s) {c.text}")
    return out


def main() -> None:
    cues, units = load()
    before = stats(cues)
    print(f"loaded {len(cues)} cues, {len(units)} units")

    config = SubtitleConfig(input_path=str(OUT / "x.webm"), output_dir=str(OUT), target_lang="zh", max_duration=5.0)
    result = join_cues(cues, units, config)
    after = stats(result.cues)

    print(f"\nops applied: {result.ops_applied}, usage: {result.usage}")
    print(f"{'metric':>10} {'before':>8} {'after':>8}")
    for k in before:
        print(f"{k:>10} {before[k]:>8} {after[k]:>8}")

    print("\n=== review spots (after) ===")
    for tag, frag in REVIEW_SPOTS:
        idx = next((i for i, c in enumerate(result.cues) if frag in c.text), None)
        print(f"{tag} ('{frag}'):")
        if idx is None:
            print("    (merged away — no cue contains the fragment)")
            idx2 = next((i for i, c in enumerate(cues) if frag in c.text), None)
            if idx2 is not None:
                print("    before:")
                print("\n".join(neighbourhood(cues, idx2)))
        else:
            print("\n".join(neighbourhood(result.cues, idx)))

    SCRATCH.mkdir(parents=True, exist_ok=True)
    save_joined_units(result.units, SCRATCH / "plan")
    with open(SCRATCH / "raw.joined.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "cue_id": c.cue_id,
                    "unit_id": c.unit_id,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                    "lang": c.lang,
                    "merged_from": c.merged_from,
                }
                for c in result.cues
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\njoined artifacts written to {SCRATCH}")


if __name__ == "__main__":
    main()
