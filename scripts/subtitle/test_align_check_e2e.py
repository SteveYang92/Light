#!/usr/bin/env python3
"""End-to-end smoke test for translation alignment check (real LLM).

Builds sample payloads, calls ``check_batch_alignment``, and prints the
request/response. Requires ``DEEPSEEK_API_KEY`` (or ``--llm-api-key``).

Usage::

    uv run python scripts/subtitle/test_align_check_e2e.py
    uv run python scripts/subtitle/test_align_check_e2e.py --case misaligned
    uv run python scripts/subtitle/test_align_check_e2e.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from light_cli.config import SubtitleConfig
from light_llm.client import OpenAIClient, format_token_usage
from light_models import Segment
from light_subtitle.translate import align_check


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    sources: list[str]
    translations: list[str]
    batch_idx: int
    expect_actionable_misaligned: bool | None  # None = don't assert


def _seg(unit_id: str, text: str) -> Segment:
    return Segment(
        unit_id=unit_id,
        start=0.0,
        end=1.0,
        speaker="",
        source_text=text,
        words=[],
    )


SCENARIOS: dict[str, Scenario] = {
    "aligned": Scenario(
        name="aligned",
        description="All translations match their sources.",
        sources=[
            "We need to fix the schedule.",
            "The pricing model changed last week.",
            "So we adjusted the roadmap.",
            "Marketing agreed to the timeline.",
            "Let's sync again tomorrow.",
        ],
        translations=[
            "我们需要调整日程。",
            "定价模式上周发生了变化。",
            "所以我们调整了路线图。",
            "市场部门同意了时间表。",
            "我们明天再同步一次。",
        ],
        batch_idx=0,
        expect_actionable_misaligned=False,
    ),
    "misaligned": Scenario(
        name="misaligned",
        description="Middle unit translation is clearly about pricing, not roadmap.",
        sources=[
            "We need to fix the schedule.",
            "The pricing model changed last week.",
            "So we adjusted the roadmap.",
            "Marketing agreed to the timeline.",
            "Let's sync again tomorrow.",
        ],
        translations=[
            "我们需要调整日程。",
            "定价模式上周发生了变化。",
            "这份报价比去年贵了很多。",  # wrong topic for source[2]
            "市场部门同意了时间表。",
            "我们明天再同步一次。",
        ],
        batch_idx=0,
        expect_actionable_misaligned=True,
    ),
    "off_by_one": Scenario(
        name="off_by_one",
        description="Unit 2 translation actually matches unit 3 source (off-by-one shift).",
        sources=[
            "We need to fix the schedule.",
            "The pricing model changed last week.",
            "So we adjusted the roadmap.",
            "Marketing agreed to the timeline.",
            "Let's sync again tomorrow.",
        ],
        translations=[
            "我们需要调整日程。",
            "定价模式上周发生了变化。",
            "市场部门同意了时间表。",  # belongs to source[3]
            "我们明天再同步一次。",  # belongs to source[4]
            "我们明天再同步一次。",
        ],
        batch_idx=0,
        expect_actionable_misaligned=True,
    ),
    "cross_batch": Scenario(
        name="cross_batch",
        description="Batch starts at index 2; before-context is source-only (no translation).",
        sources=[
            "Opening remark.",
            "Transition line.",
            "So we adjusted the roadmap.",
            "Marketing agreed to the timeline.",
            "Closing remark.",
        ],
        translations=[
            "开场白。",
            "过渡句。",
            "这份报价比去年贵了很多。",  # misaligned
            "市场部门同意了时间表。",
            "结束语。",
        ],
        batch_idx=2,
        expect_actionable_misaligned=True,
    ),
    "semantic_narrowing": Scenario(
        name="semantic_narrowing",
        description="stuff → 影像 is acceptable subtitle narrowing (Camas Prairie regression).",
        sources=[
            "Previous line about the region.",
            "stuff from the Camas Prairie.",
            "Next line continues.",
        ],
        translations=[
            "关于那个地区的上一句。",
            "卡马斯草原的影像。",
            "下一句继续。",
        ],
        batch_idx=0,
        expect_actionable_misaligned=False,
    ),
}


def _build_config(args: argparse.Namespace) -> SubtitleConfig:
    api_key = args.llm_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    return SubtitleConfig(
        input_path="dummy.mp4",
        target_lang=args.target_lang,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=api_key,
    )


def _run_scenario(scenario: Scenario, config: SubtitleConfig, *, dry_run: bool) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Case: {scenario.name}")
    print(scenario.description)
    print("=" * 60)

    all_segments = [_seg(f"u{i:02d}", s) for i, s in enumerate(scenario.sources)]
    batch_len = len(scenario.sources) - scenario.batch_idx
    batch = all_segments[scenario.batch_idx : scenario.batch_idx + batch_len]
    parsed_texts = {i: scenario.translations[scenario.batch_idx + i] for i in range(batch_len)}
    sample_indices = align_check._alignment_sample_indices(batch_len)
    tx_config = config.translate_config()
    payload = align_check._build_align_payload(
        batch,
        parsed_texts,
        sample_indices,
        all_segments,
        scenario.batch_idx,
        tx_config,
    )
    system_prompt = align_check._render_align_check_prompt(tx_config)

    print("\n--- System prompt ---")
    print(system_prompt[:800] + ("..." if len(system_prompt) > 800 else ""))
    print("\n--- User payload ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSample indices (batch-local): {sample_indices}")

    if dry_run:
        print("\n[dry-run] Skipping LLM call.")
        return True

    if not config.llm_api_key:
        print("\nERROR: No API key. Set DEEPSEEK_API_KEY or pass --llm-api-key.", file=sys.stderr)
        return False

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )
    aligned, failures, usage = align_check.check_batch_alignment(
        client,
        batch,
        parsed_texts,
        all_segments,
        scenario.batch_idx,
        tx_config,
    )

    print("\n--- Result ---")
    print(f"aligned: {aligned}")
    print(f"actionable failures: {failures}")
    print(f"usage: {format_token_usage(usage)}")

    if scenario.expect_actionable_misaligned is None:
        ok = True
    elif scenario.expect_actionable_misaligned:
        ok = not aligned and bool(failures)
        if not ok:
            print(
                f"\nUNEXPECTED: expected actionable misalignment, got aligned={aligned}, failures={failures}",
                file=sys.stderr,
            )
    else:
        ok = aligned and not failures
        if not ok:
            print(
                f"\nUNEXPECTED: expected all aligned, got aligned={aligned}, failures={failures}",
                file=sys.stderr,
            )

    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {scenario.name}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E smoke test for align check LLM")
    parser.add_argument(
        "--case",
        choices=[*SCENARIOS.keys(), "all"],
        default="all",
        help="Which scenario to run (default: all)",
    )
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-api-key", default="", help="Defaults to DEEPSEEK_API_KEY env var")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt/payload only, no LLM call")
    args = parser.parse_args()

    config = _build_config(args)
    cases = list(SCENARIOS.values()) if args.case == "all" else [SCENARIOS[args.case]]

    print(f"Model: {config.llm_model} @ {config.llm_base_url}")
    print(f"Target lang: {config.target_lang}")

    results = [_run_scenario(scenario, config, dry_run=args.dry_run) for scenario in cases]
    passed = sum(results)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"Summary: {passed}/{total} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
