"""副字幕注解 — LLM 生成阶段式内容解说。

翻译完成后，将译文按批（20 条/批）发送给 LLM。LLM 先理解对话阶段，
对有实质内容的关键概念生成自然段落的解说文本（≤100 字）。
跨批去重：LLM 上下文感知（注入已注解术语列表）+ 后处理兜底。
"""

from __future__ import annotations

import json
from pathlib import Path

from light_models import Segment, SubtitleCue, covered_source_text

from .. import logger
from ..config import SubtitleConfig
from ..llm.client import OpenAIClient
from ..llm.prompts import render_prompt
from ..usage.tracker import merge_token_usage, save_step_usage

BATCH_SIZE = 20


def generate_annotations(
    translated_cues: list[SubtitleCue],
    source_segments: list[Segment],
    config: SubtitleConfig,
    output_dir: str | Path | None = None,
) -> tuple[list[SubtitleCue], dict | None]:
    """Annotate translated cues with LLM-generated explanatory notes.

    Returns the same cue list with ``annotation`` fields populated
    where the LLM identified terms worth explaining.

    Processed in serial batches so previously annotated terms can be
    passed as context to later batches, preventing duplicates.
    A post-hoc dedup step catches any remaining duplicates.
    """
    if not config.llm_api_key or not translated_cues:
        return translated_cues, None

    source_map: dict[str, str] = {s.unit_id: s.source_text for s in source_segments}

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    annotated_terms: list[str] = []  # Cross-batch dedup context
    total_usage: dict[str, int] = {}

    for batch_start in range(0, len(translated_cues), BATCH_SIZE):
        batch = translated_cues[batch_start : batch_start + BATCH_SIZE]

        batch_data = []
        for cue in batch:
            batch_data.append(
                {
                    "unit_id": cue.unit_id,
                    "source": covered_source_text(cue, source_map),
                    "translation": cue.text.replace("\n", " "),
                }
            )

        batch_json_str = json.dumps(batch_data, ensure_ascii=False)
        system_prompt = render_prompt(
            "annotate.j2",
            batch_json=batch_json_str,
            already_annotated=annotated_terms if annotated_terms else None,
        )

        try:
            response, usage = client.chat(
                [{"role": "user", "content": system_prompt}],
                temperature=0.1,
            )
        except Exception:
            logger.warning(f"    ⚠ Annotation batch failed, skipping {len(batch)} cues")
            continue

        merge_token_usage(total_usage, usage)

        data = _extract_json(response)
        if data is None:
            continue

        cue_map: dict[str, SubtitleCue] = {c.unit_id: c for c in batch}

        for item in data:
            uid = item.get("unit_id", "")
            annotation = item.get("annotation", "").strip()
            if not uid or not annotation:
                continue

            cue = cue_map.get(uid)
            if cue is None:
                continue

            cue.annotation = annotation
            term = _extract_term(annotation)
            if term and term not in annotated_terms:
                annotated_terms.append(term)

    # Post-hoc dedup — catches any terms the LLM missed.
    _dedup_annotations(translated_cues)

    logger.info(f"    Annotation tokens: {total_usage.get('total_tokens', 0)}")
    if output_dir is not None and total_usage:
        save_step_usage(Path(output_dir) / "annotations" / "usage.json", total_usage)

    return translated_cues, total_usage or None


def _extract_json(response: str) -> list | None:
    """Extract and parse a JSON array from an LLM response."""
    raw = response.strip()
    raw = raw.replace("\\N", "\\\\N")  # escape \N for JSON compatibility

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences.
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove opening ```json or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove trailing ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _extract_term(annotation: str) -> str:
    """Extract the normalized term from an annotation string.

    "RL训练：强化学习的方法" → "rl训练"
    """
    if "：" in annotation:
        return annotation.split("：")[0].strip().lower()
    if ":" in annotation:
        return annotation.split(":")[0].strip().lower()
    return annotation.strip().lower()


def _dedup_annotations(cues: list[SubtitleCue]) -> None:
    """Remove duplicate annotations across cues (post-hoc safety net).

    Two annotations are duplicates if they share the same normalized term
    (the part before ː/:). Only the first occurrence is kept.
    """
    seen: set[str] = set()
    removed = 0
    for cue in cues:
        if not cue.annotation:
            continue
        key = _extract_term(cue.annotation)
        if key in seen:
            cue.annotation = ""
            removed += 1
        else:
            seen.add(key)
    if removed:
        logger.info(f"    Deduplicated: removed {removed} duplicate annotations")
