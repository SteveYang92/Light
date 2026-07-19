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
from ..llm.client import client_from_config
from ..llm.prompts import render_prompt
from ..usage.tracker import merge_token_usage, save_step_usage

BATCH_SIZE = 20

_SUMMARY_KEYS = ("title", "domain", "overview", "key_topics")


def generate_annotations(
    translated_cues: list[SubtitleCue],
    source_segments: list[Segment],
    config: SubtitleConfig,
    output_dir: str | Path | None = None,
    *,
    glossary: dict | None = None,
    content_summary: dict | None = None,
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

    client = client_from_config(config)

    system_prompt = _render_annotate_system_prompt(
        config, output_dir, glossary=glossary, content_summary=content_summary
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
        user_prompt = _render_annotate_user_prompt(
            batch_json_str,
            annotated_terms if annotated_terms else None,
        )

        try:
            response, usage = client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"    ⚠ Annotation batch failed ({type(e).__name__}: {e}), skipping {len(batch)} cues")
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


# ── Context loading and filtering ───────────────────────────────────────────


def _load_domain_context(output_dir: Path) -> dict | None:
    """Load cached domain context from the correct step artifact."""
    cache_path = output_dir / "transcript_correct" / "domain_context.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _filter_summary(raw: dict | None) -> dict | None:
    """Keep title, domain, overview, key_topics; drop speakers."""
    if not raw:
        return None
    filtered = {key: raw[key] for key in _SUMMARY_KEYS if raw.get(key)}
    return filtered or None


def _filter_terms(domain_context: dict | None) -> list[dict[str, str]] | None:
    """Extract term + context from domain_context terminology."""
    if not domain_context:
        return None
    terminology = domain_context.get("terminology")
    if not isinstance(terminology, list):
        return None

    terms: list[dict[str, str]] = []
    for entry in terminology:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term")
        if not term:
            continue
        item: dict[str, str] = {"term": str(term)}
        context = entry.get("context")
        if context:
            item["context"] = str(context)
        terms.append(item)

    return terms or None


def _filter_glossary(raw: dict | None) -> dict[str, str] | None:
    """Keep translated glossary entries; skip keep-as-is (src == tgt)."""
    if not raw:
        return None
    filtered = {str(src): str(tgt) for src, tgt in raw.items() if str(src) != str(tgt)}
    return filtered or None


def _filter_annotate_context(
    content_summary: dict | None,
    domain_context: dict | None,
    glossary: dict | None,
) -> tuple[dict | None, list[dict[str, str]] | None, dict[str, str] | None]:
    """Filter upstream context to annotate-specific fields."""
    summary = _filter_summary(content_summary)
    terms = _filter_terms(domain_context)
    filtered_glossary = _filter_glossary(glossary)
    return summary, terms, filtered_glossary


def _render_annotate_system_prompt(
    config: SubtitleConfig,
    output_dir: str | Path | None,
    *,
    glossary: dict | None = None,
    content_summary: dict | None = None,
) -> str:
    """Render system prompt once per video run (cache-friendly)."""
    domain_context: dict | None = None
    if output_dir is not None:
        domain_context = _load_domain_context(Path(output_dir))

    raw_glossary = (config.glossary if glossary is None else glossary) or None
    summary, terms, filtered_glossary = _filter_annotate_context(
        config.content_summary if content_summary is None else content_summary,
        domain_context,
        raw_glossary,
    )

    raw_glossary_count = len(raw_glossary) if raw_glossary else 0
    filtered_glossary_count = len(filtered_glossary) if filtered_glossary else 0
    logger.info(
        "    Annotate context: "
        f"summary={'yes' if summary else 'no'}, "
        f"terms={len(terms) if terms else 0}, "
        f"glossary={filtered_glossary_count}/{raw_glossary_count}"
    )

    return render_prompt(
        "annotate_system.j2",
        summary=summary,
        terms=terms,
        glossary=filtered_glossary,
    )


def _render_annotate_user_prompt(
    batch_json: str,
    already_annotated: list[str] | None,
) -> str:
    """Render per-batch user prompt with cue data and dedup list."""
    return render_prompt(
        "annotate_user.j2",
        batch_json=batch_json,
        already_annotated=already_annotated,
    )


# ── Response parsing and dedup ──────────────────────────────────────────────


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
