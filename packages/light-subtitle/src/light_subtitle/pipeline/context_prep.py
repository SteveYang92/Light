"""Translation context preparation — glossary + content summary extraction.

Runs after segmentation and before translation. Concatenates all segment
texts into a full transcript and sends it to the LLM for terminology and
summary extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from light_models import Segment

from .. import logger
from ..config import SubtitleConfig
from ..llm.client import OpenAIClient, format_token_usage
from ..llm.prompts import render_prompt


@dataclass
class ContextPrepResult:
    """Glossary and summary extracted for translation."""

    glossary: dict[str, str] = field(default_factory=dict)
    summary: dict | None = None


def build_transcript_text(segments: list[Segment]) -> str:
    """Join all segment source texts in playback order with speaker prefixes."""
    lines: list[str] = []
    for s in segments:
        speaker = s.speaker or "UNKNOWN"
        lines.append(f"[{speaker}] {s.source_text}")
    return "\n".join(lines)


def prepare_context(
    segments: list[Segment],
    config: SubtitleConfig,
    output_dir: str | Path,
) -> ContextPrepResult:
    """Extract glossary and content summary from *segments* via LLM."""
    if not segments or not config.llm_api_key or not config.context_prep_enabled:
        return ContextPrepResult()

    output_dir = Path(output_dir)
    context_dir = output_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    glossary_path = context_dir / "glossary.json"
    summary_path = context_dir / "summary.json"

    if glossary_path.exists() and summary_path.exists():
        result = _load_cached_context(glossary_path, summary_path)
        logger.info(f"  Context prep (cached): {len(result.glossary)} glossary terms")
        return result

    transcript_text = build_transcript_text(segments)
    (context_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )
    prompt = render_prompt(
        "context_prep.j2",
        target_lang=config.target_lang or "zh",
        transcript_text=transcript_text,
    )

    try:
        response, usage = client.chat([{"role": "user", "content": prompt}], temperature=0.1)
    except Exception as e:
        logger.warning(f"  Context prep failed: {e}")
        return ContextPrepResult()

    result = _parse_context_response(response)
    _save_context(context_dir, result)
    logger.info(
        f"  Context prep: {len(segments)} segments, {len(transcript_text)} chars, "
        f"{len(result.glossary)} glossary terms, summary={'yes' if result.summary else 'no'}, "
        f"{format_token_usage(usage)}"
    )
    return result


def load_cached_context(output_dir: str | Path) -> ContextPrepResult:
    """Load glossary and summary from a previous run."""
    context_dir = Path(output_dir) / "context"
    return _load_cached_context(context_dir / "glossary.json", context_dir / "summary.json")


def merge_glossary(auto_glossary: dict[str, str], user_glossary: dict[str, str]) -> dict[str, str]:
    """Merge auto-extracted glossary with user YAML; user entries override."""
    return {**auto_glossary, **user_glossary}


def _parse_context_response(response: str) -> ContextPrepResult:
    """Parse LLM JSON into ContextPrepResult."""
    response = response.strip()
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return ContextPrepResult()
    else:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return ContextPrepResult()

    if not isinstance(data, dict):
        return ContextPrepResult()

    summary = data.get("summary")
    if not isinstance(summary, dict):
        summary = None

    raw_glossary = data.get("glossary", {})
    glossary: dict[str, str] = {}
    if isinstance(raw_glossary, dict):
        glossary = {str(k): str(v) for k, v in raw_glossary.items() if k and v}

    return ContextPrepResult(glossary=glossary, summary=summary)


def _load_cached_context(glossary_path: Path, summary_path: Path) -> ContextPrepResult:
    """Load cached glossary and summary JSON files."""
    glossary: dict[str, str] = {}
    summary: dict | None = None

    if glossary_path.exists():
        data = json.loads(glossary_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            glossary = {str(k): str(v) for k, v in data.items()}

    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            summary = data

    return ContextPrepResult(glossary=glossary, summary=summary)


def _save_context(context_dir: Path, result: ContextPrepResult) -> None:
    """Persist glossary and summary artifacts."""
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "glossary.json").write_text(
        json.dumps(result.glossary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if result.summary is not None:
        (context_dir / "summary.json").write_text(
            json.dumps(result.summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
