"""``light polish`` — LLM transcript polish: correct → restore_punct.

Thin wrapper over :mod:`light_asr_polish`.  Reads ``transcript.json``
(``light-transcript.v1``), runs transcript correction then punctuation
restoration, and writes back the same format; per-step debug artifacts
(``transcript_correct/`` / ``punct_restore/``) land in the output dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from light_asr_polish import correct, restore_punct
from light_core import logger
from light_subtitle import export as export_module

from ..artifacts import read_transcript_words
from .common import LlmApiKey, LlmBaseUrl, LlmModel, build_client, resolve_api_key


def polish(
    input_path: Annotated[str, typer.Option("-i", "--input", help="transcript.json (light-transcript.v1)")],
    output_dir: Annotated[str, typer.Option("-o", "--output", help="Output directory")] = "./output",
    llm_base_url: LlmBaseUrl = "https://api.deepseek.com",
    llm_model: LlmModel = "deepseek-v4-flash",
    llm_api_key: LlmApiKey = "",
):
    """Polish transcript.json words via LLM (correction + punctuation)."""
    if not resolve_api_key(llm_api_key):
        raise typer.BadParameter("LLM API key required (--llm-api-key or DEEPSEEK_API_KEY).")

    source = json.loads(Path(input_path).read_text(encoding="utf-8")).get("source", "unknown")
    words = read_transcript_words(input_path)
    if not words:
        raise typer.BadParameter(f"No words found in {input_path}.")

    client = build_client(llm_base_url, llm_model, llm_api_key)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    transcript = out / "transcript.json"

    words, _ = correct(words, client, out)
    export_module.export_transcript(words, [], str(transcript), source=source)
    logger.info(f"  Correct done → {transcript}")

    words, _ = restore_punct(words, client, out)
    export_module.export_transcript(words, [], str(transcript), source=source)
    logger.info(f"  Punct restore done → {transcript}")

    typer.echo(f"transcript.json: {transcript} ({len(words)} words)")
