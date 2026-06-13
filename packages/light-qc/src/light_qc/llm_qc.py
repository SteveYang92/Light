import json
import re

from light_models import QCIssue, SubtitleCue

from .config import QCConfig
from .llm.client import OpenAIClient
from .llm.prompts import render_prompt

CHUNK_SIZE = 50


def run_llm_qc(cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
    if not config.llm_api_key:
        return []

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    # Build flat list with language tracking
    all_cues = []
    for lang, cue_list in cues.items():
        for i, cue in enumerate(cue_list):
            all_cues.append(
                {
                    "id": i + 1,
                    "start": cue.start,
                    "end": cue.end,
                    "lang": lang,
                    "text": cue.text,
                    "speaker": cue.speaker,
                }
            )

    # Chunk for long subtitle lists
    if len(all_cues) <= CHUNK_SIZE:
        return _check_batch(client, all_cues, config)

    all_issues = []
    for batch_idx in range(0, len(all_cues), CHUNK_SIZE):
        batch = all_cues[batch_idx : batch_idx + CHUNK_SIZE]
        issues = _check_batch(client, batch, config)
        all_issues.extend(issues)

    return all_issues


def _check_batch(client: OpenAIClient, batch_cues: list[dict], config: QCConfig) -> list[QCIssue]:
    prompt = render_prompt(
        "qc.j2",
        bilingual=config.bilingual,
        source_lang=config.source_lang,
        target_lang=config.target_lang or "",
        cues=batch_cues,
        shot_changes=config.shot_changes,
        glossary=config.glossary,
    )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "bilingual": config.bilingual,
                    "source_lang": config.source_lang,
                    "target_lang": config.target_lang,
                    "cues": batch_cues,
                    "shot_changes": config.shot_changes,
                    "glossary": config.glossary,
                },
                ensure_ascii=False,
            ),
        },
    ]

    response, _usage = client.chat(messages, temperature=config.llm_temperature)
    return _parse_issues(response)


def _parse_issues(response: str) -> list[QCIssue]:
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        data = json.loads(json_match.group(0))
    else:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return []

    issues = []
    for item in data.get("issues", []):
        issues.append(
            QCIssue(
                severity=item.get("severity", "suggestion"),
                category=item.get("category", "柔性策略"),
                rule=item.get("rule", ""),
                cue_id=item.get("cue_id"),
                time=item.get("time"),
                detail=item.get("detail", ""),
                fix=item.get("fix", ""),
            )
        )
    return issues
