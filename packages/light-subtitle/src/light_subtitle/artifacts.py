"""Pipeline artifact paths and (de)serialization.

Single source of truth for artifact filenames, directory layout, and the
JSON schemas of words / cues / plan units.  Resume (``state_hydrate``),
the regression harness, and the web backend all depend on these names
and byte-level JSON layouts — treat any change here as a format change.

Conventions:
- Leaf path helpers take the pipeline *output_dir*.
- The ``segment_words`` family takes a *plan_dir* because its callers
  (translate / join / hydrate internals) already hold that directory.
- Filename constants are provided for internals that already hold a
  ``tx_dir`` / ``plan_dir`` and prefer plain joins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from light_models import Segment, SubtitleCue, Word

# ── Filenames ───────────────────────────────────────────────────────────────

TRANSCRIPT_JSON = "transcript.json"  # write: export step; read: resume, QC, backend
CUES_JSON = "cues.json"  # write: export; read: TTS, pack tooling (not a player sidecar)
ANNOTATIONS_JSON = "annotations.json"  # annotations/ — write: annotate step; read: export resume
SEGMENT_JSON = "segment.json"  # segment/ — write: export step; read: resume
PLAN_JSON = "plan.json"  # plan/ — write: planner; read: resume, join
PLAN_JOINED_JSON = "plan.joined.json"  # plan/ — write: join; read: resume
SEGMENT_WORDS_JSON = "segment_words.json"  # plan/ — write: plan step; read: pace, resume, join
SEGMENT_WORDS_JOINED_JSON = "segment_words.joined.json"  # plan/ — write: join; read: resume
RAW_JSON = "raw.json"  # translations/ — write: translate save; read: resume, QC
SOURCE_JSON = "source.json"  # translations/ — write: translate save; read: review
PARTIAL_JSON = "partial.json"  # translations/ — write/read: live translation checkpoint
FINGERPRINT_JSON = "fingerprint.json"  # translations/ — write/read: cache staleness check
USAGE_JSON = "usage.json"  # translations/, plan/, and several step dirs
QUALITY_JSON = "quality.json"  # translations/ — write: evaluate step

# Player-facing sidecars share the downloaded video stem so players (IINA)
# auto-load them next to ``video.webm`` / ``video.mp4``.
VIDEO_STEM = "video"
PLAYER_SIDECAR_SUFFIXES = (
    "zh.srt",
    "zh.vtt",
    "en.srt",
    "en.vtt",
    "bilingual.ass",
    "bilingual.vtt",
    "annotations.ass",
    "annotations.vtt",
)


# ── Directory helpers ───────────────────────────────────────────────────────


def translations_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "translations"


def plan_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "plan"


def segment_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "segment"


def annotations_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "annotations"


# ── Leaf path helpers (take the pipeline output dir) ────────────────────────


def transcript_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / TRANSCRIPT_JSON


def cues_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / CUES_JSON


def annotations_path(output_dir: str | Path) -> Path:
    """Persisted unit_id → annotation text map (resume-from export needs this)."""
    return annotations_dir(output_dir) / ANNOTATIONS_JSON


def sidecar_name(suffix: str) -> str:
    """Player-facing filename: ``zh.srt`` → ``video.zh.srt``."""
    return f"{VIDEO_STEM}.{suffix.lstrip('.')}"


def sidecar_path(output_dir: str | Path, suffix: str) -> Path:
    """Path for a player-facing subtitle sidecar under *output_dir*."""
    return Path(output_dir) / sidecar_name(suffix)


def find_sidecar(directory: str | Path, suffix: str, slug: str | None = None) -> Path | None:
    """Resolve a sidecar preferring ``video.*``, then bare, then ``{slug}.*``.

    Used by merge/resume readers so older bare / slug-prefixed runs still work.
    """
    directory = Path(directory)
    suffix = suffix.lstrip(".")
    candidates = [
        directory / sidecar_name(suffix),
        directory / suffix,
    ]
    if slug:
        candidates.append(directory / f"{slug}.{suffix}")
    for path in candidates:
        if path.is_file():
            return path
    return None


def migrate_legacy_sidecars(output_dir: str | Path, slug: str | None = None) -> list[str]:
    """Move bare / ``{slug}.*`` player sidecars to ``video.*`` for IINA.

    After a re-export that rewrites language tracks but skips annotations
    (empty annotate state / ``--annotate`` off), leftover ``annotations.ass``
    would otherwise stay unprefixed.  Returns the suffixes that were moved
    or whose bare duplicates were removed.
    """
    output_dir = Path(output_dir)
    changed: list[str] = []
    for suffix in PLAYER_SIDECAR_SUFFIXES:
        target = sidecar_path(output_dir, suffix)
        legacies = [output_dir / suffix]
        if slug:
            legacies.append(output_dir / f"{slug}.{suffix}")
        for legacy in legacies:
            if not legacy.is_file():
                continue
            if legacy.resolve() == target.resolve():
                continue
            if not target.exists():
                legacy.rename(target)
                changed.append(suffix)
            else:
                legacy.unlink()
                changed.append(suffix)
    return changed


def segment_json_path(output_dir: str | Path) -> Path:
    return segment_dir(output_dir) / SEGMENT_JSON


def raw_cues_path(output_dir: str | Path) -> Path:
    return translations_dir(output_dir) / RAW_JSON


def source_cues_path(output_dir: str | Path) -> Path:
    return translations_dir(output_dir) / SOURCE_JSON


def partial_cues_path(output_dir: str | Path) -> Path:
    return translations_dir(output_dir) / PARTIAL_JSON


def fingerprint_path(output_dir: str | Path) -> Path:
    return translations_dir(output_dir) / FINGERPRINT_JSON


def translation_usage_path(output_dir: str | Path) -> Path:
    return translations_dir(output_dir) / USAGE_JSON


def quality_path(output_dir: str | Path) -> Path:
    return translations_dir(output_dir) / QUALITY_JSON


def plan_json_path(output_dir: str | Path) -> Path:
    return plan_dir(output_dir) / PLAN_JSON


def plan_joined_path(output_dir: str | Path) -> Path:
    return plan_dir(output_dir) / PLAN_JOINED_JSON


def plan_usage_path(output_dir: str | Path) -> Path:
    return plan_dir(output_dir) / USAGE_JSON


# ── Segment-words paths (take the plan dir) ─────────────────────────────────


def segment_words_path(plan_dir: str | Path) -> Path:
    return Path(plan_dir) / SEGMENT_WORDS_JSON


def segment_words_joined_path(plan_dir: str | Path) -> Path:
    return Path(plan_dir) / SEGMENT_WORDS_JOINED_JSON


def resolve_segment_words_path(plan_dir: str | Path, *, prefer_joined: bool = True) -> Path:
    """Word-timing file, preferring the joined graph when a join pass ran.

    The join pass itself must read the ORIGINAL graph (``prefer_joined=False``)
    because its 1:1 cues still reference the original unit ids.
    """
    plan_dir = Path(plan_dir)
    if prefer_joined:
        joined = segment_words_joined_path(plan_dir)
        if joined.exists():
            return joined
    return segment_words_path(plan_dir)


# ── Generic JSON I/O ────────────────────────────────────────────────────────


def write_json(path: str | Path, data: object) -> None:
    """Write *data* as pretty UTF-8 JSON (the pipeline's standard layout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_annotations(output_dir: str | Path, annotations: dict[str, str]) -> Path:
    """Persist the unit_id → annotation map for resume / re-export."""
    path = annotations_path(output_dir)
    write_json(path, dict(annotations))
    return path


def load_annotations(output_dir: str | Path) -> dict[str, str]:
    """Load persisted annotations, or ``{}`` when missing / malformed."""
    path = annotations_path(output_dir)
    if not path.is_file():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


# ── Word (de)serialization ──────────────────────────────────────────────────
#
# Single 5-field schema shared by every writer (asr checkpoints, transcript,
# segment words, debug dumps).  Key order matters for byte-identical output.


def word_to_dict(word: Word) -> dict:
    return {
        "text": word.text,
        "start": word.start,
        "end": word.end,
        "confidence": word.confidence,
        "speaker": word.speaker,
    }


def word_from_dict(raw: dict) -> Word:
    """Build a Word, tolerating missing optional keys and debug-only extras
    (e.g. the ``changed`` flag in transcript_correct dumps)."""
    return Word(
        text=raw["text"],
        start=raw["start"],
        end=raw["end"],
        confidence=raw.get("confidence", 1.0),
        speaker=raw.get("speaker"),
    )


# ── Cue (de)serialization ───────────────────────────────────────────────────


def cue_to_dict(cue: SubtitleCue) -> dict:
    """Checkpoint schema (``translations/partial.json``)."""
    return {
        "cue_id": cue.cue_id,
        "unit_id": cue.unit_id,
        "start": cue.start,
        "end": cue.end,
        "text": cue.text,
        "lang": cue.lang,
    }


def cue_to_raw_dict(cue: SubtitleCue) -> dict:
    """``raw.json`` schema: checkpoint fields plus ``merged_from`` when set."""
    data = cue_to_dict(cue)
    if cue.merged_from:
        data["merged_from"] = cue.merged_from
    return data


def cue_from_dict(raw: dict, *, default_lang: str | None = None) -> SubtitleCue:
    """Tolerant reader shared by the raw.json / partial.json loaders."""
    return SubtitleCue(
        cue_id=raw["cue_id"],
        unit_id=raw["unit_id"],
        start=raw["start"],
        end=raw["end"],
        text=raw["text"],
        lang=raw.get("lang", default_lang),
        speaker=raw.get("speaker", ""),
        merged_from=raw.get("merged_from", []),
    )


def write_raw_cues(path: str | Path, cues: list[SubtitleCue]) -> None:
    """Write ``translations/raw.json`` (or ``source.json``) cue list."""
    write_json(path, [cue_to_raw_dict(c) for c in cues])


def read_raw_cues(path: str | Path, *, default_lang: str | None = None) -> list[SubtitleCue]:
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)
    return [cue_from_dict(c, default_lang=default_lang) for c in raw_data]


# ── Segment-words map (plan/segment_words[.joined].json) ────────────────────


def segment_words_to_map(segments: list[Segment]) -> dict[str, list[dict]]:
    """unit_id → word dicts; units without words are omitted."""
    return {seg.unit_id: [word_to_dict(w) for w in seg.words] for seg in segments if seg.words}


def write_segment_words(path: str | Path, segments: list[Segment]) -> None:
    write_json(path, segment_words_to_map(segments))


def read_segment_words_map(path: str | Path) -> dict[str, list[dict]]:
    return read_json(path)


def load_segment_words_map(plan_dir: str | Path, *, prefer_joined: bool = True) -> dict[str, list[dict]] | None:
    """Resolved word-timing map (joined graph preferred), None when absent."""
    path = resolve_segment_words_path(plan_dir, prefer_joined=prefer_joined)
    if not path.exists():
        return None
    return read_segment_words_map(path)


def words_from_unit_chain(unit_ids: list[str], seg_words_map: dict[str, list[dict]]) -> list[Word]:
    """Concatenate word timing for a cue's head unit + ``merged_from`` chain."""
    words: list[Word] = []
    for uid in unit_ids:
        word_dicts = seg_words_map.get(uid)
        if word_dicts:
            words.extend(word_from_dict(w) for w in word_dicts)
    return words


# ── transcript.json / segment.json readers (resume side) ────────────────────


def read_transcript_words(path: str | Path) -> list[Word]:
    return [word_from_dict(w) for w in read_json(path).get("words", [])]


def read_segment_units(path: str | Path, words: list[Word]) -> list[Segment]:
    """Rebuild segments from ``segment/segment.json``, re-slicing word timing."""
    units = []
    for unit in read_json(path).get("units", []):
        units.append(
            Segment(
                unit_id=unit["unit_id"],
                start=unit["start"],
                end=unit["end"],
                source_text=unit.get("source_text", ""),
                speaker=unit.get("speaker"),
                words=_slice_words_for_unit(words, unit),
            )
        )
    return units


def _slice_words_for_unit(words: list[Word], unit: dict) -> list[Word]:
    start = unit.get("start", 0.0)
    end = unit.get("end", 0.0)
    matched = [w for w in words if w.start >= start - 0.05 and w.end <= end + 0.05]
    if matched:
        return matched
    return list(words)


# ── plan[.joined].json ──────────────────────────────────────────────────────


def write_plan_meta(path: str | Path, meta: list[dict], *, version: int) -> None:
    """Write ``{"version": v, "units": meta}`` (plan.json / plan.joined.json)."""
    write_json(path, {"version": version, "units": meta})


def plan_unit_from_dict(item: dict) -> Segment:
    """Rebuild a wordless unit from plan meta (word timing re-attached separately)."""
    return Segment(
        unit_id=item["unit_id"],
        start=item.get("start", 0.0),
        end=item.get("end", 0.0),
        speaker=item.get("speaker", ""),
        source_text=item.get("text", ""),
        words=[],
    )


def read_plan_units(path: str | Path) -> list[Segment]:
    return [plan_unit_from_dict(item) for item in read_json(path).get("units", [])]


# ── fingerprint.json ────────────────────────────────────────────────────────


def write_fingerprint(path: str | Path, fingerprint: str) -> None:
    """Compact one-line JSON (matches the original writer byte for byte)."""
    Path(path).write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")


def read_fingerprint(path: str | Path) -> str | None:
    return read_json(path).get("fingerprint")
