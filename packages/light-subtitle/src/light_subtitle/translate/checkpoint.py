"""Translation checkpoint — partial.json persistence and unit-graph fingerprint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from light_core import logger
from light_models import Segment, SubtitleCue

from .. import artifacts
from ..config import TranslateConfig

_PARTIAL_VERSION = 2


def segment_graph_fingerprint(segments: list[Segment]) -> str:
    """Stable hash of the translation unit graph (ids + timing)."""
    payload = [(s.unit_id, round(s.start, 3), round(s.end, 3)) for s in segments]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
    return digest[:16]


def _save_partial(
    tx_dir: Path,
    cues: list[SubtitleCue],
    segments: list[Segment],
) -> None:
    """Persist 1:1 translation checkpoint."""
    tx_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "version": _PARTIAL_VERSION,
        "segments_fingerprint": segment_graph_fingerprint(segments),
        "cues": [artifacts.cue_to_dict(c) for c in cues],
    }
    artifacts.write_json(tx_dir / artifacts.PARTIAL_JSON, data)


def _discard_partial_cache(tx_dir: Path, *, reason: str) -> None:
    path = tx_dir / artifacts.PARTIAL_JSON
    if path.exists():
        path.unlink()
        logger.info(f"  Discarded stale partial.json ({reason})")


def load_partial(
    tx_dir: Path,
    config: TranslateConfig,
    segments: list[Segment] | None = None,
) -> list[SubtitleCue]:
    """Load 1:1 partial cues.

    When *segments* is provided (translate entry), discard the checkpoint if it
    no longer matches the current translation unit graph.
    """
    path = tx_dir / artifacts.PARTIAL_JSON
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        if any(c.get("merged_from") for c in raw if isinstance(c, dict)):
            logger.warning("  Legacy partial.json contains merged cues; delete partial.json for a clean resume.")
        cues = [_cue_from_partial_dict(c, config) for c in raw]
    elif isinstance(raw, dict):
        cues = [_cue_from_partial_dict(c, config) for c in raw.get("cues", [])]
    else:
        return []

    if segments is not None and cues:
        expected = segment_graph_fingerprint(segments)
        stored = raw.get("segments_fingerprint") if isinstance(raw, dict) else None
        if stored is not None:
            if stored != expected:
                _discard_partial_cache(tx_dir, reason="segment graph changed")
                return []
        elif not _partial_matches_segments(cues, segments):
            _discard_partial_cache(tx_dir, reason="unit graph mismatch")
            return []

    return cues


def _partial_matches_segments(
    cues: list[SubtitleCue],
    segments: list[Segment],
) -> bool:
    """Heuristic for legacy partial files without a stored fingerprint."""
    segment_ids = {s.unit_id for s in segments}
    seg_by_id = {s.unit_id: s for s in segments}
    for cue in cues:
        if cue.unit_id not in segment_ids:
            return False
        seg = seg_by_id[cue.unit_id]
        if abs(cue.start - seg.start) > 0.01 or abs(cue.end - seg.end) > 0.01:
            return False
    return True


def _cue_from_partial_dict(data: dict, config: TranslateConfig) -> SubtitleCue:
    return artifacts.cue_from_dict(data, default_lang=config.target_lang)


def load_partial_cues(tx_dir: Path, config: TranslateConfig) -> list[SubtitleCue]:
    return load_partial(tx_dir, config)
