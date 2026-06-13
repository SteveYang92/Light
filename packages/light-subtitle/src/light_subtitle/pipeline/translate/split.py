"""Split overlong translation units (LLM + semantic fallback).

After compose produces semantically complete sentences, some may exceed
``max_duration``.  This module splits them into shorter sub-segments
using strategies in order:
  1. LLM batch split at natural break points.
  2. Deterministic semantic fallback at whole-word boundaries.
  3. Last-resort balanced word-boundary split; never split inside a word.

Each sub-segment is built from whole ``Word`` objects, so source text and
word-level timestamps stay aligned for downstream translation and pacing.

Usage::
    from .split import split_overlong_units
    segments = split_overlong_units(segments, config)
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass

from light_models import Segment, Word

from ... import logger
from ...config import SubtitleConfig
from ...language import SENTENCE_END
from ...llm.client import OpenAIClient
from ...llm.prompts import render_prompt

# ── Public entry point ─────────────────────────────────────────────────────

BATCH_SIZE = 8
_SOFT_MAX_DURATION_MULTIPLIER = 1.15
_MIN_CHUNK_WORDS = 3
_MIN_CHUNK_DURATION = 1.0
_CLAUSE_PUNCT = {",", ";", ":", "，", "；", "：", "—", "–"}
_STRONG_BOUNDARY_WORDS = {
    "because",
    "but",
    "so",
    "where",
    "when",
    "while",
    "although",
    "though",
    "if",
    "unless",
    "until",
    "since",
    "whereas",
}
_MEDIUM_BOUNDARY_WORDS = {"and", "or", "yet", "then", "which", "who", "that"}
_WEAK_BOUNDARY_WORDS = {"like", "now", "well", "okay", "ok"}
_PREPOSITIONS = {
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "from",
    "by",
    "about",
    "into",
    "through",
    "over",
    "under",
    "after",
    "before",
    "between",
    "during",
    "without",
    "within",
    "upon",
    "across",
    "along",
    "around",
    "behind",
    "beyond",
    "down",
    "off",
    "out",
    "up",
    "toward",
    "towards",
    "against",
    "among",
    "beside",
    "above",
    "below",
    "near",
    "inside",
    "outside",
}
_ARTICLES = {"a", "an", "the"}


@dataclass(frozen=True)
class _TextToken:
    """Normalized source token plus its span in the original text."""

    value: str
    start: int
    end: int


@dataclass
class _BatchAttempt:
    """Classified result from one LLM split batch attempt."""

    splits: dict[str, list[Segment]]
    unresolved: list[Segment]
    accepted_count: int
    one_part_count: int
    mismatch_count: int
    absent_count: int


def split_overlong_units(segments: list[Segment], config: SubtitleConfig) -> list[Segment]:
    """Split segments whose duration exceeds ``config.max_duration``.

    See module docstring for strategy details.
    """
    if config.max_duration <= 0:
        return segments

    # Phase 1: Collect overlong segments for batch LLM split.
    llm_candidates: list[Segment] = []
    if config.llm_api_key:
        for seg in segments:
            if seg.end - seg.start > config.max_duration:
                llm_candidates.append(seg)

    # Phase 2: Batch LLM split.
    batch_splits: dict[str, list[Segment]] = {}
    if llm_candidates:
        batch_splits = _llm_split_batch(llm_candidates, config)

    # Phase 3: Process segments in original order — substitute LLM splits
    # or fall back to semantic word-boundary splitting.
    result: list[Segment] = []
    for seg in segments:
        if seg.end - seg.start <= config.max_duration:
            result.append(seg)
            continue

        sub = batch_splits.get(seg.unit_id)
        if sub is not None:
            result.extend(_verify_or_split(sub, config))
            continue

        result.extend(_split_single(seg, config.max_duration))

    return result


def _split_single(seg: Segment, max_duration: float) -> list[Segment]:
    """Split a single segment at natural word boundaries."""
    return _split_at_word_boundaries(seg, max_duration, force=True)


# ── Split verification ────────────────────────────────────────────────────


def _verify_or_split(splits: list[Segment], config: SubtitleConfig) -> list[Segment]:
    """Verify LLM splits; recursively split overlong chunks when needed."""
    verified: list[Segment] = []
    for sub in splits:
        if sub.end - sub.start <= config.max_duration * _SOFT_MAX_DURATION_MULTIPLIER:
            verified.append(sub)
        else:
            verified.extend(_split_at_word_boundaries(sub, config.max_duration, force=False))
    return verified


# ── LLM-based split ────────────────────────────────────────────────────────


def _llm_split_batch(overlong_segments: list[Segment], config: SubtitleConfig) -> dict[str, list[Segment]]:
    """Split multiple overlong segments in a single LLM batch request.

    Returns dict mapping unit_id -> list[Segment] (sub-segments).
    Segments that fail (LLM error, JSON parse error, text mismatch, etc.)
    are excluded from the dict, and the caller falls back to local
    semantic word-boundary splitting.
    """
    if not overlong_segments:
        return {}

    client = OpenAIClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    result: dict[str, list[Segment]] = {}

    for batch_start in range(0, len(overlong_segments), BATCH_SIZE):
        batch = overlong_segments[batch_start : batch_start + BATCH_SIZE]
        label = f"{batch_start + 1}-{batch_start + len(batch)}"
        attempt = _run_llm_split_attempt(client, batch, config, label)
        result.update(attempt.splits)

        if attempt.unresolved:
            logger.info(
                f"  LLM batch partial ({label}): "
                f"accepted={attempt.accepted_count}, one_part={attempt.one_part_count}, "
                f"mismatch={attempt.mismatch_count}, absent={attempt.absent_count}; "
                f"retrying unresolved={len(attempt.unresolved)}"
            )
            retry = _run_llm_split_attempt(client, attempt.unresolved, config, label)
            result.update(retry.splits)
            retry_unresolved_count = len(retry.unresolved)
            logger.info(
                f"  LLM batch retry ({label}): "
                f"accepted={retry.accepted_count}, one_part={retry.one_part_count}, "
                f"mismatch={retry.mismatch_count}, absent={retry.absent_count}; "
                f"local fallback handles unresolved={retry_unresolved_count}"
            )

    return result


def _run_llm_split_attempt(
    client: OpenAIClient,
    batch: list[Segment],
    config: SubtitleConfig,
    label: str,
) -> _BatchAttempt:
    """Run one LLM split attempt and classify unresolved items."""
    batch_by_id = {seg.unit_id: seg for seg in batch}
    batch_json_str = json.dumps(_batch_payload(batch, config.max_duration), ensure_ascii=False)
    system_prompt = render_prompt("compose_split.j2", batch_json=batch_json_str)

    try:
        response, _ = client.chat(
            [{"role": "user", "content": system_prompt}],
            temperature=0.1,
        )
    except Exception:
        logger.warning(f"  ⚠ LLM batch failed ({label}), local fallback for {len(batch)} units")
        return _BatchAttempt({}, batch, 0, 0, 0, len(batch))

    data = _parse_batch_json(response)
    if data is None:
        logger.warning(f"  ⚠ LLM batch JSON parse failed ({label}), local fallback for {len(batch)} units")
        return _BatchAttempt({}, batch, 0, 0, 0, len(batch))

    raw_results = data.get("results", [])
    if not raw_results:
        logger.warning(f"  ⚠ LLM batch returned empty results ({label}), local fallback for {len(batch)} units")
        return _BatchAttempt({}, batch, 0, 0, 0, len(batch))

    splits: dict[str, list[Segment]] = {}
    seen_ids: set[str] = set()
    one_part_ids: set[str] = set()
    mismatch_ids: set[str] = set()

    for item in raw_results:
        uid = item.get("id")
        parts = [p.strip() for p in item.get("parts", []) if p.strip()]
        if not uid:
            continue
        seen_ids.add(uid)

        seg = batch_by_id.get(uid)
        if seg is None:
            continue

        if len(parts) < 2:
            one_part_ids.add(uid)
            continue

        sub_segments = _build_sub_segments(seg, parts)
        if sub_segments:
            splits[uid] = sub_segments
        else:
            mismatch_ids.add(uid)

    absent_ids = set(batch_by_id) - seen_ids
    unresolved_ids = (one_part_ids | mismatch_ids | absent_ids) - set(splits)
    unresolved = [seg for seg in batch if seg.unit_id in unresolved_ids]
    return _BatchAttempt(
        splits=splits,
        unresolved=unresolved,
        accepted_count=len(splits),
        one_part_count=len(one_part_ids),
        mismatch_count=len(mismatch_ids),
        absent_count=len(absent_ids),
    )


def _batch_payload(batch: list[Segment], max_duration: float) -> list[dict]:
    """Build the JSON payload sent to the LLM splitter."""
    return [
        {
            "id": seg.unit_id,
            "duration": round(seg.end - seg.start, 2),
            "target_duration": max_duration,
            "text": seg.source_text.strip(),
        }
        for seg in batch
    ]


def _parse_batch_json(response: str) -> dict | None:
    """Parse JSON while tolerating code fences or short leading commentary."""
    text = response.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _build_sub_segments(seg: Segment, parts: list[str]) -> list[Segment]:
    """Build sub-segments from LLM-split text parts.

    Primary strategy: word-count mapping.  Since the LLM is constrained to
    preserve the original text character-for-character (only inserting "|||"),
    the parts concatenated with spaces equal ``seg.source_text``.  We can
    therefore count words in each part and consume that many words from
    the parent's ``seg.words`` list, using exact word-boundary timestamps.

    If the LLM text doesn't match the original, return an empty list so the
    caller can use the deterministic word-boundary splitter.  We intentionally
    avoid proportional timing because it can desynchronize text and words.
    """
    if len([p for p in parts if p.strip()]) < 2:
        return []

    words = seg.words

    # Verify LLM text matches original (with whitespace normalization).
    rejoined = " ".join(parts)
    original = seg.source_text
    if _texts_match(rejoined, original) and words:
        sub_segments = _build_by_word_count(seg, parts, words)
        if _sub_segments_are_safe(sub_segments):
            return sub_segments
        logger.warning(f"    ⚠ LLM split unsafe boundaries for {seg.unit_id}, falling back to local split")
        return []

    logger.warning(
        f"    ⚠ LLM split text mismatch for {seg.unit_id} "
        f"({len(original)}ch orig vs {len(rejoined)}ch joined), "
        f"falling back to local word-boundary split"
    )
    return []


_TRAILING_PUNCT = set(".!?。！？…")


def _texts_match(a: str, b: str) -> bool:
    """Compare two texts with whitespace normalization.

    Tolerates sentence-end punctuation diffs — the LLM sometimes
    drops periods when inserting ``|||`` markers
    (e.g. ``"interesting."`` → ``"interesting"``,
    ``"a lot. so for"`` → ``"a lot so for"``).
    """
    a_norm = " ".join(a.split())
    b_norm = " ".join(b.split())
    if a_norm == b_norm:
        return True
    # Strip all sentence-end punctuation and retry.
    punct_chars = "".join(_TRAILING_PUNCT)
    a_stripped = " ".join(a_norm.translate(str.maketrans("", "", punct_chars)).split())
    b_stripped = " ".join(b_norm.translate(str.maketrans("", "", punct_chars)).split())
    return a_stripped == b_stripped


def _build_by_word_count(seg: Segment, parts: list[str], words: list) -> list[Segment]:
    """Build sub-segments by counting words in each LLM-split part.

    Since the parts exactly reconstruct the original text, the word count
    of each part exactly matches the number of transcript words it covers.
    This guarantees exact word-boundary-aligned timestamps.
    """
    sub_segments: list[Segment] = []
    word_offset = 0

    for i, part in enumerate(parts):
        part_word_count = len(part.split())
        chunk_words = words[word_offset : word_offset + part_word_count]

        if not chunk_words:
            # Empty part or word count overflow — skip
            word_offset += part_word_count
            continue

        sub_start = chunk_words[0].start
        sub_end = chunk_words[-1].end if i < len(parts) - 1 else seg.end

        sub = Segment(
            unit_id=f"{seg.unit_id}_{i}",
            start=sub_start,
            end=sub_end,
            speaker=seg.speaker,
            source_text=part,
            words=chunk_words,
        )
        sub_segments.append(sub)
        word_offset += part_word_count

    return sub_segments


def _sub_segments_are_safe(segments: list[Segment]) -> bool:
    """Validate LLM splits before accepting them over local fallback."""
    if len(segments) < 2:
        return False

    for index, segment in enumerate(segments):
        if not segment.words:
            return False
        if index > 0 and _chunk_starts_with_preposition(segment.words):
            return False
        if _chunk_ends_with_preposition(segment.words):
            return False
        if not _text_matches_word_tokens(segment.source_text, segment.words):
            return False

    return True


# ── Semantic word-boundary fallback ────────────────────────────────────────


def _split_at_word_boundaries(seg: Segment, max_duration: float, *, force: bool) -> list[Segment]:
    """Split *seg* without ever cutting inside a word.

    ``force=True`` is used for an original overlong segment, so even a
    slightly-over target is split once.  Recursive calls accept a small
    overage when another split would damage translation coherence.
    """
    words = [w for w in seg.words if _clean_word(w.text)]
    if len(words) < 2:
        return [seg]

    word_chunks = _split_word_chunks(words, max_duration, force=force)
    if len(word_chunks) <= 1:
        return [seg]

    chunk_texts = _texts_from_parent_chunks(seg.source_text, word_chunks)
    return [
        _segment_from_word_chunk(seg, chunk, f"{seg.unit_id}_{i}", chunk_texts[i])
        for i, chunk in enumerate(word_chunks)
    ]


def _split_word_chunks(words: list[Word], max_duration: float, *, force: bool) -> list[list[Word]]:
    """Recursively split word chunks by semantic score, then duration."""
    duration = _words_duration(words)
    soft_max = max_duration * _SOFT_MAX_DURATION_MULTIPLIER
    if len(words) < 2 or (not force and duration <= soft_max):
        return [words]

    split_index = _choose_split_index(words, max_duration)
    if split_index is None:
        return [words]

    left = words[:split_index]
    right = words[split_index:]
    if not left or not right:
        return [words]

    return _split_word_chunks(left, max_duration, force=False) + _split_word_chunks(
        right,
        max_duration,
        force=False,
    )


def _choose_split_index(words: list[Word], max_duration: float) -> int | None:
    """Pick the highest-scoring boundary between whole words."""
    total_duration = _words_duration(words)
    target_left = total_duration / 2 if total_duration <= max_duration * 2.2 else max_duration

    best_index: int | None = None
    best_score = float("-inf")
    for index in range(1, len(words)):
        if _is_forbidden_boundary(words, index):
            continue

        left = words[:index]
        right = words[index:]
        if _chunk_ends_with_preposition(left) or _chunk_starts_with_preposition(right):
            continue

        score = _boundary_semantic_score(words, index)
        score += _duration_score(left, right, max_duration, target_left)
        score += _chunk_size_score(left, right)

        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def _is_forbidden_boundary(words: list[Word], index: int) -> bool:
    """Return True when a boundary would split a tightly-bound phrase."""
    prev_word = _normalized_word(words[index - 1])
    next_word = _normalized_word(words[index])
    if not prev_word or not next_word:
        return True

    if prev_word in _PREPOSITIONS or prev_word in _ARTICLES:
        return True

    # Never start a continuation chunk with a bare preposition ("of this mix", "to get").
    if next_word in _PREPOSITIONS:
        return True

    if prev_word == "from" or next_word in {"to", "into"} and _has_open_from(words[:index]):
        return True

    return False


def _has_open_from(words: list[Word]) -> bool:
    """Heuristic guard for unsplit ``from X to/into Y`` constructions."""
    recent = [_normalized_word(w) for w in words[-8:]]
    return "from" in recent and "to" not in recent and "into" not in recent


def _chunk_starts_with_preposition(words: list[Word]) -> bool:
    """True when a chunk would begin with a stranded preposition."""
    if not words:
        return False
    return _normalized_word(words[0]) in _PREPOSITIONS


def _chunk_ends_with_preposition(words: list[Word]) -> bool:
    """True when a chunk would end on a preposition instead of its object."""
    if not words:
        return False
    return _normalized_word(words[-1]) in _PREPOSITIONS


def _boundary_semantic_score(words: list[Word], index: int) -> float:
    """Score semantic quality before duration is considered."""
    prev_text = _clean_word(words[index - 1].text)
    prev_word = _normalized_word(words[index - 1])
    next_word = _normalized_word(words[index])

    if prev_text and prev_text[-1] in SENTENCE_END:
        return 1000
    if prev_text and prev_text[-1] in _CLAUSE_PUNCT:
        return 760
    if next_word in _STRONG_BOUNDARY_WORDS:
        return 900
    if next_word in _MEDIUM_BOUNDARY_WORDS:
        return 620
    if next_word in _WEAK_BOUNDARY_WORDS:
        return 420
    if prev_word in _STRONG_BOUNDARY_WORDS:
        return 320
    return 100


def _duration_score(left: list[Word], right: list[Word], max_duration: float, target_left: float) -> float:
    """Reward usable subtitle durations without overriding semantic quality."""
    left_duration = _words_duration(left)
    right_duration = _words_duration(right)
    soft_max = max_duration * _SOFT_MAX_DURATION_MULTIPLIER

    score = 0.0
    if left_duration <= max_duration and right_duration <= max_duration:
        score += 180
    elif left_duration <= soft_max and right_duration <= soft_max:
        score += 80
    else:
        score -= 120 * max(0.0, max(left_duration, right_duration) - soft_max)

    score -= abs(left_duration - target_left) * 18
    return score


def _chunk_size_score(left: list[Word], right: list[Word]) -> float:
    """Penalize tiny fragments that are hard to translate naturally."""
    score = 0.0
    for chunk in (left, right):
        if len(chunk) < _MIN_CHUNK_WORDS:
            score -= 180
        if _words_duration(chunk) < _MIN_CHUNK_DURATION:
            score -= 120
    return score


def _segment_from_word_chunk(parent: Segment, words: list[Word], unit_id: str, source_text: str) -> Segment:
    """Build a segment from whole words and keep text/timing aligned."""
    return Segment(
        unit_id=unit_id,
        start=words[0].start,
        end=words[-1].end,
        speaker=parent.speaker,
        source_text=source_text or _text_from_words(words),
        words=words,
        source_cue_ids=parent.source_cue_ids,
    )


def _texts_from_parent_chunks(source_text: str, word_chunks: list[list[Word]]) -> list[str]:
    """Slice original source text at word-boundary locations.

    The transcript text can contain words without reliable timing.  We use
    timed words to choose safe boundaries, then slice the original text so
    translation does not lose untimed words.
    """
    if len(word_chunks) <= 1:
        return [source_text.strip()]

    fallback = [_text_from_words(chunk) for chunk in word_chunks]
    source_tokens = _tokenize_text(source_text)
    flat_words = [word for chunk in word_chunks for word in chunk if _normalized_word(word)]
    matched_indices = _match_words_to_source_tokens(flat_words, source_tokens)
    if matched_indices is None or len(matched_indices) != len(flat_words):
        return fallback

    boundaries = [0]
    word_offset = 0
    for chunk_index in range(1, len(word_chunks)):
        word_offset += sum(1 for word in word_chunks[chunk_index - 1] if _normalized_word(word))
        if word_offset >= len(matched_indices):
            return fallback
        boundaries.append(source_tokens[matched_indices[word_offset]].start)
    boundaries.append(len(source_text))

    texts = [source_text[boundaries[i] : boundaries[i + 1]].strip() for i in range(len(word_chunks))]
    return texts if _text_chunks_align_word_chunks(texts, word_chunks) else fallback


def _tokenize_text(text: str) -> list[_TextToken]:
    """Tokenize text with original spans for robust repeated-word alignment."""
    tokens: list[_TextToken] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isspace():
            if start is not None:
                _append_text_token(tokens, text, start, index)
                start = None
        elif start is None:
            start = index

    if start is not None:
        _append_text_token(tokens, text, start, len(text))

    return tokens


def _append_text_token(tokens: list[_TextToken], text: str, start: int, end: int) -> None:
    """Append one normalized token span when it carries lexical content."""
    value = _normalize_token(text[start:end])
    if value:
        tokens.append(_TextToken(value=value, start=start, end=end))


def _match_words_to_source_tokens(words: list[Word], source_tokens: list[_TextToken]) -> list[int] | None:
    """Match timed words to source token occurrences in monotonic order."""
    matched: list[int] = []
    cursor = 0
    for word in words:
        value = _normalized_word(word)
        if not value:
            continue
        while cursor < len(source_tokens) and source_tokens[cursor].value != value:
            cursor += 1
        if cursor >= len(source_tokens):
            return None
        matched.append(cursor)
        cursor += 1
    return matched


def _text_chunks_align_word_chunks(texts: list[str], word_chunks: list[list[Word]]) -> bool:
    """Reject sliced text that no longer contains the timed word sequence."""
    return all(_text_contains_word_sequence(text, words) for text, words in zip(texts, word_chunks, strict=True))


def _text_contains_word_sequence(text: str, words: list[Word]) -> bool:
    """True when timed words appear in order inside a text chunk."""
    text_tokens = [token.value for token in _tokenize_text(text)]
    word_tokens = [_normalized_word(word) for word in words if _normalized_word(word)]
    if not word_tokens:
        return False

    cursor = 0
    for token in text_tokens:
        if cursor < len(word_tokens) and token == word_tokens[cursor]:
            cursor += 1
    return cursor == len(word_tokens)


def _text_matches_word_tokens(text: str, words: list[Word]) -> bool:
    """True when text tokens exactly match the timed word tokens."""
    text_tokens = [token.value for token in _tokenize_text(text)]
    word_tokens = [_normalized_word(word) for word in words if _normalized_word(word)]
    return bool(word_tokens) and text_tokens == word_tokens


def _text_from_words(words: list[Word]) -> str:
    """Normalize ASR word tokens into the subtitle source text form."""
    return " ".join(clean for word in words if (clean := _clean_word(word.text)))


def _words_duration(words: list[Word]) -> float:
    """Return the span covered by a non-empty word list."""
    return words[-1].end - words[0].start if words else 0.0


def _clean_word(text: str) -> str:
    """Strip ASR token spacing without removing spoken punctuation."""
    return text.strip()


def _normalized_word(word: Word) -> str:
    """Normalize a word for boundary classification."""
    return _normalize_token(_clean_word(word.text))


def _normalize_token(text: str) -> str:
    """Normalize lexical tokens while preserving internal apostrophes."""
    return text.lower().strip(string.punctuation + "“”‘’")
