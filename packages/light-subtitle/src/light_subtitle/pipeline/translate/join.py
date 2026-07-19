"""Join pass — LLM repairs dangling/flash translated cues under hard display caps.

Runs after translation on the 1:1 cue list.  Two operation types, both
decided by the LLM and validated mechanically:

- **merge**: absorb whole dangling/flash cues into their neighbour(s).
- **shift**: move *k* EN words (with their timestamps) across a boundary
  (k>0: next→prev, k<0: prev→next), for cases where a full merge would
  exceed the display caps.  The affected pair is then re-translated for
  its new spans, so no fragile ZH text matching is involved; a failed
  re-translation or cap violation restores the original cues.

Hard validation: no overlapping operations, moved ZH text must be an
exact prefix/suffix of the donor cue (text only ever moves, never
rewrites), EN word boundaries, speaker purity, and three display caps
(chars / CPS / duration) so a fix can never recreate an overlong cue.
Plan units touched by a shift are split at the moved word boundary (id +
``a``/``b`` suffix) so bilingual EN derivation stays word-exact.  The
joined unit graph is written to ``plan/plan.joined.json``
— ``plan/plan.json`` is never modified, so re-runs are idempotent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from light_models import Segment, SubtitleCue, Word

from ... import logger
from ...config import SubtitleConfig
from ...llm.client import OpenAIClient
from ...llm.prompts import render_prompt
from ...usage.tracker import merge_token_usage
from ..plan.boundary import dangling_tail

_BATCH_CORE = 60  # cues judged per LLM call
_BATCH_CTX = 8  # context cues on each side (not judged)
_MAX_ATTEMPTS = 2

_MAX_CHARS_RATIO = 1.0  # merged ZH chars ≤ max zh chars (two display lines)
_MAX_DUR_RATIO = 1.5  # merged duration ≤ max_duration × 1.5 (pace grants merged cues the same bound)

# Recall patterns for candidate enumeration.  These only *flag* suspicious
# boundaries for the LLM to adjudicate — they never decide anything.
_DANGLE_ENDINGS = (
    "但是",
    "即使",
    "就算",
    "不过",
    "所以",
    "因为",
    "而且",
    "并且",
    "以及",
    "或者",
    "而是",
    "但",
    "而",
    "把",
    "将",
    "被",
    "使",
    "让",
    "对",
    "为",
    "从",
    "向",
    "用",
    "在",
    "与",
    "所",
    "的",
    "地",
    "得",
    "和",
    "或",
    "是",
    "会",
    "能",
    "要",
    "想",
    "像",
    "看",
    "说",
    "到底",
    "正在",
    "正",
)
_FLASH_MAX_DURATION = 1.2  # seconds; shorter cues without sentence-end punctuation are suspicious
_SENT_ENDS = "。！？…"


@dataclass
class JoinResult:
    cues: list[SubtitleCue] = field(default_factory=list)
    units: list[Segment] = field(default_factory=list)
    usage: dict | None = None
    ops_applied: int = 0


# ── Public API ────────────────────────────────────────────


def join_cues(cues: list[SubtitleCue], plan_units: list[Segment], config: SubtitleConfig) -> JoinResult:
    """Repair dangling/flash cues; returns adjusted cues and the (possibly
    split) unit graph.  No-op when there is no API key or too few cues."""
    result = JoinResult(cues=cues, units=plan_units)
    if not config.llm_api_key or len(cues) < 2:
        return result

    client = OpenAIClient(base_url=config.llm_base_url, api_key=config.llm_api_key, model=config.llm_model)
    system = render_prompt(
        "join_cues_system.j2",
        max_chars=_max_chars(config),
        max_cps=config.cps_limit,
        max_duration=round(config.max_duration * _MAX_DUR_RATIO, 1),
    )
    total_usage: dict = {}
    all_ops: list[dict] = []
    for core_start in range(0, len(cues), _BATCH_CORE):
        core = cues[core_start : core_start + _BATCH_CORE]
        ctx_before = cues[max(0, core_start - _BATCH_CTX) : core_start]
        ctx_after = cues[core_start + _BATCH_CORE : core_start + _BATCH_CORE + _BATCH_CTX]
        ops, usage = _plan_ops(client, system, cues, ctx_before, core, ctx_after, config)
        if usage:
            merge_token_usage(total_usage, usage)
        all_ops.extend(ops)

    # Apply only after every batch: op indices refer to the ORIGINAL cue
    # list, and merges shrink it — applying mid-loop would drift indices.
    shifts = [op for op in all_ops if op["type"] == "shift"]
    merges = sorted((op for op in all_ops if op["type"] == "merge"), key=lambda op: -op["from"])
    for op in shifts:
        result.cues, result.units = _apply_shift(result.cues, result.units, op, config)
        result.ops_applied += 1
    for op in merges:
        result.cues = _apply_merge(result.cues, op)
        result.ops_applied += 1
    logger.info(f"  Join: {result.ops_applied} ops applied")
    result.usage = total_usage or None
    return result


def save_joined_units(units: list[Segment], plan_dir: str | Path) -> None:
    """Persist the joined unit graph to ``plan/plan.joined.json`` (+ words)."""
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    words_map: dict[str, list[dict]] = {}
    offset = 0
    for u in units:
        meta.append(
            {
                "unit_id": u.unit_id,
                "start": round(u.start, 3),
                "end": round(u.end, 3),
                "speaker": u.speaker,
                "text": u.source_text,
                "word_start": offset,
                "word_end": offset + len(u.words),
            }
        )
        offset += len(u.words)
        if u.words:
            words_map[u.unit_id] = [
                {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence, "speaker": w.speaker}
                for w in u.words
            ]
    (plan_dir / "plan.joined.json").write_text(
        json.dumps({"version": 1, "units": meta}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (plan_dir / "segment_words.joined.json").write_text(
        json.dumps(words_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_joined_units(plan_dir: str | Path) -> list[Segment] | None:
    """Rebuild units from ``plan/plan.joined.json``; None when absent."""
    path = Path(plan_dir) / "plan.joined.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        Segment(
            unit_id=item["unit_id"],
            start=item.get("start", 0.0),
            end=item.get("end", 0.0),
            speaker=item.get("speaker", ""),
            source_text=item.get("text", ""),
            words=[],
        )
        for item in data.get("units", [])
    ]


# ── Caps ──────────────────────────────────────────────────


def _max_chars(config: SubtitleConfig) -> int:
    return 48  # two display lines of the boxed bilingual layout


def _zh_chars(text: str) -> int:
    return len(text.replace("\n", "").replace(" ", ""))


def _caps_problems(zh_text: str, start: float, end: float, config: SubtitleConfig) -> list[str]:
    """The three display caps a merge/shift result must satisfy."""
    problems = []
    chars = _zh_chars(zh_text)
    dur = end - start
    if chars > _max_chars(config):
        problems.append(f"{chars} chars over the {_max_chars(config)} cap")
    if dur > config.max_duration * _MAX_DUR_RATIO:
        problems.append(f"{dur:.1f}s over the {config.max_duration * _MAX_DUR_RATIO:.1f}s cap")
    if dur > 0 and chars / dur > config.cps_limit:
        problems.append(f"{chars / dur:.1f} cps over the {config.cps_limit} cap")
    return problems


# ── LLM planning of operations ────────────────────────────


def _cue_words(cue: SubtitleCue) -> list[Word]:
    return cue.words or []


def _plan_ops(
    client: OpenAIClient,
    system: str,
    all_cues: list[SubtitleCue],
    ctx_before: list[SubtitleCue],
    core: list[SubtitleCue],
    ctx_after: list[SubtitleCue],
    config: SubtitleConfig,
) -> tuple[list[dict], dict | None]:
    """One LLM call over a batch; returns validated ops for the core range."""
    first_idx = all_cues.index(core[0])
    core_ids = set(range(first_idx, first_idx + len(core)))
    payload = {
        "context_before": [_payload_item(all_cues.index(c), c) for c in ctx_before],
        "cues": [_payload_item(first_idx + i, c) for i, c in enumerate(core)],
        "context_after": [_payload_item(all_cues.index(c), c) for c in ctx_after],
        "candidates": [c for c in _find_candidates(all_cues) if c["boundary"] in core_ids],
    }
    total_usage: dict = {}
    feedback = ""
    for attempt in range(_MAX_ATTEMPTS):
        user = dict(payload)
        if feedback:
            user["previous_error"] = feedback
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        response, usage = client.chat(messages, temperature=config.llm_temperature)
        merge_token_usage(total_usage, usage)
        ops = _parse_ops(response)
        if ops is None:
            feedback = 'Output was not valid JSON of the form {"ops": [...]}.'
            continue
        valid, problems = _validate_ops(ops, all_cues, config, core_ids)
        if not problems:
            return valid, total_usage or None
        feedback = "Invalid ops: " + "; ".join(problems)
        logger.warning(f"  Join attempt {attempt + 1} invalid: {feedback}")
        if attempt == _MAX_ATTEMPTS - 1:
            # Last attempt: keep the valid subset, drop the offenders.
            if valid:
                logger.info(f"  Join: keeping {len(valid)} valid ops, dropping {len(problems)} invalid")
            return valid, total_usage or None
    return [], total_usage or None


def _find_candidates(cues: list[SubtitleCue]) -> list[dict]:
    """Enumerate suspicious boundaries (recall aid for the LLM).

    Flags cues that end with a dangling-style word or are flash
    fragments.  The LLM adjudicates every candidate (fix or leave);
    these patterns never decide anything on their own.
    """
    candidates = []
    for i, cue in enumerate(cues[:-1]):
        text = cue.text.rstrip()
        stripped = text.rstrip(_SENT_ENDS + "，、；：,.;:!?")
        ending = next((w for w in _DANGLE_ENDINGS if stripped.endswith(w)), None)
        dur = cue.end - cue.start
        if ending and not text.endswith(tuple(_SENT_ENDS)):
            candidates.append({"boundary": i, "reason": f"cue {i} ends with '{ending}'"})
        elif dur < _FLASH_MAX_DURATION and not text.endswith(tuple(_SENT_ENDS)):
            candidates.append({"boundary": i, "reason": f"cue {i} is a {dur:.1f}s flash fragment"})
    return candidates


def _payload_item(i: int, cue: SubtitleCue) -> dict:
    en = " ".join(w.text.strip() for w in _cue_words(cue) if w.text.strip())
    return {"i": i, "dur": round(cue.end - cue.start, 1), "zh": cue.text, "en": en}


def _parse_ops(response: str) -> list[dict] | None:
    match = re.search(r"\{[\s\S]*\}", response)
    try:
        data = json.loads(match.group(0) if match else response)
    except json.JSONDecodeError:
        return None
    raw = data.get("ops") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    ops: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        if item.get("type") == "merge" and isinstance(item.get("from"), int) and isinstance(item.get("to"), int):
            ops.append({"type": "merge", "from": item["from"], "to": item["to"]})
        elif (
            item.get("type") == "shift"
            and isinstance(item.get("boundary"), int)
            and isinstance(item.get("en_words"), int)
        ):
            ops.append(
                {
                    "type": "shift",
                    "boundary": item["boundary"],
                    "en_words": item["en_words"],
                }
            )
        else:
            return None
    return ops


def _validate_ops(
    ops: list[dict], cues: list[SubtitleCue], config: SubtitleConfig, core_ids: set[int] | None = None
) -> tuple[list[dict], list[str]]:
    """Hard checks for a batch of ops; returns (valid ops, problem strings).

    When *core_ids* is given, ops touching cues outside the judged core
    range are rejected (context cues are display-only).
    """
    problems = []
    touched: set[int] = set()
    valid: list[dict] = []
    for op in ops:
        idxs = _op_indices(op)
        err = None
        if any(i < 0 or i >= len(cues) for i in idxs):
            err = f"index out of range in {op}"
        elif core_ids is not None and not set(idxs) <= core_ids:
            err = f"op touches context cues outside the judged range: {op}"
        elif touched & set(idxs):
            err = f"overlapping ops on cues {sorted(touched & set(idxs))}"
        elif op["type"] == "merge":
            err = _validate_merge(op, cues, config)
        else:
            err = _validate_shift(op, cues, config)
        if err:
            problems.append(err)
        else:
            touched |= set(idxs)
            valid.append(op)
    return valid, problems


def _op_indices(op: dict) -> list[int]:
    if op["type"] == "merge":
        return list(range(op["from"], op["to"] + 1))
    return [op["boundary"], op["boundary"] + 1]


def _validate_merge(op: dict, cues: list[SubtitleCue], config: SubtitleConfig) -> str | None:
    start, end = op["from"], op["to"]
    if end <= start:
        return f"merge needs at least 2 cues: {op}"
    group = cues[start : end + 1]
    speakers = {c.speaker for c in group if c.speaker}
    if len(speakers) > 1:
        return f"merge {start}-{end} mixes speakers {sorted(speakers)}"
    text = _join_zh([c.text for c in group])
    problems = _caps_problems(text, group[0].start, group[-1].end, config)
    return f"merge {start}-{end}: {'; '.join(problems)}" if problems else None


def _validate_shift(op: dict, cues: list[SubtitleCue], config: SubtitleConfig) -> str | None:
    i = op["boundary"]
    k = op["en_words"]
    prev, nxt = cues[i], cues[i + 1]
    if k == 0:
        return f"shift boundary {i}: en_words must be non-zero"
    donor, n_move = (nxt, k) if k > 0 else (prev, -k)
    if n_move >= len(_cue_words(donor)):
        return f"shift boundary {i}: moving {n_move} words leaves donor empty"
    # Check if the new boundary created by the shift strands a function word.
    donor_words = _cue_words(donor)
    tail_idx = k - 1 if k > 0 else len(donor_words) + k - 1
    if 0 <= tail_idx < len(donor_words):
        bad = dangling_tail(donor_words[tail_idx])
        if bad is not None:
            return (
                f"shift boundary {i}: moving {n_move} words creates a dangling tail after "
                f"'{donor_words[tail_idx].text.strip()}' — move fewer or more words"
            )
    speakers = {c.speaker for c in (prev, nxt) if c.speaker}
    if len(speakers) > 1:
        return f"shift boundary {i}: mixes speakers {sorted(speakers)}"
    words_prev, words_next = _shifted_words(prev, nxt, k)
    max_dur = config.max_duration * _MAX_DUR_RATIO
    dur_prev = words_prev[-1].end - words_prev[0].start
    dur_next = words_next[-1].end - words_next[0].start
    if dur_prev > max_dur or dur_next > max_dur:
        return f"shift boundary {i}: {max(dur_prev, dur_next):.1f}s over the {max_dur:.1f}s cap"
    return None


def _shifted_words(prev: SubtitleCue, nxt: SubtitleCue, k: int) -> tuple[list[Word], list[Word]]:
    if k > 0:
        return _cue_words(prev) + _cue_words(nxt)[:k], _cue_words(nxt)[k:]
    return _cue_words(prev)[:k], _cue_words(prev)[k:] + _cue_words(nxt)


def _join_zh(texts: list[str]) -> str:
    return "".join(t.strip() for t in texts if t.strip())


# ── Applying operations ───────────────────────────────────


def _apply_merge(cues: list[SubtitleCue], op: dict) -> list[SubtitleCue]:
    start, end = op["from"], op["to"]
    group = cues[start : end + 1]
    head = group[0]
    head.text = _join_zh([c.text for c in group])
    head.end = group[-1].end
    absorbed = []
    for c in group[1:]:
        absorbed.extend([c.unit_id, *c.merged_from])
    head.merged_from = [*head.merged_from, *absorbed]
    head.words = [w for c in group for w in _cue_words(c)]
    return cues[: start + 1] + cues[end + 1 :]


def _apply_shift(
    cues: list[SubtitleCue], units: list[Segment], op: dict, config: SubtitleConfig
) -> tuple[list[SubtitleCue], list[Segment]]:
    """Shift EN words across a boundary, then re-translate the affected pair.

    The LLM only decides the boundary and word count; the two cues' texts
    are regenerated for their new spans (the same way translate_missing
    works), so there is no zh_move text-matching to get wrong.  When the
    re-translation fails or breaks a display cap, the original cues and
    unit graph are restored.
    """
    import copy

    i = op["boundary"]
    k = op["en_words"]
    backup_prev = copy.deepcopy(cues[i])
    backup_next = copy.deepcopy(cues[i + 1])
    backup_units = list(units)

    prev, nxt = cues[i], cues[i + 1]
    words_prev, words_next = _shifted_words(prev, nxt, k)
    moved = _cue_words(nxt)[:k] if k > 0 else _cue_words(prev)[k:]

    prev.words, nxt.words = words_prev, words_next
    prev.end = words_prev[-1].end
    nxt.start = words_next[0].start

    # Split plan units so each cue's unit chain still covers exactly its words.
    if moved:
        units, prev_chain, next_chain = _resplit_chains(
            units, prev.unit_id, prev.merged_from, nxt.unit_id, nxt.merged_from, moved, at_head=k > 0
        )
        prev.unit_id, prev.merged_from = prev_chain[0], prev_chain[1:]
        nxt.unit_id, nxt.merged_from = next_chain[0], next_chain[1:]
        for u in units:
            if u.words and dangling_tail(u.words[-1]):
                cues[i], cues[i + 1] = backup_prev, backup_next
                units[:] = backup_units
                logger.info(f"  Join: shift boundary {i} reverted (dangling tail from word split)")
                return cues, units

    texts = _retranslate_pair(prev, nxt, units, config)
    problems = []
    if texts is None:
        problems.append("re-translation failed")
    else:
        problems = _caps_problems(texts[0], prev.start, prev.end, config)
        problems += _caps_problems(texts[1], nxt.start, nxt.end, config)
    if problems:
        cues[i], cues[i + 1] = backup_prev, backup_next
        units[:] = backup_units
        logger.info(f"  Join: shift boundary {i} reverted ({'; '.join(problems)})")
        return cues, units
    prev.text, nxt.text = texts
    return cues, units


def _retranslate_pair(
    prev: SubtitleCue, nxt: SubtitleCue, units: list[Segment], config: SubtitleConfig
) -> tuple[str, str] | None:
    """Re-translate two shifted cue spans (neighbour units as context)."""
    from .translate import translate_missing  # noqa: PLC0415 — avoids import cycle

    def to_segment(cue: SubtitleCue) -> Segment:
        return Segment(
            unit_id=cue.unit_id,
            start=cue.start,
            end=cue.end,
            speaker=cue.speaker,
            source_text=" ".join(w.text.strip() for w in cue.words if w.text.strip()),
            words=list(cue.words),
        )

    prev_idx = next((i for i, u in enumerate(units) if u.unit_id == prev.unit_id), None)
    next_idx = next((i for i, u in enumerate(units) if u.unit_id == nxt.unit_id), None)
    if prev_idx is None or next_idx is None:
        return None
    lo = max(0, prev_idx - 2)
    hi = min(len(units), next_idx + 3)
    ordered = units[lo:prev_idx] + [to_segment(prev), to_segment(nxt)] + units[next_idx + 1 : hi]
    retried, _usage = translate_missing(ordered, {prev.unit_id, nxt.unit_id}, config)
    texts = {c.unit_id: c.text for c in retried}
    if prev.unit_id in texts and nxt.unit_id in texts:
        return texts[prev.unit_id], texts[nxt.unit_id]
    return None


def _resplit_chains(
    units: list[Segment],
    prev_head: str,
    prev_tail: list[str],
    next_head: str,
    next_tail: list[str],
    moved: list[Word],
    *,
    at_head: bool,
) -> tuple[list[Segment], list[str], list[str]]:
    """Split plan units so *moved* words detach from the donor's chain.

    ``at_head`` True  → moved words are the head of the next cue's chain (next→prev);
    ``at_head`` False → moved words are the tail of the prev cue's chain (prev→next).
    Returns (units, prev_chain, next_chain) with updated id lists.
    """
    by_id = {u.unit_id: u for u in units}
    donor_chain = [next_head, *next_tail] if at_head else [prev_head, *prev_tail]
    n_move = len(moved)
    # Walk the chain from the end the moved words live on.
    ordered = donor_chain if at_head else list(reversed(donor_chain))
    moved_ids_rev: list[str] = []  # chain order restored afterwards
    keep_rev: list[str] = []  # donor units/parts that stay, in walk order
    remaining = n_move
    for uid in ordered:
        unit = by_id[uid]
        nw = len(unit.words)
        take = min(remaining, nw)
        if take == 0:
            keep_rev.append(uid)
            continue
        remaining -= take
        if take == nw:
            moved_ids_rev.append(uid)  # whole unit changes sides
            continue
        # Split this unit at the moved boundary.
        moved_words = unit.words[:take] if at_head else unit.words[nw - take :]
        keep_words = unit.words[take:] if at_head else unit.words[: nw - take]
        moved_unit = Segment(
            unit_id=f"{uid}{'a' if at_head else 'b'}",
            start=moved_words[0].start,
            end=moved_words[-1].end,
            speaker=unit.speaker,
            source_text=" ".join(w.text.strip() for w in moved_words),
            words=moved_words,
        )
        keep_unit = Segment(
            unit_id=f"{uid}{'b' if at_head else 'a'}",
            start=keep_words[0].start,
            end=keep_words[-1].end,
            speaker=unit.speaker,
            source_text=" ".join(w.text.strip() for w in keep_words),
            words=keep_words,
        )
        idx = next(i for i, u in enumerate(units) if u.unit_id == uid)
        by_id[moved_unit.unit_id] = moved_unit
        by_id[keep_unit.unit_id] = keep_unit
        if at_head:
            units[idx : idx + 1] = [moved_unit, keep_unit]
            moved_ids_rev.append(moved_unit.unit_id)
            keep_rev.append(keep_unit.unit_id)
        else:
            units[idx : idx + 1] = [keep_unit, moved_unit]
            keep_rev.append(keep_unit.unit_id)
            moved_ids_rev.append(moved_unit.unit_id)
        remaining = 0

    moved_ids = moved_ids_rev if at_head else list(reversed(moved_ids_rev))
    new_donor_chain = keep_rev if at_head else list(reversed(keep_rev))
    if at_head:
        prev_chain = [prev_head, *prev_tail, *moved_ids]
        return units, prev_chain, new_donor_chain
    next_chain = [*moved_ids, next_head, *next_tail]
    return units, new_donor_chain, next_chain
