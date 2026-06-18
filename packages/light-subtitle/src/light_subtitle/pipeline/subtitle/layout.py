"""Layout (断句) — split text per-cue into viewer-friendly lines.

Language-specific line-breaking is delegated to ``language.cjk`` and
``language.english``.  This module owns the dispatcher and the
cross-cue merge logic that runs before pace.
"""

from light_models import SubtitleCue
from light_models.punctuation import CJK_ALL_PUNCT, CJK_CLAUSE_PUNCT, CJK_PARTICLES, CJK_SENTENCE_PARTICLES

from ...config import SubtitleConfig
from ...language.cjk import _rebalance_chinese_pair, split_chinese
from ...language.english import split_english

# ═══════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════


def _split_cue(cue: SubtitleCue, config: SubtitleConfig) -> list[SubtitleCue]:
    """Dispatch to Chinese or English line-breaking.

    Calls split_chinese() / split_english() from language.cjk / language.english.
    """
    text = cue.text.strip()
    if not text:
        return [cue]
    if cue.lang == "zh":
        return split_chinese(cue, text, config)
    else:
        return split_english(cue, text, config)


# ═══════════════════════════════════════════════════════════════════
# Merge short adjacent cues (runs before pace)
# ═══════════════════════════════════════════════════════════════════

_SENTENCE_ENDS = set("。？！.!?")


def _try_merge(
    prev: SubtitleCue,
    curr: SubtitleCue,
    config: SubtitleConfig,
    *,
    is_single_char: bool = False,
    force_overflow: bool = False,
) -> list[SubtitleCue] | None:
    """Merge curr into prev with structural validity checks.

    Appends curr's text to prev's last line, then validates:
      - max_lines (≤ config.max_lines)
      - Orphan lines (language-aware: ≤ 1 zh, ≤ 2 en)
      - max_chars (re-split if overflow)

    Returns list of cues on success, None on failure.
    *force_overflow* relaxes orphan/char limits for semantic merges.
    """
    prev_lines = prev.text.split("\n")
    curr_text = curr.text.strip()

    # ── Language-specific line budget ──
    max_lines = config.max_lines_zh if prev.lang == "zh" else config.max_lines

    # ── Append curr text to prev ──
    prev_last = prev_lines[-1].strip()
    if prev_last and prev_last[-1] in _SENTENCE_ENDS:
        # Start a new line only when under the line budget.
        if len(prev_lines) < max_lines:
            prev_lines.append(curr_text)
        else:
            sep = "" if prev.lang == "zh" else " "
            prev_lines[-1] = prev_lines[-1] + sep + curr_text
    elif is_single_char:
        prev_lines[-1] += curr_text
    else:
        sep = "" if prev.lang == "zh" else " "
        prev_lines[-1] = prev_lines[-1] + sep + curr_text

    # ── Max lines guard ──
    if len(prev_lines) > max_lines:
        return None

    # ── Language-aware orphan check ──
    orphan_threshold = 1 if prev.lang == "zh" else 2
    if any(len(ln.strip()) <= orphan_threshold for ln in prev_lines if ln.strip()):
        if not force_overflow:
            return None

    # Rebalance 2-line Chinese cues before further processing.
    if prev.lang == "zh" and len(prev_lines) == 2:
        prev_lines = _rebalance_chinese_pair(prev_lines)

    merged = SubtitleCue(
        cue_id=prev.cue_id,
        unit_id=prev.unit_id,
        start=prev.start,
        end=curr.end,
        text="\n".join(prev_lines),
        lang=prev.lang,
        speaker=prev.speaker,
        words=(prev.words or []) + (curr.words or []),
        qc=prev.qc,
        merged_from=list(prev.merged_from),
    )
    max_chars = config.max_chars_per_line_zh if prev.lang == "zh" else config.max_chars_per_line_en
    if max(len(ln) for ln in prev_lines) <= max_chars:
        return [merged]
    if force_overflow and prev.lang != "zh":
        return [merged]

    # Re-split when lines overflow max_chars.
    re_split = _split_cue(merged, config)
    orphan_chars = 6 if force_overflow else 3
    if 1 <= len(re_split) <= max_lines:
        if not any(len(c.text.replace("\n", "").strip()) <= orphan_chars for c in re_split):
            # Re-balance 2-line Chinese cues after merge.
            if len(re_split) == 1:
                merged_lines = re_split[0].text.split("\n")
                if len(merged_lines) == 2:
                    merged_lines = _rebalance_chinese_pair(merged_lines)
                    re_split[0].text = "\n".join(merged_lines)
            return re_split
    # Safe fallback: keep original cues separate.
    return None


# ── Conjunctions for cross-cue merge ──

_CONJUNCTIONS_MERGE = {
    # Chinese
    "所以",
    "因此",
    "那么",
    "不过",
    "但是",
    "而且",
    "然而",
    "于是",
    "接着",
    "然后",
    "却",
    "但",
    "可",
    "便",
    # English
    "And",
    "But",
    "So",
    "Or",
    "Yet",
    "However",
    "Then",
    "Also",
    "and",
    "but",
    "so",
    "or",
    "yet",
    "however",
    "then",
    "also",
}


def _is_conjunction(text: str) -> bool:
    """True if *text* is a pure conjunction / transition word."""
    stripped = text.replace("\n", "").strip().rstrip(",.;，、。")
    return stripped in _CONJUNCTIONS_MERGE


# ── Syntactic continuity helpers ──

_PREPOSITIONS = {
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "of",
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
    "besides",
    "above",
    "below",
    "near",
    "inside",
    "outside",
    "beneath",
    "underneath",
}

_CONJUNCTION_LIKE = _CONJUNCTIONS_MERGE | {
    # Also treat these as continuations when they start a cue:
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "what",
    "if",
    "whether",
    "as",
    "than",
    "like",
    "such",
    "including",
}

# ── Lowercase-start exceptions (valid English sentence starters) ──

_LOWERCASE_EXCEPTIONS = {
    # Products / brands that start lowercase but are proper nouns
    "iphone",
    "ipad",
    "ipod",
    "imac",
    "macos",
    "ios",
    "ipados",
    "ebay",
    "etrade",
    # Scientific terms
    "ph",
    "pka",
    "pkb",
    "ml",
    "cm",
    "mm",
    "km",
    "kg",
}


# ═══════════════════════════════════════════════════════════════════
# Unified "can stand alone" judgment
# ═══════════════════════════════════════════════════════════════════


def _can_stand_alone(cue: SubtitleCue, config: SubtitleConfig) -> bool:
    """Can ``cue`` stand alone as a valid subtitle?

    A cue cannot stand alone when:
      - Content is too short (≤ 1 char zh, ≤ 2 chars en)
      - Content is punctuation-only
      - It is a syntactic fragment (depends on context)
      - Duration is far below min_duration
    """
    text = cue.text.replace("\n", "").strip()
    if not text:
        return False

    # ── Too short to be meaningful ──
    if len(text) <= 1:
        return False

    lang = cue.lang

    # ═══════════════════════════════════════════════════════════════
    # Chinese fragment checks
    # ═══════════════════════════════════════════════════════════════
    if lang == "zh":
        first = text[0]
        # Punctuation-only → fragment
        if all(ch in CJK_ALL_PUNCT for ch in text):
            return False
        # Leading possessive/adverbial → needs context
        if first in CJK_PARTICLES:
            return False
        # Leading mid-sentence punctuation → continuation
        if first in CJK_CLAUSE_PUNCT:
            return False
        # Single modal particle → fragment
        if len(text) <= 2 and all(c in CJK_SENTENCE_PARTICLES for c in text):
            return False
        return True

    # ═══════════════════════════════════════════════════════════════
    # English fragment checks
    # ═══════════════════════════════════════════════════════════════
    first_word = text.split()[0].rstrip(",.;:") if text.split() else ""
    first_char = text.lstrip()[0] if text.lstrip() else ""

    if len(text) <= 2:
        return False

    # Starts with lowercase → continuation (unless known exception)
    if first_char.islower():
        if first_word.lower() not in _LOWERCASE_EXCEPTIONS:
            return False

    # Starts with preposition / conjunction-like word → continuation
    if first_word.lower() in _CONJUNCTION_LIKE | _PREPOSITIONS:
        return False

    return True


# ── Cosmetic fix: lightweight text-only repair for merge failures ──


def _remove_first_char(text: str, ch: str) -> str:
    """Remove the first occurrence of *ch* from *text*, handling newlines."""
    idx = text.find(ch)
    if idx == -1:
        return text
    return text[:idx] + text[idx + 1 :]


def _cosmetic_fix(prev: SubtitleCue, curr: SubtitleCue) -> bool:
    """Try a text-only fix when full merge is structurally impossible.

    Only moves 1-2 characters between cues. Never changes timing or line count.
    Returns True if prev and/or curr were modified in-place.
    """
    curr_text = curr.text.replace("\n", "").strip()
    if not curr_text:
        return False

    prev_lines = prev.text.split("\n")
    prev_last = prev_lines[-1]

    # Case 1: leading CJK clause punctuation → move to prev's last line
    if curr_text[0] in CJK_CLAUSE_PUNCT:
        prev_lines[-1] = prev_last + curr_text[0]
        prev.text = "\n".join(prev_lines)
        curr.text = _remove_first_char(curr.text, curr_text[0])
        if not curr.text.strip():
            prev.end = max(prev.end, curr.end)
        return True

    # Case 2: single non-alpha orphan → append to prev
    if len(curr_text) == 1 and (not curr_text.isalpha() or "\u4e00" <= curr_text <= "\u9fff"):
        prev_lines[-1] = prev_last + curr_text
        prev.text = "\n".join(prev_lines)
        prev.end = max(prev.end, curr.end)
        prev.words = (prev.words or []) + (curr.words or [])
        curr.text = ""  # absorbed into prev
        return True

    # Case 3: standalone conjunction → append to prev
    if _is_conjunction(curr_text):
        sep = "" if prev.lang == "zh" else " "
        prev_lines[-1] = prev_last + sep + curr_text
        prev.text = "\n".join(prev_lines)
        prev.end = max(prev.end, curr.end)
        prev.words = (prev.words or []) + (curr.words or [])
        curr.text = ""  # absorbed into prev
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# Merge short adjacent cues (runs before pace)
# ═══════════════════════════════════════════════════════════════════


def _merge_short_adjacent(cues, config):
    """Merge adjacent cues that cannot stand alone.

    Unified strategy — two actions instead of five:

      1. **Backward merge** (default): if ``curr`` cannot stand alone
         (``_can_stand_alone``), try merging into ``prev``.

      2. **Forward merge** (conjunctions): if ``curr`` is a pure conjunction
         and backward is not applicable, merge ``curr`` into ``next`` cue.

    All merges are validated by ``_try_merge`` with structural guards:
    max_lines, orphan lines (language-aware), and max_chars.  If any
    guard fails, the merge is safely skipped (cues stay separate).
    """
    if len(cues) < 2:
        return cues

    result = [cues[0]]
    skip_next = False

    for i in range(1, len(cues)):
        if skip_next:
            skip_next = False
            continue

        prev = result[-1]
        curr = cues[i]
        curr_text = curr.text.replace("\n", "").strip()
        if not curr_text:
            continue

        # ── 1. Forward merge: pure conjunction → merge into next cue ──
        if _is_conjunction(curr_text) and i + 1 < len(cues):
            forward = _try_merge(curr, cues[i + 1], config, force_overflow=True)
            if forward is not None:
                result.extend(forward)
                skip_next = True
                continue

        # ── 2. Backward merge: curr cannot stand alone → merge into prev ──
        gap = curr.start - prev.end
        if not _can_stand_alone(curr, config) and (prev.unit_id == curr.unit_id or gap <= 0.3):
            merged = _try_merge(prev, curr, config)
            if merged is not None:
                result[-1:] = merged
                continue
            # Full merge failed → try cosmetic fix (1-2 char move only)
            if _cosmetic_fix(prev, curr):
                result[-1] = prev
                if not curr.text.strip():
                    continue  # curr absorbed

        # ── 3. Can stand alone → keep as-is ──
        result.append(curr)

    return result


# ── Public API ──


def prepare(cues, config):
    """Split each cue's text into display-ready lines, then merge
    adjacent cues too short to stand alone.

    Returns a clean cue list ready for time-axis corrections (pace).
    This merge MUST run before pace because pace's gap/CPS calculations
    depend on the final cue structure.
    """
    result = []
    for cue in cues:
        result.extend(_split_cue(cue, config))
    result = _merge_short_adjacent(result, config)
    return result
