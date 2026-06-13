"""Polish (修边) — fix visual artifacts across cue boundaries.

After layout splits text into screens and pace assigns display time,
polish repairs cross-cue visual artifacts:

  1. SplitNameHeal   — Yann | LeCun → merged into same cue
  2. ChineseWordHeal — jieba-detected split tokens re-joined

NOTE: Orphan character / leading punctuation / standalone conjunction
merges are now handled upstream by layout._cosmetic_fix during prepare,
so they are no longer needed here.

Usage::
    from .polish import repair
    cues = repair(cues)
"""

from light_models.punctuation import CLAUSE_PUNCT, SENTENCE_ENDS

_ALL_PUNCT = SENTENCE_ENDS + CLAUSE_PUNCT


def repair(cues):
    """Fix cross-cue visual artifacts left by layout and pace.

    Each pass makes targeted repairs without affecting the core
    structure (cue count, timing boundaries) established earlier.
    """
    cues = _mend_split_names(cues)
    cues = _mend_split_chinese_words(cues)
    return cues


# ═══════════════════════════════════════════════════════════════════
# 1. Heal split proper names
# ═══════════════════════════════════════════════════════════════════


def _mend_split_names(cues):
    """Heal cues where a proper name (e.g. Yann | LeCun) was split
    across consecutive cues."""
    if len(cues) < 2:
        return cues
    result = [cues[0]]
    for i in range(1, len(cues)):
        prev = result[-1]
        curr = cues[i]
        prev_lines = prev.text.split("\n")
        prev_last = prev_lines[-1].strip()
        curr_lines = curr.text.split("\n")
        curr_first = curr_lines[0].strip()

        prev_is_name_tail = (
            prev_last
            and prev_last[0].isupper()
            and len(prev_last) <= 6
            and prev_last.isascii()
            and prev_last[-1] not in _ALL_PUNCT
        )
        curr_is_name_head = (curr_first and curr_first.split()[0][0].isupper()) if curr_first else False

        if prev_is_name_tail and curr_is_name_head:
            prev_lines[-1] = ""
            prev_new = [line for line in prev_lines if line.strip()]
            if not prev_new:
                prev_new = [" "]
            prev.text = "\n".join(prev_new)
            result[-1] = prev
            curr_lines[0] = prev_last + " " + curr_first
            curr.text = "\n".join(curr_lines)
            result.append(curr)
        else:
            result.append(curr)
    return result


# ═══════════════════════════════════════════════════════════════════
# 2. Heal split Chinese words
# ═══════════════════════════════════════════════════════════════════


def _mend_split_chinese_words(cues):
    """Heal Chinese words split across consecutive cues using jieba.

    Only applies to Chinese-language cues (lang == "zh").
    English/en cues are skipped to avoid false positives.

    e.g. cue A: "...方法，过"  cue B: "去几十年里..."
         → cue A: "...方法，"  cue B: "过去几十年里..."
    """
    import jieba

    if len(cues) < 2:
        return cues
    result = [cues[0]]
    for i in range(1, len(cues)):
        prev = result[-1]
        curr = cues[i]
        # Only apply to Chinese text — English word pairs are
        # handled correctly by space-separated text joining.
        if prev.lang != "zh" and curr.lang != "zh":
            result.append(curr)
            continue
        prev_lines = prev.text.split("\n")
        prev_last = prev_lines[-1].strip()
        curr_lines = curr.text.split("\n")
        curr_first = curr_lines[0].strip()
        if not prev_last or not curr_first:
            result.append(curr)
            continue
        combined = prev_last + curr_first
        boundary = len(prev_last)
        tokens = list(jieba.tokenize(combined))
        for word, start, end in tokens:
            if len(word) >= 2 and start < boundary < end:
                tail = word[: boundary - start]
                # Single-char tails are grammatical particles (的/了/时/地/得…),
                # not genuinely split words. Moving them creates false positives
                # like shifting "时" from "做预训练时" to the next cue.
                if len(tail) < 2:
                    continue
                if len(tail) >= len(prev_last):
                    break
                if prev_lines[-1].endswith(tail):
                    prev_lines[-1] = prev_lines[-1][: -len(tail)].rstrip()
                else:
                    prev_lines[-1] = prev_lines[-1].rstrip(tail)
                if not prev_lines[-1]:
                    prev_lines.pop()
                prev.text = "\n".join(prev_lines) if prev_lines else " "
                result[-1] = prev
                curr_lines[0] = tail + curr_lines[0]
                curr.text = "\n".join(curr_lines)
                break
        result.append(curr)
    return result
