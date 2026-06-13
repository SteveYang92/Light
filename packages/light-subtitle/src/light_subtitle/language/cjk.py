"""CJK-specific text processing — Chinese punctuation, line-breaking, and break finding.

Import explicitly from ``language.cjk``::

    from light_subtitle.language.cjk import ChineseBreakFinder, _normalize_chinese_text
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import jieba
from light_models import SubtitleCue, is_cjk
from light_models.punctuation import (
    CJK_ALL_PUNCT,
    CJK_CLAUSE_PUNCT,
    CJK_PARTICLES,
    CJK_SENTENCE_ENDS,
    CJK_SENTENCE_PARTICLES,
)

from .base import (
    BREAK_CLAUSE,
    BREAK_CONJUNCTION,
    BREAK_FALLBACK,
    BREAK_SENTENCE_END,
    BreakFinder,
)

if TYPE_CHECKING:
    from light_subtitle.config import SubtitleConfig

# ═══════════════════════════════════════════════════════════════════
# CJK grammatical particles
# ═══════════════════════════════════════════════════════════════════

# Particles that should rarely start a line
_PARTICLES = set("了着过吗呢吧啊么嘛呗的")

# Break point scores for Chinese punctuation
_CN_BREAK_PUNCT: dict[str, int] = {
    "。": BREAK_SENTENCE_END,
    "？": BREAK_SENTENCE_END,
    "！": BREAK_SENTENCE_END,
    "，": BREAK_CLAUSE,
    "、": 70,
    "；": BREAK_CLAUSE,
    "：": 65,
}

_CN_CONJUNCTIONS = {
    "但是",
    "所以",
    "而且",
    "然而",
    "因为",
    "如果",
    "虽然",
    "不过",
    "于是",
    "接着",
    "然后",
    "因此",
    "那么",
    "却",
    "但",
    "可",
    "便",
}

# ── Paired symbols ──

_PAIRED_OPEN_CLOSE: dict[str, str] = {
    "\u300a": "\u300b",  # 《 》
    "\uff08": "\uff09",  # （ ）
    "\u300c": "\u300d",  # 「 」
    "\u300e": "\u300f",  # 『 』
    "\u3010": "\u3011",  # 【 】
    "\u201c": "\u201d",  # " "
    "\u2018": "\u2019",  # ' '
}

_AMBIGUOUS_PAIRS = {'"', "'"}

# ── Display punctuation (strip_punct) ──

_FULLWIDTH_SPACE = "\u3000"

_REPLACE_WITH_SPACE = {"，", "、", "；", "："}

_PERIOD = "。"
_KEEP = {"？", "！", "…"}

# ═══════════════════════════════════════════════════════════════════
# Text utilities
# ═══════════════════════════════════════════════════════════════════


def _is_cjk_or_kana(ch: str) -> bool:
    return is_cjk(ch) or "\u3040" <= ch <= "\u30ff"


def _join_text(words: list) -> str:
    if not words:
        return ""
    sample = "".join(w.text for w in words[:10])
    cjk_count = sum(1 for ch in sample if _is_cjk_or_kana(ch))
    if cjk_count > len(sample) * 0.3:
        return "".join(w.text.strip() for w in words)
    return " ".join(t for w in words if (t := w.text.strip()))


def _is_latin(ch: str) -> bool:
    if "a" <= ch <= "z" or "A" <= ch <= "Z":
        return True
    return "\u00c0" <= ch <= "\u024f"


def _splits_english_words(text: str, pos: int) -> bool:
    if pos <= 0 or pos >= len(text):
        return False
    before_end = pos - 1
    while before_end >= 0 and text[before_end] == " ":
        before_end -= 1
    if before_end < 0:
        return False
    after_stripped = text[pos:].lstrip()
    if not after_stripped:
        return False
    return _is_latin(text[before_end]) and _is_latin(after_stripped[0])


def _forbid_paired_symbols(text: str) -> set[int]:
    forbidden: set[int] = set()
    stack: list[tuple[str, int]] = []

    for i, ch in enumerate(text):
        if ch in _PAIRED_OPEN_CLOSE:
            stack.append((ch, i))
        elif ch in _AMBIGUOUS_PAIRS:
            if stack and stack[-1][0] == ch:
                stack.pop()
            else:
                stack.append((ch, i))
        else:
            close = _PAIRED_OPEN_CLOSE.values()
            if ch in close:
                if stack and ch == _PAIRED_OPEN_CLOSE.get(stack[-1][0]):
                    open_pos = stack.pop()[1]
                    for p in range(open_pos + 1, i + 1):
                        forbidden.add(p)
                else:
                    for p in range(i + 1):
                        forbidden.add(p)
                continue

    for _, open_pos in stack:
        for p in range(open_pos + 1, len(text)):
            forbidden.add(p)

    return forbidden


# ═══════════════════════════════════════════════════════════════════
# Chinese text normalization
# ═══════════════════════════════════════════════════════════════════


def _normalize_chinese_text(text: str) -> str:
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"([，。？！、；：])\s+", r"\1", text)
    text = re.sub(r"\s+([，。？！、；：])", r"\1", text)
    _CJK = r"[\u4e00-\u9fff\u3400-\u4dbf]"
    text = re.sub(rf"({_CJK})\s+({_CJK})", r"\1\2", text)
    return text


def normalize_punctuation(text: str, lang: str) -> str:
    if lang != "zh":
        return text
    stripped = text.strip()
    if not stripped:
        return text
    if stripped[-1] not in CJK_SENTENCE_ENDS:
        last = stripped[-1]
        if last in CJK_CLAUSE_PUNCT:
            return stripped[:-1] + "。"
        return stripped + "。"
    return text


# ═══════════════════════════════════════════════════════════════════
# Chinese line-breaking
# ═══════════════════════════════════════════════════════════════════


def _break_chinese_line(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    finder = ChineseBreakFinder(text)
    lo = 1
    hi = min(max_chars, len(text) - 1)
    best_pos = finder.find_balanced(lo, hi, max_chars)

    first = text[:best_pos].rstrip()
    rest = text[best_pos:].lstrip()
    result = [first]
    result.extend(_break_chinese_line(rest, max_chars))

    if len(result) >= 2 and len(result[-1]) == 1:
        if len(result[-2]) + 1 <= max_chars + 2:
            result[-2] = result[-2] + result[-1]
            result.pop()

    return result


def _chinese_mend_lines(lines: list[str], max_chars: int) -> list[str]:
    if len(lines) < 2:
        return lines

    mended = [lines[0]]
    for line in lines[1:]:
        while line and (line[0] in CJK_ALL_PUNCT or line[0] in _PARTICLES):
            mended[-1] = mended[-1] + line[0]
            line = line[1:]
        if line:
            mended.append(line)
    return mended


# ═══════════════════════════════════════════════════════════════════
# Chinese rebalance helper
# ═══════════════════════════════════════════════════════════════════


def _rebalance_chinese_pair(lines: list[str]) -> list[str]:
    if len(lines) != 2:
        return lines
    diff = abs(len(lines[0]) - len(lines[1]))
    if diff <= 3:
        return lines

    combined = lines[0] + lines[1]
    total_chars = len(combined)
    boundary = len(lines[0])
    finder = ChineseBreakFinder(combined)
    scan_radius = max(5, boundary // 3)
    lo = max(boundary - scan_radius, 1)
    hi = min(boundary + scan_radius, total_chars - 1)
    if lo > hi:
        return lines

    best_pos = boundary
    best_score = finder.score(boundary)
    relaxed_max = max(16, len(lines[0]), len(lines[1]))
    for pos in range(lo, hi + 1):
        if pos > 0 and finder.is_forbidden(pos - 1):
            continue
        left_chars = pos
        right_chars = total_chars - pos
        if left_chars < 4 or right_chars < 4:
            continue
        if left_chars > relaxed_max or right_chars > relaxed_max:
            continue
        score = finder.score(pos)
        bal = min(left_chars, right_chars) / max(left_chars, right_chars)
        score += bal * 3
        if score > best_score:
            best_score = score
            best_pos = pos

    remaining = combined[best_pos:]
    if remaining and remaining[0] in CJK_ALL_PUNCT:
        return lines
    if len(remaining) <= 5:
        for shift in range(2, min(6, best_pos)):
            candidate = best_pos - shift
            if candidate > 0 and finder.is_forbidden(candidate - 1):
                continue
            new_l2 = combined[candidate:]
            if len(new_l2) > 16 or len(new_l2) < 4:
                continue
            best_pos = candidate
            break

    if best_pos != boundary:
        lines[0] = combined[:best_pos].rstrip()
        lines[1] = combined[best_pos:].lstrip()

    return lines


# ═══════════════════════════════════════════════════════════════════
# Chinese-specific: single cue creation
# ═══════════════════════════════════════════════════════════════════


def _make_chinese_cue(original: SubtitleCue, lines: list[str]) -> SubtitleCue:
    if len(lines) >= 2 and len(lines[0]) > len(lines[1]) + 4:
        lines = _rebalance_chinese_pair(lines)
    return SubtitleCue(
        cue_id=original.cue_id,
        unit_id=original.unit_id,
        start=original.start,
        end=original.end,
        text="\n".join(lines),
        lang=original.lang,
        speaker=original.speaker,
        words=list(original.words) if original.words else [],
    )


def split_chinese(cue: SubtitleCue, text: str, config: SubtitleConfig) -> list[SubtitleCue]:
    max_chars = config.max_chars_per_line_zh
    text = _normalize_chinese_text(text.strip())
    if not text:
        return [cue]

    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()] if "\n" in text else [text]

    lines: list[str] = []
    for ln in raw_lines:
        lines.extend(_break_chinese_line(ln, max_chars))
    lines = _chinese_mend_lines(lines, max_chars)

    return [_make_chinese_cue(cue, lines)]


# ═══════════════════════════════════════════════════════════════════
# Strip Chinese punctuation (post-formatting → pre-export)
# ═══════════════════════════════════════════════════════════════════


def _append_space(result: list[str]) -> None:
    if result and result[-1] == _FULLWIDTH_SPACE:
        return
    result.append(_FULLWIDTH_SPACE)


def _strip_line(line: str) -> str:
    result: list[str] = []
    for i, ch in enumerate(line):
        if ch in _REPLACE_WITH_SPACE:
            _append_space(result)
        elif ch == _PERIOD:
            if i == len(line) - 1:
                continue
            _append_space(result)
            if i + 1 < len(line) and line[i + 1] == _PERIOD:
                continue
        elif ch == " ":
            if result and result[-1] == _FULLWIDTH_SPACE:
                continue
            _append_space(result)
        else:
            result.append(ch)
    text = "".join(result).strip()
    return re.sub(r"\u3000{2,}", "\u3000", text)


def strip_chinese_punct(cues: list[SubtitleCue], config: SubtitleConfig | None = None) -> list[SubtitleCue]:
    for cue in cues:
        if cue.lang != "zh":
            continue
        lines = cue.text.split("\n")
        stripped = []
        for line in lines:
            stripped.append(_strip_line(line))
        cue.text = "\n".join(stripped)
    return cues


# ═══════════════════════════════════════════════════════════════════
# Heal split Chinese words across cue boundaries
# ═══════════════════════════════════════════════════════════════════


def mend_split_chinese_words(cues: list[SubtitleCue]) -> None:
    if len(cues) < 2:
        return

    for i in range(len(cues) - 1):
        curr = cues[i]
        nxt = cues[i + 1]
        if curr.lang != "zh" or nxt.lang != "zh":
            continue

        curr_text = curr.text.replace("\n", "").strip()
        nxt_text = nxt.text.replace("\n", "").strip()
        if not curr_text or not nxt_text:
            continue

        combined = curr_text + nxt_text
        tokens = list(jieba.tokenize(combined))
        if not tokens:
            continue

        cut_pos = len(curr_text)
        broken = False
        for word, start, end in tokens:
            if start < cut_pos < end:
                if end - start <= 2:
                    continue
                if len(word) == 1:
                    continue
                if word[-1] in "的了时地得":
                    continue
                broken = True
                break

        if not broken:
            continue

        if len(curr_text) + len(nxt_text) > 20:
            continue

        curr.text = combined
        nxt.text = ""


# ═══════════════════════════════════════════════════════════════════
# Chinese BreakFinder
# ═══════════════════════════════════════════════════════════════════


class ChineseBreakFinder(BreakFinder):
    """Break finder for Chinese text.

    Forbidden positions: inside jieba tokens, inside paired symbols,
    between consecutive English words, inside URLs.
    """

    def __init__(self, text: str):
        super().__init__(text)
        self._forbidden = self._build_forbidden()

    def _build_forbidden(self) -> set[int]:
        text = self.text
        forbidden: set[int] = set()

        # ASCII / Latin-1 words — forbid splitting inside them
        for m in re.finditer(r"[a-zA-Z\u00c0-\u024f]+", text):
            for pos in range(m.start() + 1, m.end()):
                forbidden.add(pos)

        # jieba Chinese tokens
        for _word, start, end in jieba.tokenize(text):
            if len(_word) >= 2:
                for pos in range(start + 1, end):
                    forbidden.add(pos)

        # Paired symbols
        forbidden |= _forbid_paired_symbols(text)

        return forbidden

    def is_forbidden(self, pos: int) -> bool:
        if pos in self._forbidden:
            return True
        if pos < len(self.text):
            if self.text[pos] == "." and pos + 1 < len(self.text) and self.text[pos + 1].isalpha():
                return True
            if pos > 0 and self.text[pos - 1] == "." and self.text[pos].isalpha():
                return True
        return _splits_english_words(self.text, pos)

    def score(self, pos: int) -> int:
        text = self.text
        score_val = BREAK_FALLBACK

        if pos > 0:
            score_val = max(score_val, _CN_BREAK_PUNCT.get(text[pos - 1], 0))

        if pos < len(text):
            for conj in _CN_CONJUNCTIONS:
                if text.startswith(conj, pos):
                    score_val = max(score_val, BREAK_CONJUNCTION)
                    break
            for conj in _CN_CONJUNCTIONS:
                if pos - len(conj) + 1 >= 0 and text.startswith(conj, pos - len(conj) + 1):
                    score_val = max(score_val, BREAK_CONJUNCTION)
                    break
            if text[pos] in CJK_PARTICLES:
                score_val = max(score_val, 30)

        return score_val


# ═══════════════════════════════════════════════════════════════════
# Standalone fragment check
# ═══════════════════════════════════════════════════════════════════


def _can_stand_alone_zh(text: str) -> bool:
    text = text.replace("\n", "").strip()
    if not text or len(text) <= 1:
        return False
    first = text[0]
    if all(ch in CJK_ALL_PUNCT for ch in text):
        return False
    if first in CJK_PARTICLES:
        return False
    if first in CJK_CLAUSE_PUNCT:
        return False
    if len(text) <= 2 and all(c in CJK_SENTENCE_PARTICLES for c in text):
        return False
    return True
