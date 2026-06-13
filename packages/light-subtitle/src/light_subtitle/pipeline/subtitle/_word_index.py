"""Word-timestamp index for subtitle cue construction.

Pre-computes whisper-token offsets → display-text word mapping,
so chunk-level word lookups avoid repeated text reconstruction
and substring matching.

Built once per semantic unit via ``_UnitWordIndex.from_words()``.
"""

from dataclasses import dataclass

from light_models import SubtitleCue, Word, is_cjk


@dataclass
class _UnitWordIndex:
    """Pre-computed lookup: whisper-token offsets → display-text word mapping."""

    words: list[Word]
    _full_text: str
    _word_offsets: list[int]
    _norm_to_raw: list[int]
    _full_norm: str

    @staticmethod
    def _normalize(text: str) -> str:
        result: list[str] = []
        for ch in text.lower():
            if ch.isalpha() or ch.isdigit():
                result.append(ch)
        return "".join(result)

    @classmethod
    def from_words(cls, words: list[Word]) -> "_UnitWordIndex | None":
        if not words:
            return None
        offsets: list[int] = []
        full_text = ""
        prev_is_cjk = bool(words) and is_cjk(words[0].text[:1]) if words[0].text else False
        for w in words:
            txt = w.text.strip()
            if not txt:
                continue
            sep = "" if prev_is_cjk and txt and is_cjk(txt[0]) else " "
            offsets.append(len(full_text) + len(sep))
            full_text += sep + txt
            prev_is_cjk = txt and is_cjk(txt[0])
        full_text = full_text.strip()
        if not full_text:
            return None
        norm_to_raw: list[int] = []
        for i, ch in enumerate(full_text.lower()):
            if ch.isalpha() or ch.isdigit():
                norm_to_raw.append(i)
        return cls(
            words=list(words),
            _full_text=full_text,
            _word_offsets=offsets,
            _norm_to_raw=norm_to_raw,
            _full_norm=cls._normalize(full_text),
        )

    def find_words(self, chunk_lines: list[str]) -> list[Word]:
        chunk_norm = self._normalize(" ".join(chunk_lines))
        pos_norm = self._full_norm.find(chunk_norm)
        if pos_norm == -1:
            first_norm = self._normalize(chunk_lines[0]) if chunk_lines else ""
            last_norm = self._normalize(chunk_lines[-1]) if chunk_lines else ""
            start_norm = self._full_norm.find(first_norm) if first_norm else 0
            end_norm = self._full_norm.rfind(last_norm) if last_norm else len(self._full_norm)
            if start_norm == -1 or end_norm == -1 or start_norm >= end_norm:
                return []
            pos_norm = start_norm
            end_norm_pos = end_norm + len(last_norm)
        else:
            end_norm_pos = pos_norm + len(chunk_norm)
        ntr = self._norm_to_raw
        if pos_norm >= len(ntr) or end_norm_pos <= 0:
            return []
        raw_pos = ntr[pos_norm]
        raw_end = ntr[min(end_norm_pos - 1, len(ntr) - 1)] + 1
        result = []
        for i, offset in enumerate(self._word_offsets):
            w_end = self._word_offsets[i + 1] if i + 1 < len(self._word_offsets) else len(self._full_text)
            if offset < raw_end and w_end > raw_pos:
                result.append(self.words[i])
        return result


def _chunk_times(
    chunk: list[str], original: SubtitleCue, word_idx: _UnitWordIndex | None, cps_limit: int
) -> tuple[float, float]:
    """Compute display-chunk start/end from word-level timestamps.

    Extends end to meet the CPS reading-speed floor (min chars/sec).
    """
    chunk_chars = sum(len(line) for line in chunk)
    min_time = max(chunk_chars / cps_limit, 0.8)
    if word_idx is not None:
        chunk_words = word_idx.find_words(chunk)
        if chunk_words:
            ws = chunk_words[0].start
            we = chunk_words[-1].end
            return ws, max(we, ws + min_time)
    return original.start, original.start + min_time
