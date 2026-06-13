"""TranslationQuality — detect translation quality issues without LLM.

Operates in two modes:

- **Bilingual mode** (Scene C / pipeline with source+target): compares
  source↔target using ``pair_bilingual`` to detect: number/fact loss,
  negation loss, proper noun over-generalization, compression issues.

- **Monolingual mode** (Scene B / standalone translated SRT): checks
  patterns in translated text only — translationese connectors, filler
  words, single-character compression.
"""

from __future__ import annotations

import re

from light_models import QCIssue, SubtitleCue, is_cjk, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class TranslationQuality(SoftRule):
    """Detect common translation issues without requiring an LLM."""

    name = "TranslationQuality"
    default_severity = "warning"

    # ── Monolingual patterns ────────────────────────────────────

    # English: machine-translation / over-literal style connectors.
    MT_PATTERNS_EN = [
        (r"\bmoreover\b", "'moreover' sounds stiff; prefer 'also' or omit"),
        (r"\bnevertheless\b", "'nevertheless' sounds stiff; prefer 'but' or 'still'"),
        (r"\bfurthermore\b", "'furthermore' sounds stiff; prefer 'also'"),
        (r"\bconsequently\b", "'consequently' sounds stiff; prefer 'so'"),
        (r"\btherefore\b", "reconsider 'therefore' in spoken subtitles"),
        (r"not only.*but also", "'not only ... but also' is over-literal; rephrase"),
    ]

    # Chinese: translationese patterns to flag.
    ZH_TRANSLATIONESE = [
        (r"\b此外\b", "翻译腔：'此外' → 建议使用 '还有' / '而且' 或省略"),
        (r"\b另外\b", "翻译腔：'另外' → 建议使用 '还有' 或省略"),
        (r"\b然而\b", "翻译腔：'然而' → 建议使用 '但' / '不过'"),
        (r"\b尽管如此\b", "翻译腔：'尽管如此' → 建议使用 '但' / '不过'"),
        (r"\b值得注意的是\b", "翻译腔：'值得注意的是' → 建议使用 '关键是' / '重点是'"),
        (r"\b在这个意义上\b", "翻译腔：'在这个意义上' → 建议使用 '就是说' / '简单说'"),
        (r"\b进行(讨论|分析|研究|处理|优化|测试)\b", "翻译腔：'进行\\1' → 建议直接使用动词"),
        (r"\b该(技术|方法|系统|问题|模型|数据)\b", "翻译腔：'该\\1' → 建议使用 '这个' / '这种' 或代词"),
        (r"\b就[\u4e00-\u9fff]{1,6}而言\b", "翻译腔：'就...而言' → 建议使用 '说到...' / '对...来说'"),
    ]

    # Filler words that clutter subtitles.
    FILLER_WORDS = {"well", "you know", "i mean", "就是说", "那个", "嗯", "呃"}

    # ── Bilingual patterns ──────────────────────────────────────

    # English number patterns to check for in source vs target.
    _SOURCE_NUMBER_RE = re.compile(
        r"(\d+[.,]?\d*\s*(?:percent|%|minutes?|seconds?|hours?|days?|years?|"
        r"million|billion|thousand|hundred|dollars?|times?|people|times|x))"
        r"|(\b\d+[.,]?\d*\b)",
    )

    # English negation words.
    _NEGATION_WORDS = {
        "not",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "can't",
        "cannot",
        "couldn't",
        "shouldn't",
        "never",
        "no",
        "nothing",
        "nobody",
        "nowhere",
        "neither",
        "nor",
        "hardly",
        "barely",
        "scarcely",
    }

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        """Run both monolingual and (if available) bilingual checks."""
        issues: list[QCIssue] = []

        # ── Phase 1: Monolingual checks (always run) ───────────
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                issues.extend(self._check_monolingual(cue, i))

        # ── Phase 2: Bilingual checks (when source is available) ─
        if config.source_lang and config.source_lang in cues:
            issues.extend(self._check_bilingual(cues, config))

        return issues

    # ── Monolingual checks ─────────────────────────────────────

    def _check_monolingual(self, cue: SubtitleCue, idx: int) -> list[QCIssue]:
        """Pattern-based checks on a single cue (no source comparison)."""
        issues: list[QCIssue] = []
        text_lower = cue.text.lower()

        # Machine-translation patterns (English only).
        if cue.lang == "en":
            for pattern, detail in self.MT_PATTERNS_EN:
                if re.search(pattern, text_lower):
                    issues.append(
                        QCIssue(
                            severity="suggestion",
                            category="柔性策略",
                            rule=self.name,
                            cue_id=idx + 1,
                            time=seconds_to_srt(cue.start),
                            detail=detail,
                            fix="使用更自然的口语表达",
                        )
                    )
                    break  # one pattern per cue is enough

        # Translationese patterns (Chinese only).
        if cue.lang == "zh":
            for pattern, detail in self.ZH_TRANSLATIONESE:
                if re.search(pattern, cue.text):
                    issues.append(
                        QCIssue(
                            severity="suggestion",
                            category="柔性策略",
                            rule=self.name,
                            cue_id=idx + 1,
                            time=seconds_to_srt(cue.start),
                            detail=detail,
                            fix="使用更自然的中文口语表达",
                        )
                    )
                    break  # one pattern per cue

        # Filler words.
        for filler in self.FILLER_WORDS:
            if filler in text_lower:
                issues.append(
                    QCIssue(
                        severity="suggestion",
                        category="柔性策略",
                        rule=self.name,
                        cue_id=idx + 1,
                        time=seconds_to_srt(cue.start),
                        detail=f"包含填充词 '{filler}'，建议省略",
                        fix="删除填充词，保持字幕简洁",
                    )
                )
                break  # one filler per cue is enough

        # Severely compressed single-character Chinese.
        stripped = cue.text.replace("\n", "").strip()
        if cue.lang == "zh" and len(stripped) == 1 and is_cjk(stripped[0]):
            issues.append(
                QCIssue(
                    severity="warning",
                    category="柔性策略",
                    rule=self.name,
                    cue_id=idx + 1,
                    time=seconds_to_srt(cue.start),
                    detail=f"字幕仅包含一个汉字 '{stripped}'，可能压缩过度",
                    fix="检查翻译是否丢失了关键信息",
                )
            )

        return issues

    # ── Bilingual checks ───────────────────────────────────────

    def _check_bilingual(
        self,
        cues: dict[str, list[SubtitleCue]],
        config: QCConfig,
    ) -> list[QCIssue]:
        """Compare source↔target for information loss and quality issues."""
        from ..base import pair_bilingual

        issues: list[QCIssue] = []

        for si, sc, ti, tc in pair_bilingual(cues, config.source_lang):
            source_lower = sc.text.lower()
            target_lower = tc.text.lower()

            # 1. Number/fact loss.
            issues.extend(self._check_number_loss(sc, tc, si, source_lower, target_lower))

            # 2. Negation loss.
            issues.extend(self._check_negation_loss(sc, tc, ti, source_lower, target_lower))

            # 3. Proper noun over-generalization (en→zh).
            if config.target_lang == "zh":
                issues.extend(self._check_over_generalization(sc, tc, ti))

        return issues

    def _check_number_loss(
        self,
        sc: SubtitleCue,
        tc: SubtitleCue,
        si: int,
        source_lower: str,
        target_lower: str,
    ) -> list[QCIssue]:
        """Detect when a source number is absent from the translation."""
        issues: list[QCIssue] = []

        numbers = self._SOURCE_NUMBER_RE.findall(source_lower)
        for match_pair in numbers:
            num_text = match_pair[0] or match_pair[1]  # group 0 or group 1
            num_text = num_text.strip()
            if not num_text:
                continue

            # Check if number appears in target.
            if num_text.lower() not in target_lower:
                # Fuzzy check: just the digits part.
                digits = re.sub(r"[^0-9]", "", num_text)
                if digits and digits not in re.sub(r"[^0-9]", "", target_lower):
                    issues.append(
                        QCIssue(
                            severity="warning",
                            category="柔性策略",
                            rule=self.name,
                            cue_id=si + 1,
                            time=seconds_to_srt(sc.start),
                            detail=f"数字信息可能丢失: 原文包含 '{num_text}'，译文中未找到对应数字",
                            fix=f"确保将 '{num_text}' 译入目标语言",
                        )
                    )
                    break  # one number issue per pair

        return issues

    def _check_negation_loss(
        self,
        sc: SubtitleCue,
        tc: SubtitleCue,
        ti: int,
        source_lower: str,
        target_lower: str,
    ) -> list[QCIssue]:
        """Detect when source negation is absent from translation."""
        issues: list[QCIssue] = []

        source_words = set(source_lower.split())
        negations = source_words & self._NEGATION_WORDS

        if negations:
            # Chinese negation check.
            zh_negations = {"不", "没", "别", "未", "无", "非", "勿", "否"}
            if any(w in target_lower for w in zh_negations):
                return []  # negation found in target

            # English→English (unlikely but handle).
            if any(w in target_lower for w in self._NEGATION_WORDS):
                return []

            # Negation missing — issue.
            neg_list = ", ".join(sorted(negations)[:3])
            issues.append(
                QCIssue(
                    severity="warning",
                    category="柔性策略",
                    rule=self.name,
                    cue_id=ti + 1,
                    time=seconds_to_srt(tc.start),
                    detail=f"否定词可能丢失: 原文包含 '{neg_list}'，译文中未检测到否定表达",
                    fix="检查翻译是否遗漏了否定语义",
                )
            )

        return issues

    def _check_over_generalization(
        self,
        sc: SubtitleCue,
        tc: SubtitleCue,
        ti: int,
    ) -> list[QCIssue]:
        """Detect when a specific proper noun is translated generically.

        English source has a proper noun (capitalized, multi-word),
        but Chinese translation is a generic description instead of
        preserving the original name.
        """
        issues: list[QCIssue] = []

        # Find candidate proper nouns in source:
        # Capitalized multi-word sequences that aren't sentence-start.
        source_text = sc.text.strip()
        words = source_text.split()
        proper_candidates: list[str] = []

        for i, w in enumerate(words):
            # Skip first word of sentence (often capitalized).
            if i == 0:
                continue
            # Word is capitalized and not ALLCAPS (acronyms are OK to keep as-is).
            if w[0].isupper() and not w.isupper() and len(w) > 1:
                # Check preceding word to confirm it's not after a period.
                if i > 0 and words[i - 1].endswith("."):
                    continue  # likely new sentence
                # Build multi-word if next word is also capitalized.
                seq = [w]
                j = i + 1
                while j < len(words) and words[j][0].isupper() and len(words[j]) > 1:
                    seq.append(words[j])
                    j += 1
                proper_candidates.append(" ".join(seq))

        # For each proper noun, check if it appears in translation.
        # If not, translation may have over-generalized.
        target_text = tc.text.replace("\n", " ").strip()

        for pn in proper_candidates[:2]:  # limit to avoid noise
            pn_lower = pn.lower()
            # Skip if source proper noun appears verbatim in translation.
            if pn_lower in target_text.lower():
                continue
            # Skip short single-word abbreviations.
            if len(pn) <= 4 and pn.isupper():
                continue

            issues.append(
                QCIssue(
                    severity="suggestion",
                    category="柔性策略",
                    rule=self.name,
                    cue_id=ti + 1,
                    time=seconds_to_srt(tc.start),
                    detail=f"专有名词 '{pn}' 在译文中未保留，可能被泛化翻译",
                    fix=f"建议保留 '{pn}' 原词或使用一致的中文译名",
                )
            )

        return issues
