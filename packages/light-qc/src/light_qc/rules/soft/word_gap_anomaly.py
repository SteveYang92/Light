import statistics

from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class WordGapAnomaly(SoftRule):
    """Detect abnormally long gaps between consecutive words within a cue.

    Per subtitle.md §3.1: word-level timestamps should be consistent.
    An outlier gap may indicate a missed word, a natural pause where a cue
    split would improve readability, or an alignment error.
    """

    name = "WordGapAnomaly"
    default_severity = "suggestion"
    languages = {"en"}  # word-level gaps not meaningful for CJK

    # Minimum number of words required to compute meaningful statistics.
    MIN_WORDS = 3

    # Absolute minimum gap (seconds) to flag — filter out noise.
    MIN_GAP_ABSOLUTE = 0.20

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                words = cue.words
                if not words or len(words) < self.MIN_WORDS:
                    continue

                # Compute inter-word gaps.
                gaps = []
                for j in range(len(words) - 1):
                    g = words[j + 1].start - words[j].end
                    gaps.append((j, g))

                if len(gaps) < 2:
                    continue

                gap_values = [g for _, g in gaps if g > 0]
                if len(gap_values) < 2:
                    continue

                median_gap = statistics.median(gap_values)

                if median_gap < 0.01:
                    continue  # all gaps nearly zero — nothing to flag

                # Use median-based threshold (robust to outliers).
                # A gap is anomalous if > 5 × median and exceeds the absolute floor.
                threshold = median_gap * 5.0

                for j, g in gaps:
                    if g >= self.MIN_GAP_ABSOLUTE and g > threshold:
                        w1 = words[j]
                        w2 = words[j + 1]
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(w1.end),
                                detail=(
                                    f"词 '{w1.text}' 与 '{w2.text}' 之间间隔 {g:.3f}s，远超中位数 {median_gap:.3f}s"
                                ),
                                fix="检查此处是否为断句点或漏词",
                            )
                        )

        return issues
