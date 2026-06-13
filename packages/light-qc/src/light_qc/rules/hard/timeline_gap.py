from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class TimelineGap(HardRule):
    """Detect same-speaker adjacent cues with a tiny gap that should be chained.

    Per subtitle.md §3.3: when gap_to_next < min_gap and the speaker is the
    same, the two cues should be chained (end of first = start of second)
    rather than left with a flicker-inducing micro-gap.

    This is complementary to GapFlash: GapFlash covers *all* micro-gaps,
    while TimelineGap specifically flags same-speaker cases where chaining is
    the correct fix.
    """

    name = "TimelineGap"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i in range(len(cue_list) - 1):
                cur, nxt = cue_list[i], cue_list[i + 1]
                gap = nxt.start - cur.end

                # Only flag positive micro-gaps (overlaps are handled by Overlap).
                if gap <= 0 or gap >= config.min_gap:
                    continue

                # Same-speaker (or both unset) → should chain.
                cur_spk = cur.speaker or ""
                nxt_spk = nxt.speaker or ""
                if cur_spk and nxt_spk and cur_spk != nxt_spk:
                    continue

                issues.append(
                    QCIssue(
                        severity="error",
                        category="硬性规则",
                        rule=self.name,
                        cue_id=i + 2,
                        time=seconds_to_srt(cur.end),
                        detail=(f"同说话人字幕间隔仅 {gap:.3f}s（< {config.min_gap}s），应做 chaining"),
                        fix="将前条出点延长至后条入点",
                    )
                )

        return issues
