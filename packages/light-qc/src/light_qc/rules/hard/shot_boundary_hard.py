from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import HardRule, _iter_cues


class ShotBoundaryHard(HardRule):
    """Hard check: cue must not cross a shot boundary unless the dialogue
    naturally spans it.

    Per subtitle.md §1.3-4 and §3.3: if the dialogue does *not* bridge the
    shot change (i.e. adjacent cues exist on both sides of the cut), the cue
    should be split at the shot boundary.
    """

    name = "ShotBoundaryHard"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        if not config.shot_changes:
            return []

        issues = []
        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                for sc in config.shot_changes:
                    if not (cue.start < sc < cue.end):
                        continue

                    # Determine if dialogue naturally bridges the cut.
                    # If we have words, check distribution around the cut.
                    naturally_bridges = False
                    if cue.words:
                        words_before = [w for w in cue.words if w.end <= sc]
                        words_after = [w for w in cue.words if w.start >= sc]
                        # Dialogue naturally bridges if both sides have words
                        # and the gap across the cut is small (< 0.3s).
                        if words_before and words_after:
                            gap_across = words_after[0].start - words_before[-1].end
                            if gap_across < 0.3:
                                naturally_bridges = True

                    if naturally_bridges:
                        continue

                    # Check if adjacent cues would make a natural split point.
                    has_prev = i > 0 and cue_list[i - 1].end <= sc
                    has_next = i + 1 < len(cue_list) and cue_list[i + 1].start >= sc

                    if has_prev or has_next:
                        issues.append(
                            QCIssue(
                                severity="error",
                                category="硬性规则",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=(f"字幕跨越镜头切点 {seconds_to_srt(sc)}，台词在该处有自然断点"),
                                fix=f"在镜头切点处切断 cue：前条 end={seconds_to_srt(sc)}",
                            )
                        )
                        break  # one issue per cue per rule

        return issues
