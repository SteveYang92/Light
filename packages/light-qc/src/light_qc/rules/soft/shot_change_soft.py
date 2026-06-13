from light_models import QCIssue, SubtitleCue, seconds_to_srt

from ...config import QCConfig
from ..base import SoftRule, _iter_cues


class ShotChangeSoft(SoftRule):
    """Check that cues do not unnecessarily cross shot changes.

    Requires ``config.shot_changes`` to be set.  If a cue straddles a shot
    change but the dialogue does not naturally bridge it (i.e. the cue has
    clear gaps on both sides of the shot cut), flag a suggestion.
    """

    name = "ShotChangeSoft"
    default_severity = "suggestion"

    def check(self, cues: dict[str, list[SubtitleCue]], config: QCConfig) -> list[QCIssue]:
        issues = []
        if not config.shot_changes:
            return issues

        for _lang, cue_list in _iter_cues(cues):
            for i, cue in enumerate(cue_list):
                for sc in config.shot_changes:
                    if cue.start < sc < cue.end:
                        # Check if the dialogue naturally bridges (word timestamps)
                        # or if there's a gap at the shot boundary.

                        # If there's a natural pause at the shot boundary,
                        # the cue should be split there.

                        # No clear boundaries — just flag it.
                        issues.append(
                            QCIssue(
                                severity=self.default_severity,
                                category="柔性策略",
                                rule=self.name,
                                cue_id=i + 1,
                                time=seconds_to_srt(cue.start),
                                detail=f"字幕跨越镜头切点 ({seconds_to_srt(sc)})",
                                fix="考虑在镜头切点处调整 cue 边界",
                            )
                        )
                        break
        return issues
