from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .word import Word


@dataclass
class SubtitleCue:
    cue_id: str
    unit_id: str
    start: float
    end: float
    text: str
    lang: str
    speaker: str = ""
    qc: dict[str, bool] = field(default_factory=dict)
    words: list["Word"] = field(default_factory=list)
    annotation: str = ""
