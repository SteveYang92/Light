from dataclasses import dataclass, field


@dataclass
class QCIssue:
    severity: str
    category: str
    rule: str
    cue_id: int | None
    time: str | None
    detail: str
    fix: str


@dataclass
class QCReport:
    total_cues: int
    errors: int
    warnings: int
    suggestions: int
    passed: bool
    bilingual: bool
    source_lang: str
    target_lang: str | None
    issues: list[QCIssue] = field(default_factory=list)
