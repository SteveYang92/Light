"""Judge protocol — one judge scores one case's step output per problem type.

Judges receive the case, its loaded fixture (step inputs), and the
``StepOutput`` produced by the runner, and return one ``ProblemTypeStats``
per problem type.  Rule judges compute deterministic metrics; LLM judges
detect semantic defects and classify them by problem type.
"""

from __future__ import annotations

from typing import Protocol

from ..loader import Fixture
from ..models import EvalCase, ProblemTypeStats, StepOutput


class Judge(Protocol):
    """Inspects a step output and reports per-problem-type statistics."""

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[ProblemTypeStats]: ...
