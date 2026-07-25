"""Judge protocol — one judge scores one case's step output per dimension.

Judges receive the case, its loaded fixture (step inputs), and the
``StepOutput`` produced by the runner, and return one ``DimensionScore``
per dimension.  Rule judges compute deterministic metrics; LLM judges
(later milestone) will emit 1-5 scores calibrated against annotations.
"""

from __future__ import annotations

from typing import Protocol

from ..loader import Fixture
from ..models import DimensionScore, EvalCase, StepOutput


class Judge(Protocol):
    """Scores a step output along named dimensions."""

    def score(self, case: EvalCase, fixture: Fixture, output: StepOutput) -> list[DimensionScore]: ...
