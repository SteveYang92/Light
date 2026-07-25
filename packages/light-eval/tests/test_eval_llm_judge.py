"""LLM-judge tests — fake chat client returning canned JSON responses."""

from __future__ import annotations

import json

from light_eval.judges.llm import BATCH_SIZE, LLMJudge
from light_eval.loader import Fixture
from light_eval.models import EvalCase, StepOutput
from light_models import Segment


class FakeClient:
    """Stand-in for OpenAIClient: pops canned responses, records calls."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], temperature: float = 0.3) -> tuple[str, dict]:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeClient ran out of canned responses")
        return self._responses.pop(0), {}


def _issue(unit_id: str, problem: str) -> dict:
    return {"unit_id": unit_id, "problem": problem}


def _verdict(dimensions: dict[str, tuple[int, str, list[dict]]]) -> str:
    """Build a judge response body: dimension → (score, summary, issues)."""
    body = {
        dim: {"score": score, "summary": summary, "issues": issues}
        for dim, (score, summary, issues) in dimensions.items()
    }
    return json.dumps(body)


def _case(step: str, params: dict | None = None) -> EvalCase:
    return EvalCase(name="c", step=step, kind="control", source="test", params=params or {"target_lang": "zh"})


def _fixture(n_units: int = 2) -> Fixture:
    segments = [
        Segment(unit_id=f"p{i:04d}", start=i * 2.0, end=(i + 1) * 2.0, speaker="", source_text=f"source {i}", words=[])
        for i in range(n_units)
    ]
    return Fixture(segments=segments)


def _plan_output(n_units: int = 2) -> StepOutput:
    units = [
        {
            "unit_id": f"p{i:04d}",
            "start": i * 2.0,
            "end": (i + 1) * 2.0,
            "text": f"source {i}",
            "speaker": "",
            "words": [],
        }
        for i in range(n_units)
    ]
    return StepOutput(case="c", output=units)


def _translate_output(n_cues: int = 2) -> StepOutput:
    cues = [
        {"cue_id": f"zh_{i:04d}", "unit_id": f"p{i:04d}", "start": i * 2.0, "end": (i + 1) * 2.0, "text": f"译文{i}"}
        for i in range(n_cues)
    ]
    return StepOutput(case="c", output=cues)


def _scores_by_dim(scores):
    return {s.dimension: s for s in scores}


# ── Scoring & parsing ───────────────────────────────────────────────────────


def test_plan_dimensions_parsed() -> None:
    client = FakeClient(
        [
            _verdict(
                {
                    "boundary_quality": (5, "clean boundaries", []),
                    "split_necessity": (3, "tail is over-fragmented", [_issue("p0001", "over-fragmented")]),
                }
            )
        ]
    )
    scores = LLMJudge(client).score(_case("plan", {}), _fixture(), _plan_output())

    by_dim = _scores_by_dim(scores)
    assert set(by_dim) == {"boundary_quality", "split_necessity"}
    assert by_dim["boundary_quality"].score == 5.0
    assert by_dim["boundary_quality"].passed
    assert by_dim["split_necessity"].score == 3.0
    assert not by_dim["split_necessity"].passed
    assert by_dim["split_necessity"].detail == "tail is over-fragmented"
    assert by_dim["split_necessity"].evidence == ["p0001"]  # deduped unit_ids from issues
    assert by_dim["split_necessity"].issues == [_issue("p0001", "over-fragmented")]


def test_translate_dimensions_parsed_and_glossary_rendered() -> None:
    fixture = _fixture()
    fixture.glossary = {"LLM": "大模型"}
    client = FakeClient(
        [
            _verdict(
                {
                    "faithfulness": (4, "minor compression", [_issue("p0000", "drops a detail")]),
                    "naturalness": (5, "natural", []),
                    "unit_integrity": (5, "aligned", []),
                    "terminology": (2, "glossary term ignored", [_issue("p0001", "glossary term ignored")]),
                }
            )
        ]
    )
    scores = LLMJudge(client).score(_case("translate"), fixture, _translate_output())

    by_dim = _scores_by_dim(scores)
    assert set(by_dim) == {"faithfulness", "naturalness", "unit_integrity", "terminology"}
    assert by_dim["faithfulness"].passed and by_dim["faithfulness"].score == 4.0
    assert not by_dim["terminology"].passed
    prompt = client.calls[0][0]["content"]
    assert "LLM → 大模型" in prompt  # glossary rendered into the rubric


def test_legacy_reason_and_evidence_schema_still_parses() -> None:
    """Old rubric responses (reason + evidence, no summary/issues) fall back cleanly."""
    body = json.dumps(
        {
            "boundary_quality": {"score": 4, "reason": "mostly clean", "evidence": ["p0000"]},
            "split_necessity": {"score": 5, "reason": "fine", "evidence": []},
        }
    )
    by_dim = _scores_by_dim(LLMJudge(FakeClient([body])).score(_case("plan", {}), _fixture(), _plan_output()))
    assert by_dim["boundary_quality"].detail == "mostly clean"
    assert by_dim["boundary_quality"].evidence == ["p0000"]
    assert by_dim["boundary_quality"].issues == []


def test_malformed_issue_entries_are_dropped() -> None:
    body = json.dumps(
        {
            "boundary_quality": {
                "score": 3,
                "summary": "mixed bag",
                "issues": [
                    {"unit_id": "p0000", "problem": "mid-clause cut"},
                    {"unit_id": "p0001"},  # missing problem
                    {"unit_id": 42, "problem": "non-str uid"},
                    "not a dict",
                ],
            }
        }
    )
    by_dim = _scores_by_dim(LLMJudge(FakeClient([body])).score(_case("plan", {}), _fixture(), _plan_output()))
    assert by_dim["boundary_quality"].issues == [_issue("p0000", "mid-clause cut")]
    assert by_dim["boundary_quality"].evidence == ["p0000"]


def test_markdown_wrapped_json_is_extracted() -> None:
    body = (
        "Here is my review:\n```json\n"
        + _verdict(
            {
                "boundary_quality": (4, "ok", []),
                "split_necessity": (5, "fine", []),
            }
        )
        + "\n```"
    )
    scores = LLMJudge(FakeClient([body])).score(_case("plan", {}), _fixture(), _plan_output())
    assert _scores_by_dim(scores)["boundary_quality"].score == 4.0


def test_scores_are_clamped_to_rubric_range() -> None:
    verdict = _verdict(
        {"boundary_quality": (9, "hyped", []), "split_necessity": (0, "harsh", [_issue("p0000", "too long")])}
    )
    by_dim = _scores_by_dim(LLMJudge(FakeClient([verdict])).score(_case("plan", {}), _fixture(), _plan_output()))
    assert by_dim["boundary_quality"].score == 5.0
    assert by_dim["split_necessity"].score == 1.0


# ── Retry ───────────────────────────────────────────────────────────────────


def test_invalid_json_retried_once() -> None:
    good = _verdict({"boundary_quality": (5, "ok", []), "split_necessity": (5, "ok", [])})
    client = FakeClient(["not json at all", good])
    scores = LLMJudge(client).score(_case("plan", {}), _fixture(), _plan_output())
    assert len(client.calls) == 2  # exactly one retry
    assert len(scores) == 2


def test_persistent_invalid_json_yields_no_scores() -> None:
    client = FakeClient(["garbage", "still garbage"])
    scores = LLMJudge(client).score(_case("plan", {}), _fixture(), _plan_output())
    assert len(client.calls) == 2  # gives up after the single retry
    assert scores == []


def test_partial_dimensions_keep_valid_ones() -> None:
    body = json.dumps({"boundary_quality": {"score": 4, "reason": "ok"}, "split_necessity": {"reason": "no score"}})
    scores = LLMJudge(FakeClient([body])).score(_case("plan", {}), _fixture(), _plan_output())
    assert _scores_by_dim(scores)["boundary_quality"].score == 4.0
    assert len(scores) == 1


# ── Batching ────────────────────────────────────────────────────────────────


def test_large_output_is_batched_and_min_score_wins() -> None:
    n = BATCH_SIZE + 10
    batch1 = _verdict({"boundary_quality": (5, "batch1 clean", []), "split_necessity": (5, "batch1 ok", [])})
    batch2 = _verdict(
        {
            "boundary_quality": (2, "batch2 mid-clause cut", [_issue("p0055", "mid-clause cut")]),
            "split_necessity": (4, "batch2 slightly long", [_issue("p0050", "slightly long")]),
        }
    )
    client = FakeClient([batch1, batch2])
    scores = LLMJudge(client).score(_case("plan", {}), _fixture(n), _plan_output(n))

    assert len(client.calls) == 2  # 60 items → 2 batches
    by_dim = _scores_by_dim(scores)
    assert by_dim["boundary_quality"].score == 2.0  # min across batches (conservative)
    assert not by_dim["boundary_quality"].passed
    assert by_dim["boundary_quality"].detail == "batch2 mid-clause cut"  # summary of the worst batch, not joined
    assert by_dim["boundary_quality"].evidence == ["p0055"]
    assert by_dim["boundary_quality"].issues == [_issue("p0055", "mid-clause cut")]
    assert by_dim["split_necessity"].score == 4.0


def test_merge_dedupes_issues_across_batches() -> None:
    n = BATCH_SIZE + 10
    shared = _issue("p0003", "mid-clause cut")
    batch1 = _verdict({"boundary_quality": (3, "batch1 has issues", [shared, _issue("p0001", "orphan verb")])})
    batch2 = _verdict({"boundary_quality": (4, "batch2 mostly fine", [shared])})  # same issue reported again
    client = FakeClient([batch1, batch2])
    by_dim = _scores_by_dim(LLMJudge(client).score(_case("plan", {}), _fixture(n), _plan_output(n)))

    assert by_dim["boundary_quality"].score == 3.0
    assert by_dim["boundary_quality"].detail == "batch1 has issues"  # lowest-scoring batch's summary
    assert by_dim["boundary_quality"].issues == [shared, _issue("p0001", "orphan verb")]  # deduped, order kept
    assert by_dim["boundary_quality"].evidence == ["p0003", "p0001"]


def test_failed_batch_does_not_sink_others() -> None:
    n = BATCH_SIZE + 1
    good = _verdict({"boundary_quality": (5, "ok", []), "split_necessity": (5, "ok", [])})
    client = FakeClient(["garbage", "garbage2", good])
    scores = LLMJudge(client).score(_case("plan", {}), _fixture(n), _plan_output(n))
    assert len(client.calls) == 3  # batch1 (2 attempts) + batch2 (1 attempt)
    assert _scores_by_dim(scores)["boundary_quality"].score == 5.0


# ── Skip conditions ─────────────────────────────────────────────────────────


def test_none_client_returns_empty() -> None:
    assert LLMJudge(None).score(_case("plan", {}), _fixture(), _plan_output()) == []


def test_skipped_or_errored_output_returns_empty() -> None:
    judge = LLMJudge(FakeClient([]))
    assert judge.score(_case("plan", {}), _fixture(), StepOutput(case="c", skipped=True)) == []
    assert judge.score(_case("plan", {}), _fixture(), StepOutput(case="c", error="boom")) == []
    assert judge.score(_case("plan", {}), _fixture(), StepOutput(case="c", output=[])) == []
