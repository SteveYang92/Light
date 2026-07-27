"""LLM-judge tests — fake chat client returning canned JSON responses.

After the rewrite the judge returns a flat ``issues`` array with
``{unit_id, problem_type, note}`` entries, aggregated into per-type
``ProblemTypeStats``.  Scores and dimension-level summaries are gone.
"""

from __future__ import annotations

import json

from light_eval.judges.llm import BATCH_SIZE, LLMJudge
from light_eval.loader import Fixture
from light_eval.models import EvalCase, ProblemTypeStats, StepOutput
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


def _issue(unit_id: str, problem_type: str, note: str = "") -> dict:
    return {"unit_id": unit_id, "problem_type": problem_type, "note": note}


def _verdict(issues: list[dict]) -> str:
    """Build a judge response body: flat ``{"issues": [...]}``."""
    return json.dumps({"issues": issues})


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


def _stats_by_type(stats: list[ProblemTypeStats]) -> dict[str, ProblemTypeStats]:
    return {s.problem_type: s for s in stats}


# ── Parsing & problem types ────────────────────────────────────────────────


def test_plan_problem_types_parsed() -> None:
    client = FakeClient(
        [
            _verdict(
                [
                    _issue("p0001", "over_fragmentation", "tail is over-fragmented"),
                ]
            )
        ]
    )
    stats = LLMJudge(client).score(_case("plan", {}), _fixture(), _plan_output())

    by_type = _stats_by_type(stats)
    expected = {
        "semantic_boundary",
        "over_fragmentation",
        "over_long_unit",
        "dangling_word",
        "empty_unit",
        "flash_unit",
    }
    assert set(by_type) == expected
    assert by_type["semantic_boundary"].passed
    assert by_type["semantic_boundary"].error_count == 0
    assert by_type["over_fragmentation"].warning_count == 1  # warning severity
    assert by_type["over_fragmentation"].passed  # no error-severity items
    assert by_type["over_fragmentation"].evidence == ["p0001"]
    assert by_type["over_fragmentation"].issues == [_issue("p0001", "over_fragmentation", "tail is over-fragmented")]


def test_translate_problem_types_parsed_and_glossary_rendered() -> None:
    fixture = _fixture()
    fixture.glossary = {"LLM": "大模型"}
    client = FakeClient(
        [
            _verdict(
                [
                    _issue("p0000", "translation_ese", "minor compression"),
                    _issue("p0001", "missing_content", "glossary term ignored"),
                ]
            )
        ]
    )
    stats = LLMJudge(client).score(_case("translate"), fixture, _translate_output())

    by_type = _stats_by_type(stats)
    expected = {
        "missing_content",
        "extra_content",
        "semantic_drift",
        "translation_ese",
        "unit_mismatch",
        "terminology_inconsistent",
        "word_choice",
        "bad_line_break",
    }
    assert set(by_type) == expected
    assert by_type["translation_ese"].passed  # warning severity
    assert by_type["translation_ese"].warning_count == 1
    assert not by_type["missing_content"].passed  # error severity
    assert by_type["missing_content"].error_count == 1
    prompt = client.calls[0][0]["content"]
    assert "LLM → 大模型" in prompt


def test_missing_or_invalid_problem_type_is_dropped() -> None:
    """Issues without a valid problem_type are filtered out by the parser."""
    body = json.dumps(
        {
            "issues": [
                {"unit_id": "p0000", "problem_type": "over_fragmentation", "note": "valid"},
                {"unit_id": "p0001", "note": "no problem_type"},
                {"unit_id": "p0002", "problem_type": "unknown_type", "note": "bad type"},
            ]
        }
    )
    stats = LLMJudge(FakeClient([body])).score(_case("plan", {}), _fixture(3), _plan_output(3))
    by_type = _stats_by_type(stats)
    assert by_type["over_fragmentation"].issues == [_issue("p0000", "over_fragmentation", "valid")]
    assert by_type["over_fragmentation"].warning_count == 1


def test_all_warning_issues_pass_the_gate() -> None:
    verdict = _verdict(
        [
            _issue("p0000", "over_fragmentation", "nitpick one"),
            _issue("p0001", "over_long_unit", "nitpick two"),
        ]
    )
    by_type = _stats_by_type(LLMJudge(FakeClient([verdict])).score(_case("plan", {}), _fixture(), _plan_output()))
    assert by_type["over_fragmentation"].passed
    assert by_type["over_long_unit"].passed


def test_suggest_overall_three_states() -> None:
    from light_eval.judges.llm import suggest_overall

    def _ps(problem_type: str, error_count: int = 0, warning_count: int = 0) -> ProblemTypeStats:
        return ProblemTypeStats(problem_type=problem_type, error_count=error_count, warning_count=warning_count)

    assert suggest_overall([_ps("a"), _ps("b")]) == "pass"
    assert suggest_overall([_ps("a", warning_count=1)]) == "borderline"
    assert suggest_overall([_ps("a", error_count=1)]) == "fail"


def test_malformed_issue_entries_are_dropped() -> None:
    body = json.dumps(
        {
            "issues": [
                {"unit_id": "p0000", "problem_type": "dangling_word", "note": "mid-clause cut"},
                {"unit_id": "p0001"},  # missing problem_type
                {"unit_id": 42, "problem_type": "dangling_word", "note": "non-str uid"},
                "not a dict",
            ]
        }
    )
    stats = LLMJudge(FakeClient([body])).score(_case("plan", {}), _fixture(), _plan_output())
    by_type = _stats_by_type(stats)
    assert by_type["dangling_word"].issues == [_issue("p0000", "dangling_word", "mid-clause cut")]
    assert by_type["dangling_word"].evidence == ["p0000"]
    assert by_type["dangling_word"].error_count == 1  # dangling_word is error severity


def test_markdown_wrapped_json_is_extracted() -> None:
    body = "Here is my review:\n```json\n" + _verdict([_issue("p0000", "dangling_word", "problem found")]) + "\n```"
    stats = LLMJudge(FakeClient([body])).score(_case("plan", {}), _fixture(), _plan_output())
    assert _stats_by_type(stats)["dangling_word"].error_count == 1


# ── Post-filter ─────────────────────────────────────────────────────────────


def test_out_of_batch_unit_id_is_dropped() -> None:
    """Issues referencing ids outside the judged batch are hallucinations."""
    verdict = _verdict(
        [
            _issue("p9999", "dangling_word", "ghost unit"),
            _issue("p0001", "dangling_word", "real issue"),
        ]
    )
    stats = LLMJudge(FakeClient([verdict])).score(_case("plan", {}), _fixture(), _plan_output())
    by_type = _stats_by_type(stats)
    assert by_type["dangling_word"].issues == [_issue("p0001", "dangling_word", "real issue")]
    assert by_type["dangling_word"].evidence == ["p0001"]


def test_translate_issues_accept_cue_and_unit_ids() -> None:
    verdict = _verdict(
        [
            _issue("zh_0000", "translation_ese", "cue-level slip"),
            _issue("p0001", "translation_ese", "unit-level slip"),
        ]
    )
    stats = LLMJudge(FakeClient([verdict])).score(_case("translate"), _fixture(), _translate_output())
    by_type = _stats_by_type(stats)
    assert by_type["translation_ese"].issues == [  # unit-sorted: p… < zh…
        _issue("p0001", "translation_ese", "unit-level slip"),
        _issue("zh_0000", "translation_ese", "cue-level slip"),
    ]


def test_infeasible_merge_suggestion_is_dropped() -> None:
    """Merging p0000+p0001 would run 12s — over the 7.0×1.15 soft cap, so the suggestion is noise."""
    output = StepOutput(
        case="c",
        output=[
            {"unit_id": "p0000", "start": 0.0, "end": 5.0, "text": "a", "speaker": "", "words": []},
            {"unit_id": "p0001", "start": 5.0, "end": 12.0, "text": "b", "speaker": "", "words": []},
        ],
    )
    verdict = _verdict(
        [
            _issue("p0000", "semantic_boundary", "建议合并 p0000 和 p0001，语义更完整"),
        ]
    )
    stats = LLMJudge(FakeClient([verdict])).score(_case("plan", {}), _fixture(), output)
    by_type = _stats_by_type(stats)
    assert by_type["semantic_boundary"].issues == []
    assert by_type["semantic_boundary"].evidence == []


def test_feasible_merge_suggestion_is_kept() -> None:
    """Default plan output merges to 4.0s — within the soft cap, so the issue survives."""
    verdict = _verdict(
        [
            _issue("p0000", "over_fragmentation", "建议合并 p0000 和 p0001"),
        ]
    )
    stats = LLMJudge(FakeClient([verdict])).score(_case("plan", {}), _fixture(), _plan_output())
    by_type = _stats_by_type(stats)
    assert by_type["over_fragmentation"].issues == [_issue("p0000", "over_fragmentation", "建议合并 p0000 和 p0001")]


def test_unverifiable_merge_suggestion_is_kept() -> None:
    """A merge hint naming no second unit cannot be re-checked — kept as-is."""
    verdict = _verdict(
        [
            _issue("p0000", "over_fragmentation", "这两个单元应合并"),
        ]
    )
    stats = LLMJudge(FakeClient([verdict])).score(_case("plan", {}), _fixture(), _plan_output())
    by_type = _stats_by_type(stats)
    assert by_type["over_fragmentation"].issues == [_issue("p0000", "over_fragmentation", "这两个单元应合并")]


# ── Prompt rendering ────────────────────────────────────────────────────────


def test_speaker_is_rendered_into_plan_prompt() -> None:
    fixture = Fixture(
        segments=[
            Segment(unit_id="p0000", start=0.0, end=2.0, speaker="A", source_text="source 0", words=[]),
            Segment(unit_id="p0001", start=2.0, end=4.0, speaker="B", source_text="source 1", words=[]),
        ]
    )
    output = StepOutput(
        case="c",
        output=[
            {"unit_id": "p0000", "start": 0.0, "end": 2.0, "text": "source 0", "speaker": "A", "words": []},
            {"unit_id": "p0001", "start": 2.0, "end": 4.0, "text": "source 1", "speaker": "B", "words": []},
        ],
    )
    verdict = _verdict([_issue("p0000", "dangling_word", "x")])
    client = FakeClient([verdict])
    LLMJudge(client).score(_case("plan", {}), fixture, output)
    prompt = client.calls[0][0]["content"]
    assert "[A] source 0" in prompt
    assert "[B] source 1" in prompt


def test_plan_prompt_only_shows_source_near_the_batch() -> None:
    fixture = _fixture(6)  # segments p0000..p0005, 2s each
    verdict = _verdict([_issue("p0000", "dangling_word", "x")])
    client = FakeClient([verdict])
    LLMJudge(client).score(_case("plan", {}), fixture, _plan_output(2))  # batch covers 0-4s
    prompt = client.calls[0][0]["content"]
    assert "source 0" in prompt
    assert "source 5" not in prompt  # far outside the batch window (+1 neighbor buffer)


def test_translate_prompt_only_shows_referenced_source_units() -> None:
    fixture = _fixture(3)
    output = StepOutput(
        case="c",
        output=[{"cue_id": "zh_0000", "unit_id": "p0000", "start": 0.0, "end": 2.0, "text": "译文0"}],
    )
    verdict = _verdict([_issue("p0000", "missing_content", "x")])
    client = FakeClient([verdict])
    LLMJudge(client).score(_case("translate"), fixture, output)
    prompt = client.calls[0][0]["content"]
    assert "source 0" in prompt
    assert "source 1" not in prompt  # unreferenced units stay out of the judge's context


def test_translate_prompt_renders_content_summary() -> None:
    fixture = _fixture()
    fixture.summary = {"overview": "A talk about evals", "key_topics": ["evals"], "speakers": {"spk1": "host"}}
    verdict = _verdict([_issue("p0000", "missing_content", "x")])
    client = FakeClient([verdict])
    LLMJudge(client).score(_case("translate"), fixture, _translate_output())
    prompt = client.calls[0][0]["content"]
    assert "A talk about evals" in prompt
    assert "spk1 (host)" in prompt


# ── Retry ───────────────────────────────────────────────────────────────────


def test_invalid_json_retried_once() -> None:
    good = _verdict([_issue("p0000", "dangling_word", "ok")])
    client = FakeClient(["not json at all", good])
    stats = LLMJudge(client).score(_case("plan", {}), _fixture(), _plan_output())
    assert len(client.calls) == 2  # exactly one retry
    assert _stats_by_type(stats)["dangling_word"].error_count == 1


def test_persistent_invalid_json_yields_no_issues() -> None:
    client = FakeClient(["garbage", "still garbage"])
    stats = LLMJudge(client).score(_case("plan", {}), _fixture(), _plan_output())
    assert len(client.calls) == 2  # gives up after the single retry
    # all problem types have zero issues after the batch was skipped
    for s in stats:
        assert s.error_count == 0
        assert s.warning_count == 0


# ── Batching ────────────────────────────────────────────────────────────────


def test_large_output_is_batched_and_issues_merged() -> None:
    n = BATCH_SIZE + 10
    batch1 = _verdict(
        [
            _issue("p0003", "dangling_word", "batch1 mid-clause cut"),
        ]
    )
    batch2 = _verdict(
        [
            _issue("p0055", "dangling_word", "batch2 mid-clause cut"),
            _issue("p0050", "over_long_unit", "slightly long"),
        ]
    )
    client = FakeClient([batch1, batch2])
    stats = LLMJudge(client).score(_case("plan", {}), _fixture(n), _plan_output(n))

    assert len(client.calls) == 2  # 60 items → 2 batches
    by_type = _stats_by_type(stats)
    assert by_type["dangling_word"].error_count == 2
    assert not by_type["dangling_word"].passed
    assert by_type["dangling_word"].evidence == ["p0003", "p0055"]
    assert by_type["over_long_unit"].warning_count == 1
    assert by_type["over_long_unit"].passed


def test_merge_dedupes_issues_across_batches() -> None:
    n = BATCH_SIZE + 10
    shared = _issue("p0003", "dangling_word", "mid-clause cut")
    batch1 = _verdict([shared, _issue("p0001", "dangling_word", "orphan verb")])
    batch2 = _verdict([shared])  # out-of-batch re-report → filtered
    client = FakeClient([batch1, batch2])
    stats = LLMJudge(client).score(_case("plan", {}), _fixture(n), _plan_output(n))
    by_type = _stats_by_type(stats)

    assert by_type["dangling_word"].error_count == 2
    assert by_type["dangling_word"].issues == [
        _issue("p0001", "dangling_word", "orphan verb"),
        shared,
    ]  # deduped, unit order
    assert by_type["dangling_word"].evidence == ["p0001", "p0003"]


def test_issues_are_sorted_by_unit_id() -> None:
    """The model emits issues in arbitrary order; merged output is unit-sorted."""
    verdict = _verdict(
        [
            _issue("p0010", "dangling_word", "late"),
            _issue("p0002", "dangling_word", "early"),
            _issue("p0001", "dangling_word", "first"),
        ]
    )
    n = 11
    stats = LLMJudge(FakeClient([verdict])).score(_case("plan", {}), _fixture(n), _plan_output(n))
    by_type = _stats_by_type(stats)
    assert [i["unit_id"] for i in by_type["dangling_word"].issues] == ["p0001", "p0002", "p0010"]
    assert by_type["dangling_word"].evidence == ["p0001", "p0002", "p0010"]


def test_failed_batch_does_not_sink_others() -> None:
    n = BATCH_SIZE + 1
    good = _verdict([_issue("p0050", "dangling_word", "ok")])
    client = FakeClient(["garbage", "garbage2", good])
    stats = LLMJudge(client).score(_case("plan", {}), _fixture(n), _plan_output(n))
    assert len(client.calls) == 3  # batch1 (2 attempts) + batch2 (1 attempt)
    assert _stats_by_type(stats)["dangling_word"].error_count == 1


# ── Skip conditions ─────────────────────────────────────────────────────────


def test_none_client_returns_empty() -> None:
    assert LLMJudge(None).score(_case("plan", {}), _fixture(), _plan_output()) == []


def test_skipped_or_errored_output_returns_empty() -> None:
    judge = LLMJudge(FakeClient([]))
    assert judge.score(_case("plan", {}), _fixture(), StepOutput(case="c", skipped=True)) == []
    assert judge.score(_case("plan", {}), _fixture(), StepOutput(case="c", error="boom")) == []
    assert judge.score(_case("plan", {}), _fixture(), StepOutput(case="c", output=[])) == []
