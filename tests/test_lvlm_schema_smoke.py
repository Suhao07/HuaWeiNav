"""Tests for the production-schema LVLM deployment receipt."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deployment.lvlm_server.schema_smoke import (
    _check_final_guard,
    build_parser,
    build_schema_cases,
    run_schema_cases,
)
from prompting.schemas import (
    GroundingResult,
    ParsedInstruction,
    ParsedRelationResult,
    ParsedTarget,
    ParsedVerification,
)


class _ScriptedStructuredClient:
    """Return schema-specific valid responses without network access."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=self.parse))
        )

    def parse(self, **kwargs):
        """Build one valid response for the requested production schema."""

        self.calls.append(kwargs)
        schema = kwargs["response_format"]
        if schema is ParsedInstruction:
            parsed = ParsedInstruction(targets=[ParsedTarget(name="book", terminal=True)])
        elif schema is GroundingResult:
            parsed = GroundingResult(detector_terms=["tv_monitor"])
        elif schema.__name__ == "CandidateConceptJudgments":
            judgment = dict(matches_concept=False, confidence=0.5,
                            terminal_eligible=False, reason="Different role.")
            parsed = schema(anchor_1=judgment, distractor_1=judgment)
        elif schema is ParsedRelationResult:
            unknown = {"verdict": "unknown", "reason": "No bound instance evidence."}
            parsed = ParsedRelationResult(subject_identity=unknown, reference_identity=unknown,
                                          relation=unknown, confidence=0.5)
        elif schema is ParsedVerification:
            parsed = ParsedVerification(
                satisfied=False,
                decision="need_better_view",
                original_instruction_review={
                    "coverage_complete": True, "interpretation_consistent": True, "task_satisfied": False,
                    "requirements": [{"quote": "Find a book on a shelf.", "scope": "candidate",
                                      "verdict": "unknown", "reason": "No current candidate-bound image ID.",
                                      "supporting_observation_ids": []}],
                },
            )
        else:
            raise AssertionError(f"unexpected schema: {schema}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content="{}"))]
        )


def test_schema_smoke_runs_all_production_trace_labels() -> None:
    """Verify all five production schema calls produce successful records."""

    client = _ScriptedStructuredClient()
    cases = build_schema_cases("data:image/jpeg;base64,AA==")

    records = run_schema_cases(client, model="strive-qwen2.5-vl-7b", cases=cases)

    assert [record["name"] for record in records] == [
        "instruction_parser",
        "concept_grounding",
        "concept_match_batch",
        "relation_verifier",
        "final_instruction_verifier",
    ]
    assert all(record["success"] for record in records)
    assert [call["trace_label"] for call in client.calls] == [
        "instruction_parser",
        "concept_grounding",
        "concept_match_batch",
        "relation_verifier",
        "final_instruction_verifier",
    ]


def test_schema_smoke_rejects_inconsistent_visual_accept() -> None:
    """Verify the deployment gate detects inconsistent visual predicates."""

    parsed = ParsedVerification(
        satisfied=True,
        semantic_satisfied=False,
        view_sufficient_for_stop=True,
        decision="accept",
    )

    failures = _check_final_guard(parsed)

    assert len(failures) == 2
    assert any("satisfied=true" in failure for failure in failures)
    assert any("decision=accept" in failure for failure in failures)


def test_schema_smoke_exposes_optional_provider_endpoint_skips() -> None:
    """Verify commercial providers can skip non-standard discovery endpoints."""

    args = build_parser().parse_args(
        [
            "--image",
            "frame.jpg",
            "--skip-health",
            "--skip-model-discovery",
        ]
    )

    assert args.skip_health is True
    assert args.skip_model_discovery is True


@pytest.mark.parametrize("predicate_ids", [[], ["capability:1", "capability:2"]])
def test_fulltext_review_is_required_typed_schema_with_conservative_defaults(predicate_ids):
    from prompting.schemas import affordance_verification_response_model

    model = affordance_verification_response_model(predicate_ids)
    schema = model.model_json_schema()
    assert "original_instruction_review" in schema["required"]
    review = schema["$defs"]["OriginalInstructionReview"]
    assert set(review["required"]) == {"coverage_complete", "interpretation_consistent", "task_satisfied", "requirements"}
    assert review["additionalProperties"] is False
    requirement = schema["$defs"]["OriginalInstructionRequirement"]
    assert set(requirement["required"]) == {"quote", "scope", "verdict", "reason", "supporting_observation_ids"}
    assert requirement["additionalProperties"] is False
    assert requirement["properties"]["quote"]["minLength"] == 1
    assert requirement["properties"]["reason"]["minLength"] == 1
    assert requirement["properties"]["scope"]["enum"] == ["candidate", "task"]
    assert requirement["properties"]["verdict"]["enum"] == ["true", "false", "unknown"]
    assert ParsedVerification().original_instruction_review.model_dump() == {
        "coverage_complete": False, "interpretation_consistent": False,
        "task_satisfied": False, "requirements": []}
    if predicate_ids:
        assert "affordance_judgments" in schema["required"]
        assert set(schema["$defs"]["AffordanceJudgments"]["required"]) == set(predicate_ids)
        judgment = {"verdict": "unknown", "reason": "No observation.", "supporting_observation_ids": []}
        for invalid in ({}, {predicate_ids[0]: judgment},
                        {**{uid: judgment for uid in predicate_ids}, "invented": judgment}):
            with pytest.raises(ValidationError):
                model(affordance_judgments=invalid)


@pytest.mark.parametrize("field,value", [("quote", ""), ("quote", 7), ("reason", ""),
                                         ("scope", "history"), ("verdict", True),
                                         ("verdict", "maybe"), ("supporting_observation_ids", [3])])
def test_fulltext_review_rejects_invalid_typed_requirements(field, value):
    requirement = {"quote": "book", "scope": "candidate", "verdict": "unknown",
                   "reason": "No bound view.", "supporting_observation_ids": []}
    requirement[field] = value
    with pytest.raises(ValidationError):
        ParsedVerification(original_instruction_review={"requirements": [requirement]})
