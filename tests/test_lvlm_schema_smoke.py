"""Tests for the production-schema LVLM deployment receipt."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deployment.lvlm_server.schema_smoke import (
    _check_final_guard,
    _check_instruction,
    _check_observation_guard,
    build_parser,
    build_schema_cases,
    run_schema_cases,
)
from prompting.schemas import (
    GroundingResult,
    ParsedInstruction,
    ParsedRelationResult,
    visual_verification_response_model,
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
            parsed = ParsedInstruction.model_validate({
                "targets": [{"id": "book", "name": "book", "attributes": {"color": "red"}},
                            {"id": "shelf", "name": "shelf", "role": "anchor", "terminal": False}],
                "constraints": [{"id": "on-shelf", "type": "spatial", "subject": "book",
                                 "relation": "on", "object": "shelf", "hardness": "hard"}],
            })
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
        elif schema.__name__ == "InstanceObservation":
            parsed = schema(observations={"schema_smoke_view": {
                "observed_category": "unknown", "visible_parts": [], "support_and_attachment": "",
                "visible_surface_content": "", "visible_context": [], "uncertainties": ["No bound instance."]}},
                reference_comparison={"verdict": "unknown", "reason": "No reference image.",
                                      "supporting_observation_ids": []})
        elif schema.__name__ == "VisualVerification":
            parsed = schema(predicate_judgments={"candidate_identity": {
                "verdict": "unknown", "reason": "Unbound image.", "supporting_observation_ids": []}})
        else:
            raise AssertionError(f"unexpected schema: {schema}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content="{}"))]
        )


def test_schema_smoke_runs_all_production_trace_labels() -> None:
    """Verify every production schema call produces a successful record."""

    client = _ScriptedStructuredClient()
    cases = build_schema_cases("data:image/jpeg;base64,AA==")

    records = run_schema_cases(client, model="strive-qwen2.5-vl-7b", cases=cases)

    assert [record["name"] for record in records] == [
        "instruction_parser",
        "concept_grounding",
        "concept_match_batch",
        "relation_verifier",
        "instance_observer",
        "final_instruction_verifier",
    ]
    assert all(record["success"] for record in records)
    assert [call["trace_label"] for call in client.calls] == [
        "instruction_parser",
        "concept_grounding",
        "concept_match_batch",
        "relation_verifier",
        "instance_observer",
        "final_instruction_verifier",
    ]


@pytest.mark.parametrize("collection,field,value,status,diagnostic", [
    ("constraints", "subject", "", "invalid", "subject: missing target reference"),
    ("constraints", "object", "", "invalid", "object: missing target reference"),
    ("constraints", "subject", "unknown", "invalid", "subject: unbound target reference"),
    ("constraints", "object", "unknown", "invalid", "object: unbound target reference"),
    ("targets", "min_count", 0, "invalid", "min_count must be positive"),
    ("targets", "min_count", -1, "invalid", "min_count must be positive"),
    ("targets", "name", "", "invalid", "name is empty"),
    ("targets", "id", "shelf", "invalid", "Target IDs must be unique"),
    ("constraints", "type", "teleport", "unsupported", "on-shelf:teleport"),
])
def test_schema_smoke_rejects_invalid_semantics(collection, field, value, status, diagnostic):
    parsed = _ScriptedStructuredClient().parse(response_format=ParsedInstruction).choices[0].message.parsed
    setattr(getattr(parsed, collection)[0], field, value)
    failures = _check_instruction(parsed)
    assert failures and all(error.startswith(f"{status}: ") for error in failures)
    assert diagnostic in failures[0]


def test_schema_smoke_keeps_observer_visual_and_final_text_only() -> None:
    cases = {case.name: case for case in build_schema_cases("data:image/jpeg;base64,AA==")}
    assert any(item["type"] == "image_url" for item in cases["instance_observer"].user_content)
    assert all(item["type"] == "text" for item in cases["final_instruction_verifier"].user_content)


@pytest.mark.parametrize("field,value", [
    ("observed_category", "chair"), ("visible_parts", ["seat"]),
    ("support_and_attachment", "On legs"), ("visible_surface_content", "Flowers"),
    ("visible_context", ["A sink beside the selected chair"]), ("uncertainties", []),
])
def test_schema_smoke_rejects_unbound_observation_claim(field, value) -> None:
    case = next(case for case in build_schema_cases("data:image/jpeg;base64,AA==")
                if case.name == "instance_observer")
    parsed = _ScriptedStructuredClient().parse(response_format=case.response_format).choices[0].message.parsed
    setattr(parsed.observations.schema_smoke_view, field, value)
    assert _check_observation_guard(parsed) == [
        "instance observer claimed evidence without a bound instance or reference"]


@pytest.mark.parametrize("verdict,support", [("same", []), ("different", []), ("unknown", ["invented"])])
def test_schema_smoke_rejects_unbound_reference_claim(verdict, support) -> None:
    case = next(case for case in build_schema_cases("data:image/jpeg;base64,AA==")
                if case.name == "instance_observer")
    parsed = _ScriptedStructuredClient().parse(response_format=case.response_format).choices[0].message.parsed
    parsed.reference_comparison.verdict = verdict
    parsed.reference_comparison.supporting_observation_ids = support
    assert _check_observation_guard(parsed) == [
        "instance observer claimed evidence without a bound instance or reference"]


def test_schema_smoke_rejects_unbound_identity_claim() -> None:
    parsed = visual_verification_response_model(["candidate_identity"])(
        predicate_judgments={"candidate_identity": {
            "verdict": "true", "reason": "A book is visible.",
            "supporting_observation_ids": ["invented"]}})
    assert _check_final_guard(parsed) == ["final verifier claimed candidate evidence from an unbound image"]


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
