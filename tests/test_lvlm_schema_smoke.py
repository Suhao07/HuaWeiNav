"""Tests for the production-schema LVLM deployment receipt."""

from __future__ import annotations

from types import SimpleNamespace

from deployment.lvlm_server.schema_smoke import (
    _check_final_guard,
    build_schema_cases,
    run_schema_cases,
)
from prompting.schemas import (
    GroundingResult,
    ParsedBatchConceptItem,
    ParsedBatchConceptMatch,
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
        elif schema is ParsedBatchConceptMatch:
            parsed = ParsedBatchConceptMatch(
                matches=[
                    ParsedBatchConceptItem(uid="anchor_1"),
                    ParsedBatchConceptItem(uid="distractor_1"),
                ]
            )
        elif schema is ParsedRelationResult:
            parsed = ParsedRelationResult(verified=False, confidence=0.5, need_better_view=True)
        elif schema is ParsedVerification:
            parsed = ParsedVerification(
                satisfied=False,
                decision="need_better_view",
                hard_constraints={"satisfied": False},
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


def test_schema_smoke_rejects_unsafe_final_accept() -> None:
    """Verify the deployment gate detects a hard-constraint STOP violation."""

    parsed = ParsedVerification(
        satisfied=True,
        semantic_satisfied=True,
        view_sufficient_for_stop=True,
        hard_constraints={"satisfied": False},
        decision="accept",
    )

    failures = _check_final_guard(parsed)

    assert len(failures) == 2
    assert any("satisfied=true" in failure for failure in failures)
    assert any("decision=accept" in failure for failure in failures)
