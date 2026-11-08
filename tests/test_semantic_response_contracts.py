"""Regression contracts for the two captured LangMap model failures."""
import pytest
from pydantic import ValidationError
from prompting.schemas import GroundingResult, ParsedConceptMatch, CandidateConceptJudgment, concept_batch_response_model
from llm_utils.structured_output import validate_response_model


def test_grounding_bounds_reject_repetition():
    with pytest.raises(ValidationError):
        GroundingResult(detector_terms=['bed'] * 5)
    with pytest.raises(ValidationError):
        GroundingResult(detector_terms=['x' * 65])
    with pytest.raises(ValidationError):
        GroundingResult(reason='x' * 161)
    parsed = GroundingResult(detector_terms=['x' * 64] * 4, reason='x' * 160)
    assert parsed.detector_terms == ['x' * 64] * 4 and parsed.reason == 'x' * 160
    assert GroundingResult(detector_terms=['fan']).detector_terms == ['fan']


@pytest.mark.parametrize("field,value", [
    ("aliases", ["container"]),
    ("description", "Any container counts as a mug."),
    ("negative_terms", ["coffee mug"]),
])
def test_grounding_schema_rejects_generated_semantic_fields(field, value):
    with pytest.raises(ValidationError) as error:
        validate_response_model(GroundingResult, {"detector_terms": ["container"], field: value})
    assert any(item["loc"] == (field,) and "extra" in item["type"] for item in error.value.errors())


def test_grounding_schema_exposes_only_detector_proposal_and_reason():
    schema = getattr(GroundingResult, "model_json_schema", GroundingResult.schema)()
    assert set(schema["properties"]) == {"detector_terms", "reason"}
    assert schema["additionalProperties"] is False


def test_candidate_schema_requires_every_uid_and_every_judgment():
    model = concept_batch_response_model(['object:a', 'object:b'])
    negative = dict(matches_concept=False, confidence=0.95, terminal_eligible=False, reason='Different category.')
    with pytest.raises(ValidationError):
        validate_response_model(model, {})
    with pytest.raises(ValidationError):
        validate_response_model(model, {'object:a': negative})
    with pytest.raises(ValidationError):
        validate_response_model(model, {'object:a': negative, 'object:b': {}})
    parsed = validate_response_model(model, {'object:a': negative, 'object:b': negative})
    assert getattr(parsed, 'object:b').matches_concept is False


def test_grounding_propagates_model_failure(monkeypatch):
    from perception import detector_adapter
    from instruction_adapter.contracts import ConceptQuery
    def fail(_):
        raise ValueError('invalid model response')
    monkeypatch.setattr(detector_adapter, 'get_client_and_model', fail)
    monkeypatch.setenv('LLM_OFFLINE', '0')
    with pytest.raises(ValueError, match='invalid model response'):
        detector_adapter._llm_ground_terms(ConceptQuery(id='target', name='fan'), ['fan'], 'self_hosted')


@pytest.mark.parametrize("schema", [ParsedConceptMatch, CandidateConceptJudgment])
def test_concept_unknown_is_explicit_and_separate_from_candidate_eligibility(schema):
    payload = dict(matches_concept=None, candidate_eligible=True, confidence=0.9,
                   terminal_eligible=False, reason="Compatible broad class; identity unresolved.")
    parsed = validate_response_model(schema, payload)
    assert parsed.matches_concept is None
    assert parsed.candidate_eligible is True and parsed.terminal_eligible is False
    del payload["matches_concept"]
    with pytest.raises(ValidationError):
        validate_response_model(schema, payload)


def test_batch_retains_unknown_alongside_negative_and_requires_all_uids():
    model = concept_batch_response_model(["broad", "unrelated"])
    unknown = dict(matches_concept=None, candidate_eligible=True, confidence=0.9,
                   terminal_eligible=False, reason="Needs instance evidence.")
    negative = dict(matches_concept=False, confidence=0.9,
                    terminal_eligible=False, reason="Different identity.")
    parsed = validate_response_model(model, {"broad": unknown, "unrelated": negative})
    assert parsed.broad.matches_concept is None and parsed.broad.candidate_eligible
    assert parsed.unrelated.matches_concept is False and not parsed.unrelated.candidate_eligible
    with pytest.raises(ValidationError):
        validate_response_model(model, {"broad": unknown})
