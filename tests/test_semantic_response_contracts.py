"""Regression contracts for the two captured LangMap model failures."""
import pytest
from pydantic import ValidationError
from prompting.schemas import GroundingResult, concept_batch_response_model
from llm_utils.structured_output import validate_response_model


def test_grounding_bounds_reject_repetition():
    with pytest.raises(ValidationError):
        GroundingResult(detector_terms=['bed'] * 5)
    with pytest.raises(ValidationError):
        GroundingResult(negative_terms=['fan light ' * 10])
    assert GroundingResult(detector_terms=['fan']).detector_terms == ['fan']


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
