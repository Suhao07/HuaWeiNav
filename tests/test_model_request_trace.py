"""Record actual inputs before network dispatch, including failed calls."""
import json
from types import SimpleNamespace
import pytest
from prompting.schemas import ParsedVerification
from llm_utils.cognav_llm_adapter import _TracingParsedCompletions


def test_failed_request_keeps_exact_input_and_original_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path))
    failure = TimeoutError("provider response unavailable")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "verify this target"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
    ]}]

    def fail(**kwargs):
        assert kwargs["messages"] == messages
        raise failure

    adapter = _TracingParsedCompletions(SimpleNamespace(with_raw_response=SimpleNamespace(parse=fail)))
    with pytest.raises(TimeoutError) as error:
        adapter.parse(model="test-model", messages=messages, response_format=ParsedVerification,
                      temperature=0.0, trace_label="final_verify")
    assert error.value is failure
    paths = list(tmp_path.glob("request_*.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    assert payload["messages"] == messages
    assert payload["model"] == "test-model"
    assert payload["parameters"] == {"temperature": 0.0}
    assert payload["response_schema"] == ParsedVerification.model_json_schema()


@pytest.mark.parametrize("recover", [True, False])
def test_invalid_json_is_saved_before_validation_and_retried_once(tmp_path, monkeypatch, recover):
    from pydantic import ValidationError
    from llm_utils.lvlm_call_tracker import reset_counts
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path))
    reset_counts()
    malformed = '{"satisfied":false "decision":"uncertain"}'
    contents = iter([malformed, '{}' if recover else malformed])
    requests = []
    def request(**kwargs):
        requests.append(kwargs)
        content = next(contents)
        def parse():
            saved = [json.loads(p.read_text()) for p in tmp_path.glob('[0-9]*.json')]
            assert saved[-1]['raw_response'] == content
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                parsed=ParsedVerification.model_validate_json(content)))])
        return SimpleNamespace(text=json.dumps({"id": "provider-id", "choices": [
            {"message": {"content": content}, "finish_reason": "stop"}]}), parse=parse)
    adapter = _TracingParsedCompletions(SimpleNamespace(with_raw_response=SimpleNamespace(parse=request)))
    messages = [{"role": "user", "content": "same frozen observation"}]
    if recover:
        result = adapter.parse(model="test", messages=messages, response_format=ParsedVerification)
        assert result.choices[0].message.parsed.decision == "uncertain"
    else:
        with pytest.raises(ValidationError):
            adapter.parse(model="test", messages=messages, response_format=ParsedVerification)
    assert len(requests) == 2 and requests[1]['messages'][:-1] == messages
    assert len(messages) == 1
    assert len(list(tmp_path.glob('error_*.json'))) == (1 if recover else 2)
    responses = [json.loads(p.read_text()) for p in sorted(tmp_path.glob('[0-9]*.json'))]
    assert responses[1]['metadata']['retry_of'] == responses[0]['metadata']['request_id']


def test_verification_schema_declares_nested_fields():
    schema = ParsedVerification.model_json_schema()
    assert 'hard_constraints' not in schema['properties']
    assert 'improve_goals' in schema['$defs']['ParsedViewObjective']['properties']
