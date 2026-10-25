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

    adapter = _TracingParsedCompletions(SimpleNamespace(parse=fail))
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
