"""Focused tests for the self-hosted structured LVLM boundary."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from llm_utils.cognav_llm_adapter import _CogNavParsedChat, get_client_and_model
from llm_utils.lvlm_call_tracker import counts, reset_counts
from llm_utils.self_hosted_llm_adapter import (
    HttpVlmSettings,
    SelfHostedOpenAICompatibleClient,
)
from llm_utils.structured_output import (
    StructuredOutputError,
    extract_json_object,
    inject_json_schema,
)


@pytest.fixture(autouse=True)
def isolated_traces(tmp_path, monkeypatch):
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path / 'model_requests'))
    reset_counts()
    yield
    reset_counts()


class _Decision(BaseModel):
    """Minimal conservative verifier response used by adapter tests."""

    satisfied: bool = False
    decision: str = "uncertain"
    reason: str = "fallback"


class _FakeCompletions:
    """Return scripted OpenAI-compatible responses and retain requests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.finish_reason = "stop"
        self.refusal = None

    def create(self, **kwargs):
        """Return the next scripted chat completion."""

        self.requests.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=self.refusal),
                                     finish_reason=self.finish_reason)]
        )


def _fake_client(responses: list[str]):
    """Build an OpenAI-like client around scripted completions."""

    completions = _FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


@pytest.mark.parametrize("guided", ["0", "1"])
@pytest.mark.parametrize("verdict", ["true", "false", "unknown"])
def test_scripted_two_stage_verifier_uses_production_adapter_and_traces(tmp_path, monkeypatch, guided, verdict):
    import base64
    import json
    from hashlib import sha256
    import instruction_adapter.verifier as module
    from prompting.schemas import visual_verification_response_model
    from verification.contracts import CandidateInstance

    image = tmp_path / "frame.png"
    image.write_bytes(b"scripted-image-bytes-not-a-real-vlm-test")
    facts = {"observed_category": "chair", "visible_parts": ["seat", "backrest", "legs"],
             "support_and_attachment": "Legs on floor.", "visible_surface_content": "",
             "visible_context": ["A sink and stove are across the doorway."],
             "uncertainties": ["The candidate's room is not established."]}
    observation = {"observations": {"now": facts}}
    final = visual_verification_response_model(["candidate_identity"])(
        observed_category="invented category", observed_structure="Invented structure",
        predicate_judgments={"candidate_identity": {"verdict": verdict,
            "reason": "Scripted response, not a model capability judgment.",
            "supporting_observation_ids": ["now"]}}).model_dump()
    fake, completions = _fake_client([json.dumps(observation), json.dumps(final)])
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake, parse_retries=4)
    monkeypatch.setattr(module, "get_client_and_model", lambda provider: (client, "scripted-local"))
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "0")
    monkeypatch.setenv("VLN_LVLM_GUIDED_JSON", guided)
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path / "requests"))
    result = module.FinalInstructionVerifier(vlm="self_hosted", evidence_dir=str(tmp_path / "images")).verify(
        "Find a backless seat.", {"targets": [{"id": "goal", "name": "backless seat"}]},
        CandidateInstance("candidate-1", "untrusted detector label"), {
            "current_rgb_with_bbox_path": str(image), "geometry": {"bbox_xyxy": [0, 0, 4, 4]},
            "observation_identity": {"candidate_uid": "candidate-1", "task_revision": 1, "observation_id": "now"},
            "hard_stop_constraints": {"satisfied": True}})
    assert len(completions.requests) == 2
    observer_request, final_request = completions.requests
    assert "backless seat" not in json.dumps(observer_request)
    assert "untrusted detector label" not in json.dumps(completions.requests)
    assert base64.b64decode(observer_request["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1]) == image.read_bytes()
    assert all(item["type"] == "text" for item in final_request["messages"][1]["content"])
    bound = json.loads(final_request["messages"][1]["content"][1]["text"])["bound_observations"]["now"]
    assert bound["facts"] == facts
    assert bound["images"][0]["image_sha256"] == sha256(image.read_bytes()).hexdigest()
    assert result.accepted is (verdict == "true")
    assert result.observed_category == "chair"
    assert result.decision == {"true": "accept", "false": "reject_candidate", "unknown": "uncertain"}[verdict]
    requests = {item["request_id"]: item for path in (tmp_path / "requests").glob("request_*.json")
                for item in [json.loads(path.read_text())]}
    responses = [json.loads(path.read_text()) for path in sorted((tmp_path / "requests").glob("[0-9]*.json"))]
    assert [item["kind"] for item in responses] == ["instance_observer", "final_instruction_verifier"]
    for response, sent, expected in zip(responses, completions.requests, (observation, final)):
        saved = requests[response["metadata"]["request_id"]]
        assert saved["messages"] == sent["messages"]
        assert json.loads(response["raw_response"]) == expected
        assert response["metadata"]["attempt"] == 1
        assert "parse_retries" not in sent
    if guided == "1":
        schema = observer_request["extra_body"]["guided_json"]
        assert "visible_context" in schema["$defs"]["ObservedInstanceFacts"]["required"]


def test_schema_injection_preserves_multimodal_content() -> None:
    messages = [
        {"role": "system", "content": "verify target"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "is this a chair?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
            ],
        },
    ]

    normalized = inject_json_schema(messages, _Decision)

    assert "JSON schema" in normalized[0]["content"]
    assert normalized[1]["content"] == messages[1]["content"]
    assert normalized is not messages


def test_json_extractor_repairs_common_provider_formatting() -> None:
    payload = extract_json_object("```json\n{satisfied: False, decision: 'uncertain', reason: 'bad view',}\n```")

    assert payload == {
        "satisfied": False,
        "decision": "uncertain",
        "reason": "bad view",
    }


def test_self_hosted_adapter_retries_invalid_json_and_validates_response() -> None:
    fake, completions = _fake_client(
        [
            "not valid json",
            '{"satisfied": true, "decision": "accept", "reason": "target visible"}',
        ]
    )
    client = SelfHostedOpenAICompatibleClient(
        base_url="http://lvlm.example/v1",
        parse_retries=1,
        client=fake,
    )

    result = client.beta.chat.completions.parse(
        model="strive-qwen-vl",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
                    {"type": "text", "text": "verify"},
                ],
            }
        ],
        response_format=_Decision,
        trace_label="test_verifier",
        temperature=0,
    )

    assert result.choices[0].message.parsed.satisfied is True
    assert result.choices[0].message.parsed.decision == "accept"
    assert len(completions.requests) == 2
    assert counts()['calls'] == counts()['responses'] == {'test_verifier': 2}
    assert counts()['total_calls'] == counts()['total_responses'] == 2
    assert isinstance(completions.requests[0]["messages"][-1]["content"], list)
    assert "corrected JSON object" in completions.requests[1]["messages"][-1]["content"]
    original = completions.requests[0]["messages"]
    assert completions.requests[1]["messages"][:len(original)] == original
    assert "response_format" not in completions.requests[0]


def test_self_hosted_adapter_propagates_exhaustion_despite_legacy_fallback_setting(monkeypatch) -> None:
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")
    fake, completions = _fake_client(["service returned prose"] * 2)
    client = SelfHostedOpenAICompatibleClient(
        base_url="http://lvlm.example/v1",
        parse_retries=1,
        client=fake,
    )

    with pytest.raises(ValueError):
        client.beta.chat.completions.parse(
            model="strive-qwen-vl",
            messages=[{"role": "user", "content": "verify"}],
            response_format=_Decision,
        )

    assert len(completions.requests) == 2
    assert counts()["total_calls"] == counts()["total_responses"] == 2


def test_provider_factory_selects_self_hosted_client(monkeypatch) -> None:
    # Other provider tests may intentionally enable offline mode. This factory
    # test exercises the online self-hosted branch and must own its environment.
    monkeypatch.delenv("LLM_OFFLINE", raising=False)
    sentinel = object()
    monkeypatch.setenv("STRIVE_LVLM_MODEL", "strive-qwen-test")
    monkeypatch.setattr(
        SelfHostedOpenAICompatibleClient,
        "from_environment",
        classmethod(lambda cls: sentinel),
    )

    client, model = get_client_and_model("self_hosted")

    assert client is sentinel
    assert model == "strive-qwen-test"


def test_http_settings_prefer_vln_variables_and_support_legacy_aliases(monkeypatch) -> None:
    monkeypatch.setenv("VLN_LVLM_BASE_URL", "http://vln-server:8000/v1")
    monkeypatch.setenv("VLN_LVLM_API_KEY", "vln-token")
    monkeypatch.setenv("VLN_LVLM_MODEL", "qwen-vl-server")
    monkeypatch.setenv("STRIVE_LVLM_BASE_URL", "http://legacy-server:8000/v1")
    monkeypatch.setenv("STRIVE_LVLM_API_KEY", "legacy-token")

    settings = HttpVlmSettings.from_environment()

    assert settings.base_url == "http://vln-server:8000/v1"
    assert settings.api_key == "vln-token"


def test_provider_factory_accepts_served_model_alias(monkeypatch) -> None:
    monkeypatch.delenv("LLM_OFFLINE", raising=False)
    monkeypatch.delenv("VLN_LVLM_MODEL", raising=False)
    monkeypatch.delenv("STRIVE_LVLM_MODEL", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("STRIVE_LVLM_SERVED_MODEL", "qwen-vl-served")
    sentinel = object()
    monkeypatch.setattr(
        SelfHostedOpenAICompatibleClient,
        "from_environment",
        classmethod(lambda cls: sentinel),
    )

    client, model = get_client_and_model("self_hosted")

    assert client is sentinel
    assert model == "qwen-vl-served"


def test_schema_echo_is_retried_instead_of_accepting_defaults(monkeypatch):
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "0")
    fake, completions = _fake_client([
        '{"properties":{"satisfied":true},"type":"object"}',
        '{"satisfied":false,"decision":"uncertain","reason":"missing evidence"}',
    ])
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake)
    result = client.beta.chat.completions.parse(
        model="qwen", messages=[{"role":"user","content":"verify"}], response_format=_Decision)
    assert len(completions.requests) == 2
    assert result.choices[0].message.parsed.reason == "missing evidence"


def test_guided_json_requires_complete_instance_and_rejects_trailing_text(monkeypatch):
    monkeypatch.setenv("VLN_LVLM_GUIDED_JSON", "1")
    fake, completions = _fake_client([
        '{"satisfied":false} trailing schema',
        '{"satisfied":false,"decision":"uncertain","reason":"missing evidence"}',
    ])
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake)
    client.beta.chat.completions.parse(
        model="qwen", messages=[{"role":"user","content":"verify"}], response_format=_Decision)
    assert len(completions.requests) == 2
    schema = completions.requests[0]["extra_body"]["guided_json"]
    assert set(schema["required"]) == {"satisfied", "decision", "reason"}
    assert schema["additionalProperties"] is False


def test_length_retry_increases_budget_and_preserves_candidates():
    """截断重试沿用候选事实，增长预算，并保存正常完成结果。"""
    requests = []
    original = [{"role": "user", "content": [{"type": "text", "text": 'candidates: ["a", "b"]'}]}]

    def create(**kwargs):
        requests.append(kwargs)
        truncated = len(requests) == 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"satisfied":' if truncated else
                                    '{"satisfied":false,"decision":"uncertain","reason":"evidence"}'),
            finish_reason="length" if truncated else "stop")])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake)
    client.beta.chat.completions.parse(model="qwen", messages=original,
                                      response_format=_Decision, max_tokens=3200)
    assert [r["max_tokens"] for r in requests] == [3200, 6400]
    assert requests[1]["messages"][:len(requests[0]["messages"])] == requests[0]["messages"]


@pytest.mark.parametrize("failure_type", [ConnectionError, ValueError, PermissionError, OSError])
@pytest.mark.parametrize("parse_kwargs", [{}, {"parse_retries": 0}])
def test_transport_failure_preserves_exception_without_format_retry(monkeypatch, failure_type, parse_kwargs):
    """Transport failures never enter format repair or conservative fallback."""
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")
    calls = []
    failure = failure_type("transport unavailable")

    def create(**kwargs):
        calls.append(kwargs)
        raise failure

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake, parse_retries=3)
    with pytest.raises(failure_type) as caught:
        client.beta.chat.completions.parse(model="qwen", messages=[], response_format=_Decision,
                                          **parse_kwargs)
    assert caught.value is failure
    assert len(calls) == 1
    assert counts()['total_calls'] == 1 and counts()['total_responses'] == 0
    assert "parse_retries" not in calls[0]


def test_structured_output_error_preserves_message_and_content():
    error = StructuredOutputError("invalid response")
    assert isinstance(error, ValueError)
    assert str(error) == "invalid response"
    assert error.raw_content == ""
    assert StructuredOutputError("invalid response", " raw response ").raw_content == " raw response "


@pytest.mark.parametrize("content,finish_reason,guided", [
    ("  not JSON\n", "stop", "0"),
    ('{"satisfied":', "length", "0"),
    ('{"satisfied":true}', "length", "0"),
    ('{"satisfied":"not a boolean"}', "stop", "0"),
    ('{"properties":{"satisfied":true},"type":"object"}', "stop", "0"),
    ('{"satisfied":true} trailing text', "stop", "1"),
])
def test_zero_parse_retries_overrides_repairs_and_fallback(monkeypatch, content, finish_reason, guided):
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")
    monkeypatch.setenv("VLN_LVLM_GUIDED_JSON", guided)
    fake, completions = _fake_client([content])
    completions.finish_reason = finish_reason
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake, parse_retries=4)

    with pytest.raises(StructuredOutputError) as caught:
        client.beta.chat.completions.parse(model="qwen", messages=[], response_format=_Decision,
                                          parse_retries=0)

    assert caught.value.raw_content == content
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value) == str(caught.value.__cause__)
    assert len(completions.requests) == 1
    assert "parse_retries" not in completions.requests[0]


def test_per_call_retry_override_does_not_change_configured_default(monkeypatch):
    monkeypatch.setenv("VLN_LVLM_GUIDED_JSON", "0")
    fake, completions = _fake_client(["invalid", "invalid", '{"satisfied":true}'])
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake, parse_retries=1)
    with pytest.raises(StructuredOutputError):
        client.beta.chat.completions.parse(model="qwen", messages=[], response_format=_Decision,
                                          parse_retries=0)
    result = client.beta.chat.completions.parse(model="qwen", messages=[], response_format=_Decision)
    assert result.choices[0].message.parsed.satisfied is True
    assert len(completions.requests) == 3


@pytest.mark.parametrize("content,error_type", [
    ("not JSON", ValueError),
    ('{"satisfied":"not a boolean"}', ValidationError),
])
def test_default_exhausted_parse_errors_keep_original_type(monkeypatch, content, error_type):
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "0")
    fake, completions = _fake_client([content] * 2)
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake)
    with pytest.raises(error_type) as caught:
        client.beta.chat.completions.parse(model="qwen", messages=[], response_format=_Decision)
    assert not isinstance(caught.value, StructuredOutputError)
    assert len(completions.requests) == 2


@pytest.mark.parametrize("content", [" not JSON\n", '{"satisfied":', '{"satisfied":"not a boolean"}'])
def test_cognav_zero_parse_retries_preserves_invalid_content(monkeypatch, content):
    adapter = _CogNavParsedChat.__new__(_CogNavParsedChat)
    fake, completions = _fake_client([content])
    adapter._client = SimpleNamespace(chat_completion=fake.chat.completions.create)
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")

    with pytest.raises(StructuredOutputError) as caught:
        adapter.parse(model="cognav", messages=[], response_format=_Decision, parse_retries=0)

    assert caught.value.raw_content == content
    assert isinstance(caught.value.__cause__, ValueError)
    assert len(completions.requests) == 1
    assert "parse_retries" not in completions.requests[0]


@pytest.mark.parametrize("parse_kwargs", [{}, {"parse_retries": 0}])
def test_cognav_transport_value_error_is_not_wrapped(monkeypatch, parse_kwargs):
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")
    failure = ValueError("transport rejected request")
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise failure

    adapter = _CogNavParsedChat.__new__(_CogNavParsedChat)
    adapter._client = SimpleNamespace(chat_completion=fail)
    with pytest.raises(ValueError) as caught:
        adapter.parse(model="cognav", messages=[], response_format=_Decision, **parse_kwargs)
    assert caught.value is failure
    assert len(calls) == 1
    assert counts()['total_calls'] == 1 and counts()['total_responses'] == 0


@pytest.mark.parametrize("provider", ["self_hosted", "cognav"])
@pytest.mark.parametrize("filtered", [False, True])
def test_provider_refusals_do_not_enter_semantic_repair(monkeypatch, provider, filtered):
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")
    fake, completions = _fake_client(['{"satisfied":true}'])
    completions.finish_reason = "content_filter" if filtered else "stop"
    completions.refusal = None if filtered else "provider refusal"
    if provider == "self_hosted":
        adapter = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake,
                                                   parse_retries=4).beta.chat.completions
    else:
        adapter = _CogNavParsedChat.__new__(_CogNavParsedChat)
        adapter._client = SimpleNamespace(chat_completion=fake.chat.completions.create)
    with pytest.raises(RuntimeError, match="provider refused"):
        adapter.parse(model="qwen", messages=[], response_format=_Decision, parse_retries=0)
    assert len(completions.requests) == 1


def test_cognav_default_parse_fallback_is_preserved(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")
    fake, completions = _fake_client(["invalid JSON"])
    adapter = _CogNavParsedChat.__new__(_CogNavParsedChat)
    adapter._client = SimpleNamespace(chat_completion=fake.chat.completions.create)
    result = adapter.parse(model="cognav", messages=[], response_format=_Decision)
    assert result.choices[0].message.parsed.satisfied is False
    assert len(completions.requests) == 1
    assert counts()['total_calls'] == counts()['total_responses'] == 1
