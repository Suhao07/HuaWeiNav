"""Focused tests for the self-hosted structured LVLM boundary."""

from types import SimpleNamespace

from pydantic import BaseModel

from llm_utils.cognav_llm_adapter import get_client_and_model
from llm_utils.self_hosted_llm_adapter import (
    HttpVlmSettings,
    SelfHostedOpenAICompatibleClient,
)
from llm_utils.structured_output import extract_json_object, inject_json_schema


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

    def create(self, **kwargs):
        """Return the next scripted chat completion."""

        self.requests.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _fake_client(responses: list[str]):
    """Build an OpenAI-like client around scripted completions."""

    completions = _FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


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
    assert isinstance(completions.requests[0]["messages"][-1]["content"], list)
    assert "corrected JSON object" in completions.requests[1]["messages"][-1]["content"]
    original = completions.requests[0]["messages"]
    assert completions.requests[1]["messages"][:len(original)] == original
    assert "response_format" not in completions.requests[0]


def test_self_hosted_adapter_falls_back_conservatively(monkeypatch) -> None:
    monkeypatch.setenv("STRIVE_LLM_PARSE_FALLBACK", "1")
    fake, _ = _fake_client(["service returned prose"])
    client = SelfHostedOpenAICompatibleClient(
        base_url="http://lvlm.example/v1",
        parse_retries=0,
        client=fake,
    )

    result = client.beta.chat.completions.parse(
        model="strive-qwen-vl",
        messages=[{"role": "user", "content": "verify"}],
        response_format=_Decision,
    )

    parsed = result.choices[0].message.parsed
    assert parsed.satisfied is False
    assert parsed.decision == "uncertain"


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


def test_transport_failure_preserves_exception_without_format_retry():
    """传输失败由传输层处理，格式修复只处理实际返回的响应。"""
    import pytest
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("transport unavailable")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = SelfHostedOpenAICompatibleClient(base_url="http://local/v1", client=fake)
    with pytest.raises(ConnectionError):
        client.beta.chat.completions.parse(model="qwen", messages=[], response_format=_Decision)
    assert len(calls) == 1
