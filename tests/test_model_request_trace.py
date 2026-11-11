"""Record actual inputs before network dispatch, including failed calls."""
import json
from types import SimpleNamespace
import pytest
from prompting.schemas import ParsedVerification
from llm_utils.cognav_llm_adapter import _TracingParsedCompletions
from llm_utils.lvlm_call_tracker import counts, reset_counts


@pytest.fixture(autouse=True)
def isolated_counts():
    reset_counts()
    yield
    reset_counts()


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
    assert counts()['calls'] == {'final_verify': 1}
    assert counts()['total_calls'] == 1
    assert counts()['responses'] == {} and counts()['total_responses'] == 0
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
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path))
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
        assert result.choices[0].message.parsed.observed_category == ""
    else:
        with pytest.raises(ValidationError):
            adapter.parse(model="test", messages=messages, response_format=ParsedVerification)
    assert len(requests) == 2 and requests[1]['messages'][:-1] == messages
    assert len(messages) == 1
    assert len(list(tmp_path.glob('error_*.json'))) == (1 if recover else 2)
    responses = [json.loads(p.read_text()) for p in sorted(tmp_path.glob('[0-9]*.json'))]
    assert responses[1]['metadata']['retry_of'] == responses[0]['metadata']['request_id']
    assert counts()['calls'] == counts()['responses'] == {'parse': 2}
    assert counts()['total_calls'] == counts()['total_responses'] == 2


def test_verification_schema_declares_nested_fields():
    schema = ParsedVerification.model_json_schema()
    assert 'hard_constraints' not in schema['properties']
    assert 'improve_goals' in schema['$defs']['ParsedViewObjective']['properties']


@pytest.mark.parametrize('explicit', [None, {'type': 'enabled'}])
def test_doubao_thinking_default_and_explicit_override(tmp_path, monkeypatch, explicit):
    """Provider requests carry the chosen thinking mode and retain usage evidence."""
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    expected = explicit or {'type': 'disabled'}
    def request(**kwargs):
        assert kwargs['extra_body']['thinking'] == expected
        assert json.dumps(ParsedVerification.model_json_schema(), ensure_ascii=False,
                          separators=(',', ':')) in kwargs['messages'][0]['content']
        assert kwargs['response_format'] == {'type': 'text'}
        from openai.types.chat import ChatCompletion
        payload = dict(id='test', model='doubao-seed-2-0-pro-260215', created=0,
            object='chat.completion', choices=[dict(index=0, message=dict(role='assistant', content='{}'),
            finish_reason='stop')], usage=dict(completion_tokens=1,prompt_tokens=1,total_tokens=2,
            completion_tokens_details=dict(reasoning_tokens=0)))
        return SimpleNamespace(text=json.dumps(payload), parse=lambda: ChatCompletion(**payload))
    adapter = _TracingParsedCompletions(SimpleNamespace(with_raw_response=SimpleNamespace(create=request)))
    extra = {'extra_body': {'thinking': explicit}} if explicit else {}
    adapter.parse(model='doubao-seed-2-0-pro-260215', messages=[],
                  response_format=ParsedVerification, **extra)
    response = json.loads(next(tmp_path.glob('[0-9]*.json')).read_text())
    assert response['metadata']['usage']['completion_tokens_details']['reasoning_tokens'] == 0


@pytest.mark.parametrize('recover', [True, False])
def test_truncated_output_shares_single_regeneration_budget(tmp_path, monkeypatch, recover):
    """截断输出保留原文；一次重生成后仍失败则传播SDK异常。"""
    from openai import LengthFinishReasonError
    from openai.types.chat import ChatCompletion
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    requests = []
    completion = ChatCompletion(id='truncated', model='test', created=0,
        object='chat.completion', choices=[dict(index=0, finish_reason='length',
        message=dict(role='assistant', content='{"satisfied":'))])
    failure = LengthFinishReasonError(completion=completion)
    def request(**kwargs):
        requests.append(kwargs)
        success = recover and len(requests) == 2
        raw = '{}' if success else '{"satisfied":'
        def parse():
            assert any(json.loads(p.read_text())['raw_response'] == raw
                       for p in tmp_path.glob('[0-9]*.json'))
            if not success:
                raise failure
            return ParsedVerification()
        return SimpleNamespace(text=json.dumps({'choices': [{'message': {'content': raw},
            'finish_reason': 'stop' if success else 'length'}]}), parse=parse)
    adapter = _TracingParsedCompletions(SimpleNamespace(with_raw_response=SimpleNamespace(parse=request)))
    messages = [{'role': 'user', 'content': 'frozen image evidence'}]
    if recover:
        assert adapter.parse(model='test', messages=messages,
                             response_format=ParsedVerification).observed_category == ''
    else:
        with pytest.raises(LengthFinishReasonError) as caught:
            adapter.parse(model='test', messages=messages, response_format=ParsedVerification)
        assert caught.value is failure
    assert len(requests) == 2
    assert requests[1]['messages'][:-1] == messages


@pytest.mark.parametrize('recover', [True, False])
def test_doubao_text_keeps_strict_local_validation(tmp_path, monkeypatch, recover):
    from openai.types.chat import ChatCompletion
    from pydantic import ValidationError
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raw = '{}' if recover and len(calls) == 2 else '{"decision":'
        completion = ChatCompletion(id='text', model=kwargs['model'], created=0,
            object='chat.completion', choices=[dict(index=0, finish_reason='stop',
            message=dict(role='assistant', content=raw))])
        return SimpleNamespace(text=completion.model_dump_json(), parse=lambda: completion)
    adapter = _TracingParsedCompletions(SimpleNamespace(with_raw_response=SimpleNamespace(create=create)))
    kwargs = dict(model='doubao-seed-2-0-pro-260215', messages=[{'role':'user','content':'same evidence'}],
                  response_format=ParsedVerification)
    if recover:
        assert adapter.parse(**kwargs).choices[0].message.parsed.observed_category == ''
    else:
        with pytest.raises(ValidationError):
            adapter.parse(**kwargs)
    assert len(calls) == 2
    assert all(call['response_format'] == {'type':'text'} for call in calls)
    assert len(list(tmp_path.glob('error_*.json'))) == (1 if recover else 2)


def test_real_sdk_doubao_uses_standard_create_route(tmp_path, monkeypatch):
    import httpx
    from openai import OpenAI
    from prompting.schemas import ParsedInstanceCorrespondence
    from llm_utils.cognav_llm_adapter import TracingOpenAICompatibleClient
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    calls = []
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path == '/v1/chat/completions'
        assert body['response_format'] == {'type': 'text'}
        return httpx.Response(200, json=dict(id='fixture', model=body['model'], created=0,
            object='chat.completion', choices=[dict(index=0, finish_reason='stop',
            message=dict(role='assistant', content=json.dumps(dict(outcome='fragments',
                evidence_summary='controlled SDK fixture'))))]))
    with OpenAI(api_key='fixture-only', base_url='https://fixture.invalid/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as sdk:
        result = TracingOpenAICompatibleClient(sdk).beta.chat.completions.parse(
            model='doubao-seed-2-0-pro-260215', messages=[{'role':'user','content':'fixture'}],
            response_format=ParsedInstanceCorrespondence)
    assert result.choices[0].message.parsed.outcome == 'fragments'
    assert len(calls) == 1


@pytest.mark.parametrize('model', ['test-model', 'doubao-test'])
@pytest.mark.parametrize('content,finish_reason', [
    ('  not JSON\n', 'stop'),
    ('{"satisfied":', 'length'),
    ('{}', 'length'),
    ('{"satisfied":"not a boolean"}', 'stop'),
])
def test_sdk_zero_retry_preserves_invalid_content(tmp_path, monkeypatch, model, content, finish_reason):
    import httpx
    from openai import OpenAI, LengthFinishReasonError
    from pydantic import ValidationError
    from llm_utils.cognav_llm_adapter import TracingOpenAICompatibleClient
    from llm_utils.structured_output import StructuredOutputError

    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    monkeypatch.setenv('STRIVE_LLM_PARSE_FALLBACK', '1')
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert 'parse_retries' not in body
        return httpx.Response(200, json=dict(id='fixture', model=model, created=0,
            object='chat.completion', choices=[dict(index=0, finish_reason=finish_reason,
            message=dict(role='assistant', content=content))]))

    with OpenAI(api_key='fixture-only', base_url='https://fixture.invalid/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as sdk:
        with pytest.raises(StructuredOutputError) as caught:
            TracingOpenAICompatibleClient(sdk).beta.chat.completions.parse(
                model=model, messages=[], response_format=ParsedVerification, parse_retries=0)
    assert caught.value.raw_content == content
    cause_type = LengthFinishReasonError if finish_reason == 'length' else ValidationError
    assert isinstance(caught.value.__cause__, cause_type)
    assert len(calls) == 1
    recorded = json.loads(next(tmp_path.glob('request_*.json')).read_text())
    assert 'parse_retries' not in recorded['parameters']


@pytest.mark.parametrize('model', ['test-model', 'doubao-test'])
@pytest.mark.parametrize('parse_kwargs', [{}, {'parse_retries': 0}])
def test_tracing_transport_value_error_is_unchanged(tmp_path, monkeypatch, model, parse_kwargs):
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    monkeypatch.setenv('STRIVE_LLM_PARSE_FALLBACK', '1')
    failure = ValueError('transport rejected request')
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise failure

    adapter = _TracingParsedCompletions(SimpleNamespace(
        with_raw_response=SimpleNamespace(parse=fail, create=fail)))
    with pytest.raises(ValueError) as caught:
        adapter.parse(model=model, messages=[], response_format=ParsedVerification, **parse_kwargs)
    assert caught.value is failure
    assert len(calls) == 1
    assert counts()['total_calls'] == 1 and counts()['total_responses'] == 0
    assert 'parse_retries' not in calls[0]


@pytest.mark.parametrize('model', ['test-model', 'doubao-test'])
@pytest.mark.parametrize('finish_reason', ['stop', 'content_filter'])
def test_sdk_zero_retry_preserves_refusal_semantics(tmp_path, monkeypatch, model, finish_reason):
    import httpx
    from openai import OpenAI, ContentFilterFinishReasonError
    from llm_utils.cognav_llm_adapter import TracingOpenAICompatibleClient

    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=dict(id='fixture', model=model, created=0,
            object='chat.completion', choices=[dict(index=0, finish_reason=finish_reason,
            message=dict(role='assistant', content=None, refusal='provider refusal'))]))

    with OpenAI(api_key='fixture-only', base_url='https://fixture.invalid/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as sdk:
        parse = TracingOpenAICompatibleClient(sdk).beta.chat.completions.parse
        kwargs = dict(model=model, messages=[], response_format=ParsedVerification, parse_retries=0)
        if finish_reason == 'content_filter':
            with pytest.raises(ContentFilterFinishReasonError):
                parse(**kwargs)
        else:
            message = parse(**kwargs).choices[0].message
            assert message.refusal == 'provider refusal'
            assert message.parsed is None
    assert len(calls) == 1


def test_tracing_per_call_retry_budget_is_not_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        content = '{}' if len(calls) == 3 else 'invalid'
        return SimpleNamespace(text=json.dumps({'choices': [{'message': {'content': content},
            'finish_reason': 'stop'}]}), parse=lambda: ParsedVerification.model_validate_json(content))

    adapter = _TracingParsedCompletions(SimpleNamespace(with_raw_response=SimpleNamespace(parse=request)))
    result = adapter.parse(model='test', messages=[], response_format=ParsedVerification, parse_retries=2)
    assert result.observed_category == ''
    assert len(calls) == 3
    assert counts()['total_calls'] == counts()['total_responses'] == 3
    assert all('parse_retries' not in call for call in calls)


@pytest.mark.parametrize('provider', ['openai', 'doubao', 'self_hosted'])
@pytest.mark.parametrize('recover', [True, False])
def test_sdk_transport_retries_belong_to_one_adapter_request(tmp_path, monkeypatch, provider, recover):
    import httpx
    from openai import OpenAI, InternalServerError
    from llm_utils.cognav_llm_adapter import TracingOpenAICompatibleClient
    from llm_utils.self_hosted_llm_adapter import SelfHostedOpenAICompatibleClient

    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1 or not recover:
            return httpx.Response(503, headers={'retry-after-ms': '1'},
                                  json={'error': {'message': 'fixture unavailable'}})
        return httpx.Response(200, json=dict(id='fixture', model='test', created=0,
            object='chat.completion', choices=[dict(index=0, finish_reason='stop',
            message=dict(role='assistant', content=ParsedVerification(observed_category='chair').model_dump_json()))]))

    with OpenAI(api_key='fixture-only', base_url='https://fixture.invalid/v1', max_retries=1,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as sdk:
        adapter = (SelfHostedOpenAICompatibleClient(base_url='https://fixture.invalid/v1', client=sdk)
                   if provider == 'self_hosted' else TracingOpenAICompatibleClient(sdk))
        kwargs = dict(model='doubao-test' if provider == 'doubao' else 'test-model',
                      messages=[], response_format=ParsedVerification, trace_label='verify')
        if recover:
            assert adapter.beta.chat.completions.parse(**kwargs).choices[0].message.parsed.observed_category == 'chair'
        else:
            with pytest.raises(InternalServerError):
                adapter.beta.chat.completions.parse(**kwargs)
    assert len(calls) == 2
    assert counts()['calls'] == {'verify': 1}
    assert counts()['total_calls'] == 1 and counts()['total_responses'] == int(recover)
    assert counts()['responses'] == ({'verify': 1} if recover else {})
    assert 'excludes SDK-internal transport retries' in counts()['calls_scope']
    assert len(list(tmp_path.glob('request_*.json'))) == 1
    assert len(list(tmp_path.glob('[0-9]*.json'))) == int(recover)


def test_request_persistence_failure_prevents_dispatch_and_count(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from llm_utils import lvlm_call_tracker as tracker

    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    failure = OSError('request trace unavailable')
    monkeypatch.setattr(tracker.json, 'dump', Mock(side_effect=failure))
    dispatch = Mock()
    adapter = _TracingParsedCompletions(SimpleNamespace(with_raw_response=SimpleNamespace(parse=dispatch)))
    with pytest.raises(OSError) as caught:
        adapter.parse(model='test-model', messages=[], response_format=ParsedVerification)
    assert caught.value is failure
    dispatch.assert_not_called()
    assert counts()['total_calls'] == counts()['total_responses'] == 0
