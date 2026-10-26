import os
import json
import sys
import time
from types import SimpleNamespace

from constants import COGNAV_MODEL_NAME, DEFAULT_VLM, GEMINI_MODEL_NAME, require_gemini_key
from llm_utils.lvlm_call_tracker import record_call, record_request, record_parse_error
from llm_utils.structured_output import (
    extract_json_object as _extract_json_object,
    fallback_payload as _fallback_payload,
    inject_json_schema,
    validate_response_model as _validate_model,
)


def _openai_client_class():
    from openai import OpenAI

    return OpenAI


def _add_cognav_to_path() -> None:
    cognav_root = os.getenv("COGNAV_OBJNAV_PATH")
    if not cognav_root:
        return
    if os.path.isdir(cognav_root) and cognav_root not in sys.path:
        sys.path.insert(0, cognav_root)


class _CogNavParsedChat:
    def __init__(self) -> None:
        self._client = None
        if os.getenv("LLM_OFFLINE", "0") in ("1", "true", "True"):
            return
        _add_cognav_to_path()
        from utils.llm_client import LLMClient

        self._client = LLMClient(apikey_file=os.getenv("COGNAV_APIKEY_FILE", "./apikey.txt"))

    def parse(self, model, messages, response_format, **kwargs):
        # CogNav LLMClient 不原生支持 OpenAI beta.parse。
        # 因此把 Pydantic schema 注入 system prompt，再把返回 JSON 校验回同一 schema。
        if os.getenv("LLM_OFFLINE", "0") in ("1", "true", "True"):
            parsed = _validate_model(response_format, _fallback_payload(response_format))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=""))]
            )

        normalized = inject_json_schema(messages, response_format)

        if self._client is None:
            raise RuntimeError("CogNav LLM client is unavailable; use LLM_OFFLINE=1 or --vlm ark/openai/gemini")

        trace_label = kwargs.get("trace_label") or getattr(response_format, "__name__", None) or "parse"
        request_id = record_request(str(trace_label), model=model, messages=normalized,
                                    response_format=response_format,
                                    parameters={"temperature": kwargs.get("temperature", 0.0),
                                                "max_tokens": kwargs.get("max_tokens")})
        started = time.perf_counter()
        completion = self._client.chat_completion(
            messages=normalized,
            model=model,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens"),
        )
        content = completion.choices[0].message.content
        trace_label = kwargs.get("trace_label") or getattr(response_format, "__name__", None) or "parse"
        record_call(
            str(trace_label),
            raw_response=str(content or ""),
            metadata={
                "model": model,
                "request_id": request_id,
                "response_format": getattr(response_format, "__name__", str(response_format)),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            },
        )
        try:
            parsed_payload = _extract_json_object(content)
        except Exception:
            # 解析失败保留原始异常和对应请求记录。
            # 默认传播解析错误；历史兼容实验可显式开启 fallback。
            if os.getenv("STRIVE_LLM_PARSE_FALLBACK", "0").lower() in ("0", "false", "no", "off"):
                raise
            print(
                "[CogNavLLMAdapter] JSON parse failed; using conservative fallback. "
                f"Raw response prefix={str(content)[:300]!r}"
            )
            parsed_payload = _fallback_payload(response_format)
        parsed = _validate_model(response_format, parsed_payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))]
        )


class _CogNavChatCompletions:
    def __init__(self) -> None:
        self.completions = SimpleNamespace(parse=_CogNavParsedChat().parse)


class _CogNavBeta:
    def __init__(self) -> None:
        self.chat = _CogNavChatCompletions()


class CogNavOpenAICompatibleClient:
    def __init__(self) -> None:
        self.beta = _CogNavBeta()


class _TracingParsedCompletions:
    def __init__(self, inner) -> None:
        self._inner = inner

    def parse(self, *args, **kwargs):
        """Save raw content before SDK validation; retry one malformed response.

        重新生成只恢复请求协议，输入观测保持不变。两次校验失败后传播
        原始异常；超时、鉴权、拒绝及截断异常由其原有执行边界处理。
        """
        from pydantic import ValidationError

        trace_label = kwargs.pop("trace_label", "parse")
        if kwargs["model"].startswith("doubao-"):
            # 导航默认采用直接回答；显式请求仍可为复杂任务开启思考。
            kwargs["extra_body"] = {"thinking": {"type": "disabled"},
                                    **kwargs.get("extra_body", {})}
        original_messages = list(kwargs["messages"])
        retry_of = None
        for attempt in range(2):
            request_id = record_request(
                str(trace_label), model=kwargs["model"], messages=kwargs["messages"],
                response_format=kwargs["response_format"],
                parameters={key: value for key, value in kwargs.items()
                            if key not in {"model", "messages", "response_format"}})
            started = time.perf_counter()
            response = self._inner.with_raw_response.parse(*args, **kwargs)
            payload = json.loads(response.text)
            choice = payload["choices"][0]
            record_call(str(trace_label), raw_response=choice["message"].get("content") or "",
                        metadata={"client": "openai_compatible", "request_id": request_id,
                                  "provider_request_id": payload.get("id"),
                                  "finish_reason": choice.get("finish_reason"),
                                  "usage": payload.get("usage"),
                                  "attempt": attempt, "retry_of": retry_of,
                                  "latency_ms": (time.perf_counter() - started) * 1000.0})
            try:
                return response.parse()
            except ValidationError as error:
                record_parse_error(request_id, error, attempt=attempt)
                if attempt == 1:
                    raise
                retry_of = request_id
                kwargs["messages"] = original_messages + [{
                    "role": "user", "content": (
                        "The previous response failed JSON/schema validation. Regenerate one valid "
                        "JSON object matching the supplied schema using the same observation. "
                        "Preserve uncertainty where evidence is insufficient. Validation details: "
                        + json.dumps(error.errors(include_input=False, include_context=False), ensure_ascii=False)
                    ),
                }]


class _TracingChat:
    def __init__(self, inner) -> None:
        self.completions = _TracingParsedCompletions(inner.completions)


class _TracingBeta:
    def __init__(self, inner) -> None:
        self.chat = _TracingChat(inner.chat)


class TracingOpenAICompatibleClient:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.beta = _TracingBeta(inner.beta)


def get_client_and_model(vlm: str):
    # 统一 LLM 入口：VLN 上层使用 OpenAI-compatible parse 形式。
    # 离线 smoke 使用本文件内的保守 fallback，不依赖外部 LLM client。
    backend = (vlm or DEFAULT_VLM or "cognav").lower()
    if os.getenv("LLM_OFFLINE", "0") in ("1", "true", "True"):
        return CogNavOpenAICompatibleClient(), COGNAV_MODEL_NAME
    if backend == "cognav":
        return CogNavOpenAICompatibleClient(), COGNAV_MODEL_NAME

    if backend == "gemini":
        OpenAI = _openai_client_class()
        return (
            TracingOpenAICompatibleClient(OpenAI(
                api_key=require_gemini_key(),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )),
            GEMINI_MODEL_NAME,
        )
    if backend in ("self_hosted", "self-hosted", "ms_swift", "ms-swift", "qwen_local", "qwen-local"):
        from llm_utils.self_hosted_llm_adapter import SelfHostedOpenAICompatibleClient

        model = (
            os.getenv("VLN_LVLM_MODEL")
            or os.getenv("STRIVE_LVLM_MODEL")
            or os.getenv("STRIVE_LVLM_SERVED_MODEL")
            or os.getenv("VLM_MODEL")
            or os.getenv("LLM_MODEL")
            or COGNAV_MODEL_NAME
        )
        return SelfHostedOpenAICompatibleClient.from_environment(), model
    if backend in ("ark", "openai_compatible", "openai-compatible"):
        OpenAI = _openai_client_class()
        api_key = os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        return (
            TracingOpenAICompatibleClient(OpenAI(api_key=api_key, base_url=base_url)),
            COGNAV_MODEL_NAME,
        )
    if backend == "openai":
        OpenAI = _openai_client_class()
        kwargs = {}
        if os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE_URL"):
            kwargs["base_url"] = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE_URL")
        return TracingOpenAICompatibleClient(OpenAI(**kwargs)), os.getenv("OPENAI_MODEL", "gpt-4o")
    raise ValueError(f"Invalid VLM: {vlm}")
