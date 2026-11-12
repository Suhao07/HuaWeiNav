"""HTTP client adapter for a self-hosted OpenAI-compatible LVLM.

The model server remains provider-neutral and exposes ordinary chat
completions. This adapter owns VLN's Pydantic schema contract because not all
OpenAI-compatible servers implement ``beta.chat.completions.parse``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional

from llm_utils.lvlm_call_tracker import record_response, record_request, record_parse_error
from llm_utils.structured_output import (
    StructuredOutputError,
    extract_json_object,
    inject_json_schema,
    validate_response_model,
)


class SelfHostedOpenAICompatibleClient:
    """Expose VLN's structured parse API over an HTTP chat-completion service.

    Args:
        base_url: OpenAI-compatible API root, including the ``/v1`` suffix.
        api_key: Bearer token configured by the model server. ``EMPTY`` is
            accepted for trusted, authentication-free development networks.
        timeout_s: Per-request HTTP timeout.
        transport_retries: Retry count delegated to the OpenAI HTTP client.
        parse_retries: Additional model calls allowed when returned text does
            not satisfy the requested Pydantic schema.
        client: Optional injected OpenAI-like client used by tests.

        Raises:
            ValueError: If ``base_url`` is empty.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        timeout_s: float = 45.0,
        transport_retries: int = 2,
        parse_retries: int = 1,
        client: Optional[Any] = None,
    ) -> None:
        normalized_url = str(base_url or "").strip().rstrip("/")
        if not normalized_url:
            raise ValueError("self-hosted LVLM base_url is required")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=str(api_key or "EMPTY"),
                base_url=normalized_url,
                timeout=float(timeout_s),
                max_retries=max(0, int(transport_retries)),
            )
        parsed_chat = _SelfHostedParsedChat(
            client=client,
            base_url=normalized_url,
            parse_retries=max(0, int(parse_retries)),
        )
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parsed_chat.parse))
        )

    @classmethod
    def from_environment(cls) -> "SelfHostedOpenAICompatibleClient":
        """Build the robot-side HTTP client from deployment environment variables.

        Returns:
            Configured self-hosted client.

        Raises:
            ValueError: If neither self-hosted nor generic base URL is set.
        """

        settings = HttpVlmSettings.from_environment()
        return cls(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout_s=settings.timeout_s,
            transport_retries=settings.transport_retries,
            parse_retries=settings.parse_retries,
        )


@dataclass(frozen=True)
class HttpVlmSettings:
    """Configuration for the robot-to-server HTTP VLM boundary.

    Args:
        base_url: OpenAI-compatible API root, including ``/v1``.
        api_key: Bearer token sent to the remote service.
        timeout_s: Per-request timeout in seconds.
        transport_retries: HTTP transport retries performed by the SDK.
        parse_retries: Additional text-only retries for invalid JSON output.
    """

    base_url: str
    api_key: str = "EMPTY"
    timeout_s: float = 45.0
    transport_retries: int = 2
    parse_retries: int = 1

    @classmethod
    def from_environment(cls) -> "HttpVlmSettings":
        """Resolve preferred VLN variables and compatibility aliases.

        Returns:
            Normalized HTTP client settings.

        Raises:
            ValueError: If no remote API base URL is configured.
        """

        base_url = _first_env(
            "VLN_LVLM_BASE_URL",
            "STRIVE_LVLM_BASE_URL",
            "VLM_API_BASE_URL",
            "LLM_API_BASE_URL",
            "OPENAI_BASE_URL",
        )
        if not base_url:
            raise ValueError(
                "remote VLM base URL is required; set VLN_LVLM_BASE_URL "
                "or the STRIVE_LVLM_BASE_URL compatibility variable"
            )
        return cls(
            base_url=base_url,
            api_key=_first_env(
                "VLN_LVLM_API_KEY",
                "STRIVE_LVLM_API_KEY",
                "VLM_API_KEY",
                "OPENAI_API_KEY",
            )
            or "EMPTY",
            timeout_s=float(_first_env("VLN_LVLM_TIMEOUT_S", "STRIVE_LVLM_TIMEOUT_S") or 45),
            transport_retries=int(
                _first_env(
                    "VLN_LVLM_TRANSPORT_RETRIES",
                    "STRIVE_LVLM_TRANSPORT_RETRIES",
                )
                or 2
            ),
            parse_retries=int(
                _first_env("VLN_LVLM_PARSE_RETRIES", "STRIVE_LVLM_PARSE_RETRIES") or 1
            ),
        )


class _SelfHostedParsedChat:
    """Translate one structured parse request into chat completion calls."""

    def __init__(self, *, client: Any, base_url: str, parse_retries: int) -> None:
        self.client = client
        self.base_url = base_url
        self.parse_retries = parse_retries

    def parse(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, Any]],
        response_format: Any,
        **kwargs: Any,
    ) -> Any:
        """Return an OpenAI-beta-like completion with a validated model.

        Args:
            model: Served model name advertised by the remote API.
            messages: OpenAI-compatible text or multimodal messages.
            response_format: Pydantic response model class.
            **kwargs: Supported generation options, VLN ``trace_label``, and
                ``parse_retries`` overriding the configured repair budget.
                Zero disables repair calls.
                ``on_retry`` charges the caller's budget before each additional request.

        Returns:
            Completion-like object whose message exposes ``parsed`` and raw
            ``content`` fields.

        Raises:
            StructuredOutputError: If response validation fails with zero retries.
            Exception: Transport errors propagate unchanged. Validation errors
                propagate after the repair budget is exhausted.
        """

        trace_label = str(
            kwargs.pop("trace_label", "")
            or getattr(response_format, "__name__", "")
            or "self_hosted_parse"
        )
        parse_retries = max(0, int(kwargs.pop("parse_retries", self.parse_retries)))
        on_retry = kwargs.pop("on_retry", None)
        normalized = inject_json_schema(messages, response_format)
        json_schema = _response_json_schema(response_format) if _guided_json_enabled() else None
        fields = _response_json_schema(response_format)["properties"]
        last_content = ""
        last_error: Optional[Exception] = None

        generation_kwargs = _generation_kwargs(kwargs)
        for attempt in range(parse_retries + 1):
            if attempt and on_retry is not None:
                on_retry()
            request_messages = normalized if attempt == 0 else _repair_messages(normalized)
            started = time.perf_counter()
            if json_schema is not None:
                generation_kwargs["extra_body"] = {"guided_json": json_schema}
            request_id = record_request(trace_label, model=model, messages=request_messages,
                                        response_format=response_format, parameters=generation_kwargs)
            completion = self.client.chat.completions.create(
                model=model,
                messages=request_messages,
                **generation_kwargs,
            )
            last_content = _message_text(completion)
            finish_reason = getattr(completion.choices[0], "finish_reason", None)
            usage = getattr(completion, "usage", None)
            record_response(
                trace_label,
                raw_response=last_content,
                metadata={
                    "client": "self_hosted_openai_compatible",
                    "request_id": request_id,
                    "base_url": self.base_url,
                    "model": model,
                    "response_format": getattr(response_format, "__name__", str(response_format)),
                    "attempt": attempt + 1,
                    "finish_reason": finish_reason,
                    "usage": usage.model_dump() if usage is not None else None,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                },
            )
            if getattr(completion.choices[0].message, "refusal", None) or finish_reason == "content_filter":
                raise RuntimeError("Model provider refused the structured response")
            try:
                if finish_reason == "length":
                    generation_kwargs["max_tokens"] = 2 * generation_kwargs.get("max_tokens", 1024)
                    raise ValueError("Output reached its token budget; regenerate the complete JSON")
                payload = json.loads(last_content) if json_schema is not None else extract_json_object(last_content)
                if not isinstance(payload, dict) or not set(payload).intersection(fields):
                    raise ValueError("Response must contain model fields as a JSON instance")
                parsed = validate_response_model(response_format, payload)
            except ValueError as exc:
                record_parse_error(request_id, exc, attempt=attempt + 1)
                last_error = exc
                if attempt >= parse_retries:
                    break
            else:
                return _parsed_completion(parsed, last_content)

        assert last_error is not None
        if parse_retries == 0:
            raise StructuredOutputError(str(last_error), raw_content=last_content) from last_error
        raise last_error


def _generation_kwargs(values: Mapping[str, Any]) -> dict[str, Any]:
    """Select generation fields supported by ms-swift chat completions."""

    output: dict[str, Any] = {"temperature": float(values.get("temperature", 0.0))}
    for key in ("max_tokens", "top_p", "seed", "stop"):
        value = values.get(key)
        if value is not None:
            output[key] = value
    return output


def _response_json_schema(response_format: Any) -> dict[str, Any]:
    """Build the JSON schema sent to a guided-decoding provider."""

    schema_builder = getattr(response_format, "model_json_schema", None)
    if schema_builder is None:
        schema_builder = getattr(response_format, "schema", None)
    if schema_builder is None:
        raise TypeError("response_format must be a Pydantic model class")
    schema = schema_builder()
    if not isinstance(schema, dict):
        raise TypeError("response_format JSON schema must be a dictionary")
    # 服务端必须生成完整字段，模型默认值仅供内部对象构造使用。
    schema["required"] = list(schema["properties"])
    schema["additionalProperties"] = False
    return schema


def _guided_json_enabled() -> bool:
    """Return whether the remote provider should enforce the JSON schema."""

    return os.getenv("VLN_LVLM_GUIDED_JSON", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _first_env(*names: str) -> str:
    """Return the first non-empty environment value in precedence order."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _message_text(completion: Any) -> str:
    """Extract text from one OpenAI-compatible completion."""

    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "") if isinstance(item, dict) else getattr(item, "text", ""))
            for item in content
        )
    return str(content or "")


def _repair_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Regenerate from the same complete observation and candidate context."""

    # 不回灌截断的重复文本，避免将退化输出变成下一次生成的示例。
    return [*messages, {
        "role": "user",
        "content": (
            "The previous response did not complete the required JSON schema. "
            "Regenerate one complete corrected JSON object from the same observation. "
            "Keep field explanations brief and non-repetitive, preserve uncertainty, "
            "and include every required field. Return only JSON."
        ),
    }]


def _parsed_completion(parsed: Any, content: str) -> Any:
    """Build the completion shape expected by existing VLN call sites."""

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))]
    )
