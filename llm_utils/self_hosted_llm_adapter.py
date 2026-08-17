"""HTTP client adapter for a self-hosted OpenAI-compatible LVLM.

The model server remains provider-neutral and exposes ordinary chat
completions. This adapter owns VLN's Pydantic schema contract because not all
OpenAI-compatible servers implement ``beta.chat.completions.parse``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional

from llm_utils.lvlm_call_tracker import record_call
from llm_utils.structured_output import (
    extract_json_object,
    fallback_payload,
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
            **kwargs: Supported generation options plus VLN ``trace_label``.

        Returns:
            Completion-like object whose message exposes ``parsed`` and raw
            ``content`` fields.

        Raises:
            Exception: If transport/validation fails and conservative fallback
                is disabled with ``STRIVE_LLM_PARSE_FALLBACK=0``.
        """

        trace_label = str(
            kwargs.pop("trace_label", "")
            or getattr(response_format, "__name__", "")
            or "self_hosted_parse"
        )
        normalized = inject_json_schema(messages, response_format)
        last_content = ""
        last_error: Optional[Exception] = None

        for attempt in range(self.parse_retries + 1):
            request_messages = normalized if attempt == 0 else _repair_messages(normalized, last_content)
            started = time.perf_counter()
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=request_messages,
                    **_generation_kwargs(kwargs),
                )
                last_content = _message_text(completion)
                record_call(
                    trace_label,
                    raw_response=last_content,
                    metadata={
                        "client": "self_hosted_openai_compatible",
                        "base_url": self.base_url,
                        "model": model,
                        "response_format": getattr(response_format, "__name__", str(response_format)),
                        "attempt": attempt + 1,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                    },
                )
                payload = extract_json_object(last_content)
                parsed = validate_response_model(response_format, payload)
                return _parsed_completion(parsed, last_content)
            except Exception as exc:
                last_error = exc
                if attempt >= self.parse_retries:
                    break

        if not _fallback_enabled():
            assert last_error is not None
            raise last_error

        # 解析或服务失败只能得到保守结果，不能因为 fallback 进入 ACCEPT/STOP。
        parsed = validate_response_model(response_format, fallback_payload(response_format))
        return _parsed_completion(parsed, last_content)


def _generation_kwargs(values: Mapping[str, Any]) -> dict[str, Any]:
    """Select generation fields supported by ms-swift chat completions."""

    output: dict[str, Any] = {"temperature": float(values.get("temperature", 0.0))}
    for key in ("max_tokens", "top_p", "seed", "stop"):
        value = values.get(key)
        if value is not None:
            output[key] = value
    return output


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


def _repair_messages(messages: list[dict[str, Any]], content: str) -> list[dict[str, Any]]:
    """Build one bounded schema-repair turn without resending image bytes twice."""

    # 第二轮只纠正结构化文本。保留包含 JSON schema 的 system message，
    # 不再上传原始图像；上一轮回答已经携带待修复的语义内容。
    repaired = [
        dict(message)
        for message in messages
        if message.get("role") == "system" and isinstance(message.get("content"), str)
    ]
    repaired.append({"role": "assistant", "content": str(content or "")[:8000]})
    repaired.append(
        {
            "role": "user",
            "content": (
                "The previous response did not satisfy the required JSON schema. "
                "Return only one corrected JSON object. Do not add explanation."
            ),
        }
    )
    return repaired


def _parsed_completion(parsed: Any, content: str) -> Any:
    """Build the completion shape expected by existing VLN call sites."""

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))]
    )


def _fallback_enabled() -> bool:
    """Return whether structured failures should yield conservative output."""

    return os.getenv("STRIVE_LLM_PARSE_FALLBACK", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
