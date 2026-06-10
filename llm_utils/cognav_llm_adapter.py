import ast
import json
import os
import re
import sys
from types import SimpleNamespace
from typing import Any, get_origin

from constants import COGNAV_MODEL_NAME, DEFAULT_VLM, GEMINI_MODEL_NAME, require_gemini_key
from llm_utils.lvlm_call_tracker import record_call


def _openai_client_class():
    from openai import OpenAI

    return OpenAI


def _add_cognav_to_path() -> None:
    cognav_root = os.getenv("COGNAV_OBJNAV_PATH")
    if not cognav_root:
        return
    if os.path.isdir(cognav_root) and cognav_root not in sys.path:
        sys.path.insert(0, cognav_root)


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _json_candidates(text: str) -> list[str]:
    # STRIVE 调用方期望拿到结构化 JSON；模型偶尔会包 markdown，
    # 这里尽量抽取第一个 JSON object，再交给 Pydantic 做严格校验。
    text = _strip_json_fence(text)
    candidates = [text]

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    return candidates


def _loads_json_like(candidate: str) -> dict[str, Any]:
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 常见格式问题：尾逗号。保持修复范围很小，避免把自由文本误修成 JSON。
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
    if without_trailing_commas != candidate:
        try:
            return json.loads(without_trailing_commas)
        except json.JSONDecodeError:
            pass

    # 常见格式问题：{res: "book"} 这类裸 key。只修复 object key 位置，
    # 不对普通文本做全局替换，避免误伤解释内容。
    quoted_keys = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)',
        r'\1"\2"\3',
        without_trailing_commas,
    )
    if quoted_keys != without_trailing_commas:
        try:
            return json.loads(quoted_keys)
        except json.JSONDecodeError:
            pass
        json_booleans = re.sub(r"\bTrue\b", "true", quoted_keys)
        json_booleans = re.sub(r"\bFalse\b", "false", json_booleans)
        json_booleans = re.sub(r"\bNone\b", "null", json_booleans)
        try:
            return json.loads(json_booleans)
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(quoted_keys)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Ark/VLM 偶尔返回 Python dict 风格：单引号、True/False/None。
    # literal_eval 比 eval 安全，解析后仍要求是 dict，再交给 Pydantic 校验。
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    raise ValueError("candidate is not a JSON object")


def _extract_json_object(text: str) -> dict[str, Any]:
    for candidate in _json_candidates(text):
        try:
            return _loads_json_like(candidate)
        except Exception:
            continue
    raise ValueError(f"LLM response is not JSON: {text[:500]}")


def _fallback_value(annotation):
    origin = get_origin(annotation)
    if origin in (list, tuple):
        return []
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is str:
        return "fallback"
    return None


def _fallback_scalar_for_name(name: str, annotation):
    if annotation is str:
        lowered = name.lower()
        if lowered in ("res", "result", "label", "object", "target"):
            return "unknown"
        if lowered == "decision":
            return "uncertain"
        if lowered in ("reason", "explanation", "view_feedback", "preferred_view_goal"):
            return "fallback"
    return _fallback_value(annotation)


def _fallback_payload(response_format) -> dict[str, Any]:
    # 仅用于 smoke 测试：LLM 离线或返回空内容时给出保守默认值，
    # 让 Habitat/视觉/建图主链路能继续验证。
    payload = {}
    for name, field in response_format.model_fields.items():
        value = _fallback_scalar_for_name(name, field.annotation)
        payload[name] = [] if value is None else value
    return payload


class _CogNavParsedChat:
    def __init__(self) -> None:
        _add_cognav_to_path()
        from utils.llm_client import LLMClient

        self._client = LLMClient(apikey_file=os.getenv("COGNAV_APIKEY_FILE", "./apikey.txt"))

    def parse(self, model, messages, response_format, **kwargs):
        # CogNav LLMClient 不原生支持 OpenAI beta.parse。
        # 因此把 Pydantic schema 注入 system prompt，再把返回 JSON 校验回同一 schema。
        if os.getenv("LLM_OFFLINE", "0") in ("1", "true", "True"):
            parsed = response_format.model_validate(_fallback_payload(response_format))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=""))]
            )

        schema = response_format.model_json_schema()
        schema_prompt = (
            "Return only one JSON object that conforms to this JSON schema. "
            "Do not include markdown or extra text.\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        normalized = list(messages)
        if normalized and normalized[0].get("role") == "system":
            normalized[0] = {
                **normalized[0],
                "content": f"{normalized[0].get('content', '')}\n\n{schema_prompt}",
            }
        else:
            normalized.insert(0, {"role": "system", "content": schema_prompt})

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
                "response_format": getattr(response_format, "__name__", str(response_format)),
            },
        )
        try:
            parsed_payload = _extract_json_object(content)
        except Exception:
            # 生产运行中 VLM 偶发 JSON-like 格式错误不能中断整条导航链。
            # 默认对 parse error 做保守 fallback；如需调试原始异常，可显式关闭。
            if os.getenv("STRIVE_LLM_PARSE_FALLBACK", "1").lower() in ("0", "false", "no", "off"):
                raise
            print(
                "[CogNavLLMAdapter] JSON parse failed; using conservative fallback. "
                f"Raw response prefix={str(content)[:300]!r}"
            )
            parsed_payload = _fallback_payload(response_format)
        parsed = response_format.model_validate(parsed_payload)
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
        trace_label = kwargs.pop("trace_label", "parse")
        completion = self._inner.parse(*args, **kwargs)
        try:
            content = completion.choices[0].message.content
        except Exception:
            content = ""
        record_call(str(trace_label), raw_response=str(content or ""), metadata={"client": "openai_compatible"})
        return completion


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
    # 统一 LLM 入口：STRIVE 上层仍使用 OpenAI-compatible parse 形式，
    # 实际请求由 CogNav_ObjNav/utils/llm_client.py 负责处理 Ark/OpenAI/离线模式。
    backend = (vlm or DEFAULT_VLM or "cognav").lower()
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
    if backend == "openai":
        OpenAI = _openai_client_class()
        return TracingOpenAICompatibleClient(OpenAI()), "gpt-4o"
    raise ValueError(f"Invalid VLM: {vlm}")
