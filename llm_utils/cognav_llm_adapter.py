import json
import os
import re
import sys
from types import SimpleNamespace
from typing import Any, get_origin

from constants import COGNAV_MODEL_NAME, DEFAULT_VLM, GEMINI_MODEL_NAME, require_gemini_key


def _openai_client_class():
    from openai import OpenAI

    return OpenAI


def _add_cognav_to_path() -> None:
    cognav_root = os.getenv("COGNAV_OBJNAV_PATH")
    if not cognav_root:
        return
    if os.path.isdir(cognav_root) and cognav_root not in sys.path:
        sys.path.insert(0, cognav_root)


def _extract_json_object(text: str) -> dict[str, Any]:
    # STRIVE 调用方期望拿到结构化 JSON；模型偶尔会包 markdown，
    # 这里尽量抽取第一个 JSON object，再交给 Pydantic 做严格校验。
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
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


def _fallback_payload(response_format) -> dict[str, Any]:
    # 仅用于 smoke 测试：LLM 离线或返回空内容时给出保守默认值，
    # 让 Habitat/视觉/建图主链路能继续验证。
    payload = {}
    for name, field in response_format.model_fields.items():
        value = _fallback_value(field.annotation)
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
        try:
            parsed_payload = _extract_json_object(content)
        except Exception:
            if os.getenv("STRIVE_LLM_FALLBACK", "0") != "1":
                raise
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


def get_client_and_model(vlm: str):
    # 统一 LLM 入口：STRIVE 上层仍使用 OpenAI-compatible parse 形式，
    # 实际请求由 CogNav_ObjNav/utils/llm_client.py 负责处理 Ark/OpenAI/离线模式。
    backend = (vlm or DEFAULT_VLM or "cognav").lower()
    if backend == "cognav":
        return CogNavOpenAICompatibleClient(), COGNAV_MODEL_NAME

    if backend == "gemini":
        OpenAI = _openai_client_class()
        return (
            OpenAI(
                api_key=require_gemini_key(),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            GEMINI_MODEL_NAME,
        )
    if backend == "openai":
        OpenAI = _openai_client_class()
        return OpenAI(), "gpt-4o"
    raise ValueError(f"Invalid VLM: {vlm}")
