"""Shared structured-output utilities for VLN model providers.

The navigation stack uses Pydantic response models at every semantic boundary.
Provider adapters may implement structured decoding differently, but they must
all inject the same schema, validate the same payload, and fail conservatively.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable, Mapping, get_origin


def inject_json_schema(
    messages: Iterable[Mapping[str, Any]],
    response_format: Any,
) -> list[dict[str, Any]]:
    """Append a Pydantic JSON schema contract to the system message.

    Args:
        messages: OpenAI-compatible message records. Multimodal content is kept
            unchanged.
        response_format: Pydantic v1/v2 model class describing the response.

    Returns:
        A detached message list containing the strict JSON-only instruction.

    Raises:
        TypeError: If ``response_format`` does not expose a JSON schema API.
    """

    schema_builder = getattr(response_format, "model_json_schema", None)
    if schema_builder is not None:
        schema = schema_builder()
    else:
        schema_builder = getattr(response_format, "schema", None)
        if schema_builder is None:
            raise TypeError("response_format must be a Pydantic model class")
        schema = schema_builder()

    schema_prompt = (
        "Return only one JSON object that conforms to this JSON schema. "
        "Do not include markdown, analysis, or extra text.\n"
        f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    )
    normalized = [dict(message) for message in messages]
    if normalized and normalized[0].get("role") == "system":
        system_content = normalized[0].get("content", "")
        if not isinstance(system_content, str):
            raise TypeError("the system message content must be text")
        normalized[0] = {
            **normalized[0],
            "content": f"{system_content}\n\n{schema_prompt}".strip(),
        }
    else:
        normalized.insert(0, {"role": "system", "content": schema_prompt})
    return normalized


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON-like object from a model response.

    The parser accepts a small set of common formatting mistakes: Markdown
    fences, trailing commas, bare object keys, and Python literal booleans. It
    never evaluates arbitrary code and still requires the final value to be a
    dictionary.

    Args:
        text: Raw model response.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no object-like payload can be parsed.
    """

    for candidate in _json_candidates(text):
        try:
            return _loads_json_like(candidate)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError(f"LLM response is not JSON: {str(text)[:500]}")


def fallback_payload(response_format: Any) -> dict[str, Any]:
    """Build a conservative payload for an unavailable structured provider.

    Args:
        response_format: Pydantic v1/v2 response model class.

    Returns:
        Field-shaped fallback data. Decisions are ``uncertain`` and booleans
        are false so a transport or parse failure cannot authorize navigation
        success.
    """

    payload: dict[str, Any] = {}
    fields = getattr(response_format, "model_fields", None)
    if fields is None:
        fields = getattr(response_format, "__fields__", {})
    if not fields:
        for name, annotation in getattr(response_format, "__annotations__", {}).items():
            value = _fallback_scalar_for_name(name, annotation)
            payload[name] = [] if value is None else value
        return payload
    for name, field in fields.items():
        annotation = getattr(field, "annotation", None) or getattr(field, "outer_type_", str)
        value = _fallback_scalar_for_name(name, annotation)
        payload[name] = [] if value is None else value
    return payload


def validate_response_model(response_format: Any, payload: dict[str, Any]) -> Any:
    """Validate a structured payload across Pydantic v1 and v2.

    Args:
        response_format: Pydantic model class.
        payload: Parsed object returned by the provider.

    Returns:
        Validated Pydantic model instance.

    Raises:
        Exception: Propagates the response model's validation error.
    """

    validator = getattr(response_format, "model_validate", None)
    if validator is not None:
        return validator(payload)
    parser = getattr(response_format, "parse_obj", None)
    if parser is not None:
        return parser(payload)
    return response_format(**payload)


def _strip_json_fence(text: str) -> str:
    """Remove one optional Markdown JSON fence."""

    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _json_candidates(text: str) -> list[str]:
    """Return whole-response and first-object parse candidates."""

    normalized = _strip_json_fence(text)
    candidates = [normalized]
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        candidates.append(normalized[start:end + 1])
    return candidates


def _loads_json_like(candidate: str) -> dict[str, Any]:
    """Parse one narrowly repaired JSON-like object candidate."""

    value = candidate.strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 只修复对象结构中的常见格式问题，不对自由文本做全局重写。
    repaired = re.sub(r",\s*([}\]])", r"\1", value)
    repaired = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)',
        r'\1"\2"\3',
        repaired,
    )
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    json_repaired = re.sub(r"\bTrue\b", "true", repaired)
    json_repaired = re.sub(r"\bFalse\b", "false", json_repaired)
    json_repaired = re.sub(r"\bNone\b", "null", json_repaired)
    try:
        parsed = json.loads(json_repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    parsed = ast.literal_eval(repaired)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("candidate is not a JSON object")


def _fallback_value(annotation: Any) -> Any:
    """Return a conservative generic value for one annotation."""

    origin = get_origin(annotation)
    if origin in (list, tuple):
        return []
    if origin is dict or annotation is dict:
        return {}
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is str:
        return "fallback"
    return None


def _fallback_scalar_for_name(name: str, annotation: Any) -> Any:
    """Return a field-aware conservative scalar."""

    if annotation is str:
        lowered = name.lower()
        if lowered in ("res", "result", "label", "object", "target"):
            return "unknown"
        if lowered == "decision":
            return "uncertain"
        if lowered in ("reason", "explanation", "view_feedback", "preferred_view_goal"):
            return "fallback"
    return _fallback_value(annotation)
