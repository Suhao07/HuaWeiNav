from __future__ import annotations

import json
import os
import time
from uuid import uuid4
from collections import defaultdict
from typing import Any


_COUNTS: dict[str, int] = defaultdict(int)
_CACHE_HITS: dict[str, int] = defaultdict(int)
_RAW_INDEX = 0


def record_request(kind: str, *, model: str, messages, response_format, parameters: dict) -> str:
    """Persist model-visible input before dispatch and return its request ID.

    Args:
        kind: Semantic operation recorded by the caller.
        model: Configured provider model name.
        messages: Actual messages, including inline image bytes when present.
        response_format: Structured response model supplied to the adapter.
        parameters: Generation options, excluding credentials and client settings.

    Returns:
        Unique ID used to associate the response with this exact input.
    """
    request_id = uuid4().hex
    trace_dir = os.getenv("STRIVE_LVLM_TRACE_DIR") or "outputs/model_requests"
    os.makedirs(trace_dir, exist_ok=True)
    schema_builder = getattr(response_format, "model_json_schema", None)
    schema = schema_builder() if schema_builder is not None else response_format.schema()
    payload = {"request_id": request_id, "kind": kind, "time": time.time(),
               "model": model, "messages": list(messages), "response_schema": schema,
               "parameters": parameters}
    path = os.path.join(trace_dir, f"request_{request_id}.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return request_id


def reset_counts() -> None:
    global _RAW_INDEX
    _COUNTS.clear()
    _CACHE_HITS.clear()
    _RAW_INDEX = 0


def set_trace_dir(path: str) -> None:
    if path:
        os.environ["STRIVE_LVLM_TRACE_DIR"] = path
        os.makedirs(path, exist_ok=True)


def record_cache_hit(kind: str) -> None:
    _CACHE_HITS[str(kind or "unknown")] += 1


def record_call(kind: str, *, raw_response: str = "", metadata: dict[str, Any] | None = None) -> None:
    """Record one real LVLM/LLM request.

    此记录保存计数、原始返回文本和 request_id。完整模型输入由 record_request
    在发送前单独保存，响应通过 request_id 关联。
    """

    global _RAW_INDEX
    kind = str(kind or "unknown")
    _COUNTS[kind] += 1
    trace_dir = os.getenv("STRIVE_LVLM_TRACE_DIR", "")
    if not trace_dir:
        return
    os.makedirs(trace_dir, exist_ok=True)
    _RAW_INDEX += 1
    payload = {
        "index": _RAW_INDEX,
        "kind": kind,
        "time": time.time(),
        "metadata": dict(metadata or {}),
        "raw_response": str(raw_response or ""),
    }
    path = os.path.join(trace_dir, f"{_RAW_INDEX:04d}_{kind}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def counts() -> dict[str, Any]:
    return {
        "calls": dict(sorted(_COUNTS.items())),
        "cache_hits": dict(sorted(_CACHE_HITS.items())),
        "total_calls": int(sum(_COUNTS.values())),
        "total_cache_hits": int(sum(_CACHE_HITS.values())),
    }


def counts_compact() -> str:
    return json.dumps(counts(), ensure_ascii=False, sort_keys=True)
