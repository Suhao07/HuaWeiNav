from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any


_COUNTS: dict[str, int] = defaultdict(int)
_CACHE_HITS: dict[str, int] = defaultdict(int)
_RAW_INDEX = 0


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

    这里只记录可审计元数据和原始返回文本，不保存 prompt 中的大图 base64
    避免日志膨胀。调用方仍在各自模块保存必要 evidence 图像。
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
