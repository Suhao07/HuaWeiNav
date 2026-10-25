"""Serialize runtime facts and append decisions to JSONL."""
from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from real_robot.contracts import RuntimeDecision, ViewpointResult

class RuntimeDecisionJsonlWriter:
    """Append real-robot runtime decisions to a JSONL log file."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the writer and create parent directories.

        Args:
            path: Destination JSONL file path.
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: RuntimeDecision) -> Dict[str, Any]:
        """Append one decision to disk.

        Args:
            decision: Runtime decision emitted by `SysNavInstructionRuntime`.

        Returns:
            JSON-friendly dictionary that was written.
        """

        payload = runtime_decision_to_dict(decision)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return payload


def runtime_decision_to_dict(decision: RuntimeDecision) -> Dict[str, Any]:
    """Return a JSON-friendly runtime decision dictionary.

    Args:
        decision: Runtime decision emitted by the real-robot loop.

    Returns:
        JSON-serializable dictionary.
    """

    return _json_ready(decision)


def runtime_viewpoint_result_to_dict(result: Optional[ViewpointResult]) -> Dict[str, Any]:
    """Return a JSON-friendly viewpoint result dictionary."""

    if result is None:
        return {}
    return _json_ready(result)



def _json_ready(value: Any) -> Any:
    """Convert dataclasses, enums, tuples, and paths into JSON-ready values."""

    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
