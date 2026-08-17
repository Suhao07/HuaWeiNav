"""Validated ownership contract for a real robot motion handoff.

The contract is deliberately independent of ROS.  It records which external
controller owns the waypoint and velocity interfaces, which safety limits have
been reviewed, and whether VLN is authorized to submit live goals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Union


class ControllerContractError(ValueError):
    """Raised when a controller contract is missing or fails a safety gate."""


@dataclass(frozen=True)
class ControllerContract:
    """Represent one parsed, robot-specific lower-controller contract.

    Args:
        source_path: Filesystem path of the YAML contract used for the run.
        values: The ``controller_contract`` mapping from the YAML document.
    """

    source_path: str
    values: Mapping[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Return a nested contract value without exposing YAML traversal.

        Args:
            *keys: Mapping keys from the contract root to the requested value.
            default: Value returned when a key is missing.

        Returns:
            The nested value, or ``default`` when the path is absent.
        """

        current: Any = self.values
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                return default
            current = current[key]
        return current

    @property
    def is_approved(self) -> bool:
        """Return whether the contract has received explicit human approval."""

        return str(self.get("approval_status", default="")).strip().lower() == "approved"


def load_controller_contract(path: Union[str, Path]) -> ControllerContract:
    """Load a robot contract from a YAML file.

    Args:
        path: YAML path containing a top-level ``controller_contract`` mapping.

    Returns:
        Parsed immutable contract wrapper.

    Raises:
        ControllerContractError: If the path, YAML parser, or schema is invalid.
    """

    contract_path = Path(path).expanduser()
    if not str(path).strip():
        raise ControllerContractError("controller contract path is empty")
    if not contract_path.is_file():
        raise ControllerContractError(f"controller contract does not exist: {contract_path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise ControllerContractError("PyYAML is required to load controller contracts") from exc
    try:
        document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ControllerContractError(f"unable to parse controller contract: {contract_path}") from exc
    values = document.get("controller_contract") if isinstance(document, Mapping) else None
    if not isinstance(values, Mapping):
        raise ControllerContractError(
            f"controller contract must contain a mapping named controller_contract: {contract_path}"
        )
    return ControllerContract(source_path=str(contract_path), values=values)


def validate_controller_contract(
    contract: ControllerContract,
    *,
    waypoint_topic: str,
    world_frame: str,
    action_name: str,
    require_approved: bool = True,
) -> None:
    """Validate safety gates and runtime interface compatibility.

    Args:
        contract: Parsed robot-specific contract.
        waypoint_topic: Topic the VLN/SysNav motion server will own.
        world_frame: Frame expected by the lower planner.
        action_name: Task-level action used by the high-level runtime.
        require_approved: Require explicit approval for live handoff.

    Raises:
        ControllerContractError: If any required gate or interface is invalid.
    """

    errors: list[str] = []
    if require_approved and not contract.is_approved:
        errors.append("approval_status must be approved")
    if contract.get("allow_strive_waypoint_handoff", default=False) is not True:
        errors.append("allow_strive_waypoint_handoff must be true")
    if contract.get("cmd_vel_direct_publish", default=True) is not False:
        errors.append("cmd_vel_direct_publish must be false")
    if contract.get("final_cmd_vel_owner", default="") != "safety_velocity_mux":
        errors.append("final_cmd_vel_owner must be safety_velocity_mux")
    if contract.get("sensor_watchdog_required", default=False) is not True:
        errors.append("sensor_watchdog_required must be true")

    expected_waypoint = _normalize_topic(waypoint_topic)
    contract_waypoint = _normalize_topic(str(contract.get("waypoint", "topic", default="")))
    if not contract_waypoint:
        errors.append("waypoint.topic must be configured")
    elif contract_waypoint != expected_waypoint:
        errors.append(f"waypoint.topic={contract_waypoint!r} does not match runtime {expected_waypoint!r}")

    expected_frame = str(world_frame or "").strip()
    contract_frame = str(contract.get("waypoint", "frame_id", default="")).strip()
    if not contract_frame:
        errors.append("waypoint.frame_id must be configured")
    elif contract_frame != expected_frame:
        errors.append(f"waypoint.frame_id={contract_frame!r} does not match runtime {expected_frame!r}")

    configured_action = str(contract.get("waypoint", "action_name", default="")).strip()
    if not configured_action:
        errors.append("waypoint.action_name must be configured")
    elif configured_action != str(action_name or "").strip():
        errors.append(f"waypoint.action_name={configured_action!r} does not match runtime {action_name!r}")

    for key in ("message_type", "action_message_type"):
        if not str(contract.get("waypoint", key, default="")).strip():
            errors.append(f"waypoint.{key} must be configured")
    for key in ("xy_goal_tolerance_m", "timeout_s"):
        if _positive_number(contract.get("waypoint", key, default=0.0)) is False:
            errors.append(f"waypoint.{key} must be positive")
    if _nonnegative_number(contract.get("waypoint", "yaw_goal_tolerance_rad", default=0.0)) is False:
        errors.append("waypoint.yaw_goal_tolerance_rad must be non-negative")

    if contract.get("feedback", "action_result_authoritative", default=False) is not True:
        errors.append("feedback.action_result_authoritative must be true")
    for key in ("reached_value", "blocked_value", "timeout_value"):
        if not str(contract.get("feedback", key, default="")).strip():
            errors.append(f"feedback.{key} must be configured")

    for key in (
        "max_linear_speed_mps",
        "max_angular_speed_rps",
        "max_linear_accel_mps2",
        "max_angular_accel_rps2",
        "command_watchdog_timeout_s",
    ):
        if _positive_number(contract.get("safety", key, default=0.0)) is False:
            errors.append(f"safety.{key} must be positive")
    if contract.get("safety", "emergency_stop_verified", default=False) is not True:
        errors.append("safety.emergency_stop_verified must be true")
    if not str(contract.get("safety", "emergency_stop_topic", default="")).strip():
        errors.append("safety.emergency_stop_topic must be configured")
    if not str(contract.get("safety", "manual_takeover_topic", default="")).strip():
        errors.append("safety.manual_takeover_topic must be configured")
    if not str(contract.get("safety", "manual_takeover_procedure", default="")).strip():
        errors.append("safety.manual_takeover_procedure must be documented")

    if errors:
        raise ControllerContractError(
            f"invalid controller contract {contract.source_path}: " + "; ".join(errors)
        )


def _normalize_topic(value: str) -> str:
    """Normalize leading/trailing separators for topic comparisons."""

    text = str(value or "").strip()
    if not text:
        return ""
    return "/" + text.strip("/")


def _positive_number(value: Any) -> bool:
    """Return whether a value is finite and strictly greater than zero."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _nonnegative_number(value: Any) -> bool:
    """Return whether a value is finite and non-negative."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0
