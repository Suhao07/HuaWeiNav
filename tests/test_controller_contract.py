from pathlib import Path

import pytest

from real_robot.control.controller_contract import (
    ControllerContractError,
    load_controller_contract,
    validate_controller_contract,
)


VALID_CONTRACT = """
controller_contract:
  approval_status: approved
  approval_reference: review-001
  owner: platform-team
  allow_strive_waypoint_handoff: true
  cmd_vel_direct_publish: false
  final_cmd_vel_owner: safety_velocity_mux
  sensor_watchdog_required: true
  waypoint:
    topic: /way_point
    message_type: geometry_msgs/PointStamped
    frame_id: map
    action_name: /strive/execute_waypoint
    action_message_type: strive_motion_msgs/action/ExecuteWaypoint
    xy_goal_tolerance_m: 0.35
    yaw_goal_tolerance_rad: 0.2
    timeout_s: 60.0
  feedback:
    status_topic: /local_planner/status
    message_type: std_msgs/String
    action_result_authoritative: true
    reached_value: REACHED
    blocked_value: BLOCKED
    timeout_value: TIMEOUT
  safety:
    max_linear_speed_mps: 0.3
    max_angular_speed_rps: 0.8
    max_linear_accel_mps2: 0.4
    max_angular_accel_rps2: 0.8
    command_watchdog_timeout_s: 0.25
    emergency_stop_topic: /platform/estop_active
    emergency_stop_verified: true
    manual_takeover_topic: /platform/manual_takeover
    manual_takeover_procedure: operator takeover tested
"""


def test_valid_controller_contract_passes(tmp_path: Path) -> None:
    path = tmp_path / "controller.yaml"
    path.write_text(VALID_CONTRACT, encoding="utf-8")

    contract = load_controller_contract(path)
    validate_controller_contract(
        contract,
        waypoint_topic="way_point",
        world_frame="map",
        action_name="/strive/execute_waypoint",
    )

    assert contract.is_approved
    assert contract.get("safety", "max_linear_speed_mps") == 0.3


def test_unapproved_controller_contract_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "controller.yaml"
    path.write_text(VALID_CONTRACT.replace("approval_status: approved", "approval_status: unapproved"), encoding="utf-8")

    with pytest.raises(ControllerContractError, match="approval_status"):
        validate_controller_contract(
            load_controller_contract(path),
            waypoint_topic="/way_point",
            world_frame="map",
            action_name="/strive/execute_waypoint",
        )


def test_controller_contract_rejects_interface_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "controller.yaml"
    path.write_text(VALID_CONTRACT, encoding="utf-8")

    with pytest.raises(ControllerContractError, match="waypoint.topic"):
        validate_controller_contract(
            load_controller_contract(path),
            waypoint_topic="/another_waypoint",
            world_frame="map",
            action_name="/strive/execute_waypoint",
        )


def test_missing_controller_contract_fails_before_motion() -> None:
    with pytest.raises(ControllerContractError, match="does not exist"):
        load_controller_contract("/tmp/strive-contract-that-does-not-exist.yaml")
