#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_strive_real_robot_acceptance}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m py_compile \
  real_robot/contracts.py \
  real_robot/sysnav_ros_adapters.py \
  real_robot/sysnav_goal_resolver.py \
  real_robot/sysnav_runtime.py \
  real_robot/observation_cache.py \
  real_robot/waypoint_adapter.py \
  real_robot/motion_safety.py \
  real_robot/control/controller_contract.py \
  real_robot/ros_motion_action.py \
  planning/semantic_snapshot_context.py \
  real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/projection_config.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/strive_sysnav_bringup/instruction_runtime_node.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/strive_sysnav_bringup/waypoint_adapter_node.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/launch/strive_instruction_runtime.launch.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/launch/sysnav_detection_mapping.launch.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/launch/waypoint_adapter.launch.py \
  real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/motion_server.py \
  real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/motion_hil.py \
  real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/safety_velocity_mux.py \
  real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/lower_bag_probe.py

PYTHONPATH=. "${PYTHON_BIN}" -m pytest -q \
  tests/test_real_robot_acceptance.py \
  tests/test_sysnav_ros_adapters.py \
  tests/test_sysnav_goal_resolver.py \
  tests/test_sysnav_runtime.py \
  tests/test_semantic_snapshot_context.py \
  tests/test_observation_cache.py \
  tests/test_real_robot_contracts.py \
  tests/test_motion_action_contract.py \
  tests/test_motion_safety.py \
  tests/test_motion_safety_state.py \
  tests/test_controller_contract.py \
  tests/test_sysnav_migration_boundary.py \
  tests/test_real_robot_projection_config.py \
  tests/test_waypoint_adapter.py

echo "real-robot offline acceptance checks passed"
