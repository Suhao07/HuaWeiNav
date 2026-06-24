#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_strive_real_robot_acceptance}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

python -m py_compile \
  real_robot/contracts.py \
  real_robot/sysnav_ros_adapters.py \
  real_robot/sysnav_runtime.py \
  real_robot/observation_cache.py \
  planning/semantic_snapshot_context.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/strive_sysnav_bringup/instruction_runtime_node.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/launch/strive_instruction_runtime.launch.py \
  real_robot/ros2_ws/src/strive_sysnav_bringup/launch/sysnav_detection_mapping.launch.py

PYTHONPATH=. pytest -q \
  tests/test_real_robot_acceptance.py \
  tests/test_sysnav_ros_adapters.py \
  tests/test_sysnav_runtime.py \
  tests/test_semantic_snapshot_context.py \
  tests/test_observation_cache.py \
  tests/test_real_robot_contracts.py

echo "real-robot offline acceptance checks passed"
