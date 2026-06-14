#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
OVERLAY_SETUP="${WS_DIR}/install/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 2
fi
if [[ ! -f "${OVERLAY_SETUP}" ]]; then
  echo "Overlay setup not found: ${OVERLAY_SETUP}" >&2
  echo "Run scripts/build_real_robot_ros_ws.sh first." >&2
  exit 2
fi

set +u
source "${ROS_SETUP}"
source "${OVERLAY_SETUP}"
set -u

MODEL_ARGS=()
if [[ -n "${SYSNAV_DETECTOR_MODEL_PATH:-}" ]]; then
  if [[ ! -f "${SYSNAV_DETECTOR_MODEL_PATH}" ]]; then
    echo "SYSNAV_DETECTOR_MODEL_PATH does not exist: ${SYSNAV_DETECTOR_MODEL_PATH}" >&2
    exit 2
  fi
  MODEL_ARGS+=("detector_model_path:=${SYSNAV_DETECTOR_MODEL_PATH}")
fi
if [[ -n "${SYSNAV_DETECTOR_MODEL_TYPE:-}" ]]; then
  MODEL_ARGS+=("detector_model_type:=${SYSNAV_DETECTOR_MODEL_TYPE}")
fi
if [[ -n "${SYSNAV_SAM2_CHECKPOINT:-}" ]]; then
  if [[ ! -f "${SYSNAV_SAM2_CHECKPOINT}" ]]; then
    echo "SYSNAV_SAM2_CHECKPOINT does not exist: ${SYSNAV_SAM2_CHECKPOINT}" >&2
    exit 2
  fi
  MODEL_ARGS+=("sam2_checkpoint:=${SYSNAV_SAM2_CHECKPOINT}")
fi

# 核心：STRIVE 启动 vendored SysNav detector/mapping，输出 /detection_result 和 /object_nodes_list。
exec ros2 launch strive_sysnav_bringup sysnav_detection_mapping.launch.py "${MODEL_ARGS[@]}" "$@"
