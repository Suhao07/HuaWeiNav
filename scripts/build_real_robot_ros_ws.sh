#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  echo "Set ROS_DISTRO or install ROS2 before building the real-robot overlay." >&2
  exit 2
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is not available. Install python3-colcon-common-extensions first." >&2
  exit 2
fi

source "${ROS_SETUP}"

cd "${WS_DIR}"

# 核心：只构建检测/语义建图所需 overlay。完整 SysNav local planner 后续单独迁入。
colcon build \
  --symlink-install \
  --packages-up-to tare_planner semantic_mapping strive_sysnav_bringup \
  "$@"
