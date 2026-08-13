#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
COLCON_GLOBAL_ARGS=()
COLCON_BUILD_ARGS=()

# Keep build/install/log locations explicit. colcon treats these as global
# arguments; parsing them before the package selection avoids silently
# reusing a CMake cache from another workspace mount.
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-base|--install-base|--log-base)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a path" >&2
        exit 2
      fi
      case "$1" in
        --build-base) COLCON_BUILD_ARGS+=(--build-base "$2") ;;
        --install-base) COLCON_BUILD_ARGS+=(--install-base "$2") ;;
        --log-base) COLCON_GLOBAL_ARGS+=(--log-base "$2") ;;
      esac
      shift 2
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${STRIVE_COLCON_LOG_BASE:-}" && ! " ${COLCON_GLOBAL_ARGS[*]} " =~ " --log-base " ]]; then
  COLCON_GLOBAL_ARGS+=(--log-base "${STRIVE_COLCON_LOG_BASE}")
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  echo "Set ROS_DISTRO or install ROS2 before building the real-robot overlay." >&2
  exit 2
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is not available. Install python3-colcon-common-extensions first." >&2
  exit 2
fi

# ROS setup files may read unset environment variables. Keep nounset disabled
# only while sourcing the external ROS environment.
set +u
source "${ROS_SETUP}"
set -u

cd "${WS_DIR}"

# 核心：消息、SysNav 局部规划、检测建图与任务级 motion overlay 分层构建。
# local_planner/pathFollower 的最终输出是 /cmd_vel/autonomy，真实底盘仍由
# SafetyVelocityMux 和外部安全契约控制。
colcon "${COLCON_GLOBAL_ARGS[@]}" build \
  --symlink-install \
  "${COLCON_BUILD_ARGS[@]}" \
  --packages-up-to tare_planner terrain_analysis local_planner semantic_mapping strive_motion_msgs strive_sysnav_motion strive_sysnav_bringup \
  "${REMAINING_ARGS[@]}"
