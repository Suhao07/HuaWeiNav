#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
OVERLAY_SETUP="${WS_DIR}/install/setup.bash"

usage() {
  cat <<EOF
Usage: scripts/run_real_robot_waypoint_adapter.sh [launch args...]

Starts only the configurable STRIVE waypoint format adapter.  It reads
${STRIVE_WAYPOINT_ADAPTER_INPUT_TOPIC:-/way_point} and publishes nothing by
default. Set output_enabled:=true only after the external controller contract
and adapter parameters have been reviewed.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi
if [[ ! -f "${ROS_SETUP}" || ! -f "${OVERLAY_SETUP}" ]]; then
  echo "ROS setup or real-robot overlay is missing; build the overlay first." >&2
  exit 2
fi

set +u
source "${ROS_SETUP}"
source "${OVERLAY_SETUP}"
set -u
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH="${WAYPOINT_ADAPTER_CONFIG:-}"
ARGS=()
if [[ -n "${CONFIG_PATH}" ]]; then
  ARGS+=("config_path:=${CONFIG_PATH}")
fi
exec ros2 launch strive_sysnav_bringup waypoint_adapter.launch.py "${ARGS[@]}" "$@"
