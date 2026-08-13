#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
OVERLAY_SETUP="${STRIVE_OVERLAY_SETUP:-${WS_DIR}/install/setup.bash}"
if [[ -d "${OVERLAY_SETUP}" ]]; then
  OVERLAY_SETUP="${OVERLAY_SETUP}/setup.bash"
fi
RUN_ID="${LOWER_BAG_SMOKE_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIRECTORY="${LOWER_BAG_SMOKE_DIRECTORY:-${REPO_ROOT}/logs/real_robot_bag_smoke_${RUN_ID}}"
BAG_PATH="${RUN_DIRECTORY}/synthetic_lower_planner_bag"

if [[ ! -f "${ROS_SETUP}" || ! -f "${OVERLAY_SETUP}" ]]; then
  echo "ROS or STRIVE overlay setup is missing; build the ROS workspace first." >&2
  exit 2
fi

set +u
source "${ROS_SETUP}"
source "${OVERLAY_SETUP}"
set -u

mkdir -p "${RUN_DIRECTORY}"
python3 "${REPO_ROOT}/scripts/generate_synthetic_lower_planner_bag.py" \
  "${BAG_PATH}" \
  --frames "${LOWER_BAG_SMOKE_FRAMES:-80}" \
  --rate-hz "${LOWER_BAG_SMOKE_RATE_HZ:-20}"

export LOWER_BAG_ODOM_TOPIC=/aft_mapped_to_init
export LOWER_BAG_POINTCLOUD_TOPIC=/cloud_registered
export LOWER_BAG_REQUIRED_TOPICS=/aft_mapped_to_init,/cloud_registered
export LOWER_BAG_RUN_DIRECTORY="${RUN_DIRECTORY}/replay"
bash "${REPO_ROOT}/scripts/run_lower_planner_bag_replay.sh" "${BAG_PATH}"

echo "synthetic lower planner bag smoke passed: ${RUN_DIRECTORY}"
