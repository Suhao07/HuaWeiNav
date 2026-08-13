#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
OVERLAY_SETUP="${STRIVE_OVERLAY_SETUP:-${WS_DIR}/install/setup.bash}"

usage() {
  cat <<EOF
Usage: scripts/run_lower_planner_bag_replay.sh BAG_PATH

Replay recorded odometry and registered point-cloud topics through the
migrated SysNav localPlanner. The script starts no pathFollower,
SafetyVelocityMux, chassis bridge, or /cmd_vel publisher.

Environment:
  LOWER_BAG_ODOM_TOPIC=/aft_mapped_to_init
  LOWER_BAG_POINTCLOUD_TOPIC=/cloud_registered
  LOWER_BAG_REQUIRED_TOPICS=/aft_mapped_to_init,/cloud_registered
  LOWER_BAG_GOAL_X=2.0
  LOWER_BAG_GOAL_Y=0.0
  LOWER_BAG_TIMEOUT_S=30
  LOWER_BAG_RUN_DIRECTORY=/tmp/strive_lower_planner_bag_replay
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

BAG_PATH="${1:-}"
if [[ -z "${BAG_PATH}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -e "${BAG_PATH}" ]]; then
  echo "Bag path does not exist: ${BAG_PATH}" >&2
  exit 2
fi
if [[ ! -f "${ROS_SETUP}" || ! -f "${OVERLAY_SETUP}" ]]; then
  echo "ROS or STRIVE overlay setup is missing; build the ROS workspace first." >&2
  exit 2
fi

set +u
source "${ROS_SETUP}"
source "${OVERLAY_SETUP}"
set -u

RUN_DIRECTORY="${LOWER_BAG_RUN_DIRECTORY:-/tmp/strive_lower_planner_bag_replay}"
mkdir -p "${RUN_DIRECTORY}"

ros2 bag info "${BAG_PATH}" | tee "${RUN_DIRECTORY}/bag_info.txt"
required_topics="${LOWER_BAG_REQUIRED_TOPICS:-}"
if [[ -n "${required_topics}" ]]; then
  IFS=',' read -r -a topic_list <<< "${required_topics}"
  for topic in "${topic_list[@]}"; do
    topic="$(echo "${topic}" | xargs)"
    [[ -n "${topic}" ]] || continue
    if ! grep -Fq "${topic}" "${RUN_DIRECTORY}/bag_info.txt"; then
      echo "Required lower-planner bag topic is missing: ${topic}" >&2
      exit 3
    fi
  done
fi

planner_prefix="$(ros2 pkg prefix local_planner)"
ODOM_TOPIC="${LOWER_BAG_ODOM_TOPIC:-/aft_mapped_to_init}"
POINTCLOUD_TOPIC="${LOWER_BAG_POINTCLOUD_TOPIC:-/cloud_registered}"
WAYPOINT_TOPIC="/strive/replay_way_point"
PATH_TOPIC="/strive/replay_path"
STATUS_TOPIC="/strive/replay_planner_status"
ARTIFACT_PATH="${RUN_DIRECTORY}/lower_planner_probe.json"

PLANNER_PID=""
PROBE_PID=""
BAG_PID=""
cleanup() {
  for pid in "${PROBE_PID:-}" "${PLANNER_PID:-}" "${BAG_PID:-}"; do
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

printf '%s\n' \
  "bag_path=${BAG_PATH}" \
  "odom_topic=${ODOM_TOPIC}" \
  "pointcloud_topic=${POINTCLOUD_TOPIC}" \
  "waypoint_topic=${WAYPOINT_TOPIC}" \
  "path_topic=${PATH_TOPIC}" \
  "planner_status_topic=${STATUS_TOPIC}" \
  "cmd_vel_started=false" \
  > "${RUN_DIRECTORY}/replay_config.txt"

echo "== Start migrated SysNav localPlanner only =="
ros2 run local_planner localPlanner \
  --ros-args \
  -p pathFolder:="${planner_prefix}/share/local_planner/paths" \
  -p useTerrainAnalysis:=false \
  -p autonomyMode:=true \
  -p autonomySpeed:=0.5 \
  -p maxSpeed:=0.5 \
  -p status_topic:="${STATUS_TOPIC}" \
  -r /state_estimation:="${ODOM_TOPIC}" \
  -r /registered_scan:="${POINTCLOUD_TOPIC}" \
  -r /way_point:="${WAYPOINT_TOPIC}" \
  -r /path:="${PATH_TOPIC}" \
  > "${RUN_DIRECTORY}/local_planner.log" 2>&1 &
PLANNER_PID=$!
sleep 1
if ! kill -0 "${PLANNER_PID}" >/dev/null 2>&1; then
  sed -n '1,240p' "${RUN_DIRECTORY}/local_planner.log" >&2 || true
  exit 4
fi

echo "== Start lower bag probe =="
ros2 run strive_sysnav_motion lower_bag_probe \
  --ros-args \
  -p waypoint_topic:="${WAYPOINT_TOPIC}" \
  -p path_topic:="${PATH_TOPIC}" \
  -p odom_topic:="${ODOM_TOPIC}" \
  -p pointcloud_topic:="${POINTCLOUD_TOPIC}" \
  -p planner_status_topic:="${STATUS_TOPIC}" \
  -p goal_x:="${LOWER_BAG_GOAL_X:-2.0}" \
  -p goal_y:="${LOWER_BAG_GOAL_Y:-0.0}" \
  -p timeout_s:="${LOWER_BAG_TIMEOUT_S:-30}" \
  -p artifact_path:="${ARTIFACT_PATH}" \
  > "${RUN_DIRECTORY}/lower_bag_probe.log" 2>&1 &
PROBE_PID=$!

# 给 probe 建立订阅后再播放短 bag，避免第一帧在订阅建立前丢失。
sleep 1
echo "== Start rosbag =="
ros2 bag play "${BAG_PATH}" --clock > "${RUN_DIRECTORY}/rosbag.log" 2>&1 &
BAG_PID=$!

set +e
wait "${PROBE_PID}"
probe_status=$?
set -e
if [[ "${probe_status}" -ne 0 ]]; then
  echo "Lower planner bag replay failed; probe log:" >&2
  sed -n '1,240p' "${RUN_DIRECTORY}/lower_bag_probe.log" >&2 || true
  exit "${probe_status}"
fi

echo "Lower planner bag replay passed: ${ARTIFACT_PATH}"
