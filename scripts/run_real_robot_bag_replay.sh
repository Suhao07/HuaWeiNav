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
Usage: scripts/run_real_robot_bag_replay.sh BAG_PATH [runtime launch args...]

Replays a rosbag and starts only the STRIVE high-level instruction runtime.
It does not start SysNav detector/mapping, lower controllers, /way_point, or
/cmd_vel publishers by default.

Default runtime topic mapping:
  object_topic=${BAG_OBJECT_TOPIC:-/object_nodes_list}
  room_topic=${BAG_ROOM_TOPIC:-/room_nodes_list}
  odom_topic=${BAG_ODOM_TOPIC:-/aft_mapped_to_init}
  image_topic=${BAG_IMAGE_TOPIC:-/camera/image}
  detection_topic=${BAG_DETECTION_TOPIC:-/detection_result}
  path_topic=${BAG_PATH_TOPIC:-/path}

Useful environment variables:
  BAG_LOOP=1
  BAG_RATE=0.5
  STRIVE_INSTRUCTION="find a book"
  STRIVE_DATASET_TARGET=book
  STRIVE_POLICY_MODE=semantic_snapshot
  STRIVE_INSTRUCTION_PLAN_BACKEND=rules
  STRIVE_RUN_DIRECTORY=/tmp/strive_real_robot_bag_replay

Any extra arguments are passed to strive_instruction_runtime.launch.py and can
override the defaults above.
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
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
shift

if [[ ! -e "${BAG_PATH}" ]]; then
  echo "Bag path does not exist: ${BAG_PATH}" >&2
  exit 2
fi
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

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

BAG_PLAY_ARGS=("--clock")
if is_true "${BAG_LOOP:-0}"; then
  BAG_PLAY_ARGS+=("--loop")
fi
if [[ -n "${BAG_RATE:-}" ]]; then
  BAG_PLAY_ARGS+=("--rate" "${BAG_RATE}")
fi

RUNTIME_ARGS=(
  "use_sim_time:=true"
  "dry_run:=${STRIVE_DRY_RUN:-true}"
  "dry_run_status:=${STRIVE_DRY_RUN_STATUS:-idle}"
  "policy_mode:=${STRIVE_POLICY_MODE:-wait}"
  "instruction:=${STRIVE_INSTRUCTION:-}"
  "dataset_target:=${STRIVE_DATASET_TARGET:-}"
  "instruction_plan_backend:=${STRIVE_INSTRUCTION_PLAN_BACKEND:-rules}"
  "vlm:=${STRIVE_VLM:-cognav}"
  "enable_final_verifier:=${STRIVE_ENABLE_FINAL_VERIFIER:-false}"
  "evidence_mode:=${STRIVE_EVIDENCE_MODE:-auto}"
  "prior_map_path:=${STRIVE_PRIOR_MAP_PATH:-}"
  "run_directory:=${STRIVE_RUN_DIRECTORY:-/tmp/strive_real_robot_bag_replay}"
  "object_topic:=${BAG_OBJECT_TOPIC:-/object_nodes_list}"
  "room_topic:=${BAG_ROOM_TOPIC:-/room_nodes_list}"
  "odom_topic:=${BAG_ODOM_TOPIC:-/aft_mapped_to_init}"
  "image_topic:=${BAG_IMAGE_TOPIC:-/camera/image}"
  "detection_topic:=${BAG_DETECTION_TOPIC:-/detection_result}"
  "path_topic:=${BAG_PATH_TOPIC:-/path}"
  "planner_status_topic:=${BAG_PLANNER_STATUS_TOPIC:-}"
  "depth_topic:=${BAG_DEPTH_TOPIC:-}"
  "pointcloud_topic:=${BAG_POINTCLOUD_TOPIC:-}"
  "lower_controller_enabled:=false"
)

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

echo "== Bag info =="
ros2 bag info "${BAG_PATH}" || true

PIDS=()
echo "== Start bag replay =="
ros2 bag play "${BAG_PATH}" "${BAG_PLAY_ARGS[@]}" &
PIDS+=("$!")

echo "== Start STRIVE instruction runtime =="
ros2 launch strive_sysnav_bringup strive_instruction_runtime.launch.py "${RUNTIME_ARGS[@]}" "$@" &
PIDS+=("$!")

set +e
wait -n "${PIDS[@]}"
status=$?
set -e
cleanup
exit "${status}"
