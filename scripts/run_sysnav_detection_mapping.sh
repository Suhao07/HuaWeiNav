#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
OVERLAY_SETUP="${WS_DIR}/install/setup.bash"
START_STRIVE_RUNTIME="${START_STRIVE_RUNTIME:-0}"

usage() {
  cat <<EOF
Usage: scripts/run_sysnav_detection_mapping.sh [sysnav launch args...]

Starts SysNav detector + semantic mapping only by default.

Optional high-level runtime:
  START_STRIVE_RUNTIME=1 scripts/run_sysnav_detection_mapping.sh [sysnav launch args...]

When START_STRIVE_RUNTIME=1, configure the runtime through environment
variables such as:
  STRIVE_INSTRUCTION
  STRIVE_DATASET_TARGET
  STRIVE_POLICY_MODE
  STRIVE_INSTRUCTION_PLAN_BACKEND
  STRIVE_DRY_RUN
  STRIVE_RUN_DIRECTORY
  STRIVE_PRIOR_MAP_PATH
  STRIVE_OBJECT_TOPIC / STRIVE_ROOM_TOPIC / STRIVE_ODOM_TOPIC / STRIVE_IMAGE_TOPIC

The runtime remains dry-run by default and does not publish /way_point unless
STRIVE_DRY_RUN=false and the safety parameters allow it.
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

append_runtime_arg() {
  local -n args_ref=$1
  local launch_name="$2"
  local value="$3"
  [[ -n "${value}" ]] || return 0
  args_ref+=("${launch_name}:=${value}")
}

runtime_args() {
  RUNTIME_ARGS=()
  append_runtime_arg RUNTIME_ARGS "instruction" "${STRIVE_INSTRUCTION:-}"
  append_runtime_arg RUNTIME_ARGS "dataset_target" "${STRIVE_DATASET_TARGET:-}"
  append_runtime_arg RUNTIME_ARGS "policy_mode" "${STRIVE_POLICY_MODE:-wait}"
  append_runtime_arg RUNTIME_ARGS "instruction_plan_backend" "${STRIVE_INSTRUCTION_PLAN_BACKEND:-rules}"
  append_runtime_arg RUNTIME_ARGS "vlm" "${STRIVE_VLM:-cognav}"
  append_runtime_arg RUNTIME_ARGS "enable_final_verifier" "${STRIVE_ENABLE_FINAL_VERIFIER:-false}"
  append_runtime_arg RUNTIME_ARGS "evidence_mode" "${STRIVE_EVIDENCE_MODE:-auto}"
  append_runtime_arg RUNTIME_ARGS "prior_map_path" "${STRIVE_PRIOR_MAP_PATH:-}"
  append_runtime_arg RUNTIME_ARGS "prior_map_source" "${STRIVE_PRIOR_MAP_SOURCE:-auto}"
  append_runtime_arg RUNTIME_ARGS "prior_map_alignment" "${STRIVE_PRIOR_MAP_ALIGNMENT:-identity}"
  append_runtime_arg RUNTIME_ARGS "run_directory" "${STRIVE_RUN_DIRECTORY:-/tmp/strive_real_robot_runtime}"
  append_runtime_arg RUNTIME_ARGS "dry_run" "${STRIVE_DRY_RUN:-true}"
  append_runtime_arg RUNTIME_ARGS "dry_run_status" "${STRIVE_DRY_RUN_STATUS:-idle}"
  append_runtime_arg RUNTIME_ARGS "lower_controller_enabled" "${STRIVE_LOWER_CONTROLLER_ENABLED:-false}"
  append_runtime_arg RUNTIME_ARGS "waypoint_topic" "${STRIVE_WAYPOINT_TOPIC:-/way_point}"
  append_runtime_arg RUNTIME_ARGS "test_waypoint_topic" "${STRIVE_TEST_WAYPOINT_TOPIC:-/strive/test_way_point}"
  append_runtime_arg RUNTIME_ARGS "hold_topic" "${STRIVE_HOLD_TOPIC:-}"
  append_runtime_arg RUNTIME_ARGS "cancel_topic" "${STRIVE_CANCEL_TOPIC:-}"
  append_runtime_arg RUNTIME_ARGS "emergency_stop_topic" "${STRIVE_EMERGENCY_STOP_TOPIC:-}"
  append_runtime_arg RUNTIME_ARGS "allow_emergency_stop_publish" "${STRIVE_ALLOW_EMERGENCY_STOP_PUBLISH:-false}"
  append_runtime_arg RUNTIME_ARGS "object_topic" "${STRIVE_OBJECT_TOPIC:-/object_nodes_list}"
  append_runtime_arg RUNTIME_ARGS "room_topic" "${STRIVE_ROOM_TOPIC:-/room_nodes_list}"
  append_runtime_arg RUNTIME_ARGS "odom_topic" "${STRIVE_ODOM_TOPIC:-${ODOM_TOPIC:-/aft_mapped_to_init}}"
  append_runtime_arg RUNTIME_ARGS "path_topic" "${STRIVE_PATH_TOPIC:-/path}"
  append_runtime_arg RUNTIME_ARGS "planner_status_topic" "${STRIVE_PLANNER_STATUS_TOPIC:-}"
  append_runtime_arg RUNTIME_ARGS "image_topic" "${STRIVE_IMAGE_TOPIC:-${CAMERA_TOPIC:-/camera/image}}"
  append_runtime_arg RUNTIME_ARGS "detection_topic" "${STRIVE_DETECTION_TOPIC:-/detection_result}"
  append_runtime_arg RUNTIME_ARGS "depth_topic" "${STRIVE_DEPTH_TOPIC:-}"
  append_runtime_arg RUNTIME_ARGS "pointcloud_topic" "${STRIVE_POINTCLOUD_TOPIC:-}"
  append_runtime_arg RUNTIME_ARGS "persist_observation_images" "${STRIVE_PERSIST_OBSERVATION_IMAGES:-false}"
  append_runtime_arg RUNTIME_ARGS "observation_image_directory" "${STRIVE_OBSERVATION_IMAGE_DIRECTORY:-}"
  append_runtime_arg RUNTIME_ARGS "decision_period_s" "${STRIVE_DECISION_PERIOD_S:-1.0}"
  append_runtime_arg RUNTIME_ARGS "use_sim_time" "${STRIVE_USE_SIM_TIME:-false}"
}

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

# 核心：默认只启动 vendored SysNav detector/mapping，输出 /detection_result 和 /object_nodes_list。
if ! is_true "${START_STRIVE_RUNTIME}"; then
  exec ros2 launch strive_sysnav_bringup sysnav_detection_mapping.launch.py "${MODEL_ARGS[@]}" "$@"
fi

runtime_args
PIDS=()
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

ros2 launch strive_sysnav_bringup sysnav_detection_mapping.launch.py "${MODEL_ARGS[@]}" "$@" &
PIDS+=("$!")
ros2 launch strive_sysnav_bringup strive_instruction_runtime.launch.py "${RUNTIME_ARGS[@]}" &
PIDS+=("$!")

set +e
wait -n "${PIDS[@]}"
status=$?
set -e
cleanup
exit "${status}"
