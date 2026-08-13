#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
OVERLAY_SETUP="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}/install/setup.bash"

PLATFORM="${PLATFORM:-mecanum}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/cloud_registered}"
ODOM_TOPIC="${ODOM_TOPIC:-/aft_mapped_to_init}"
CAMERA_TOPIC="${CAMERA_TOPIC:-/camera/image}"
VIEWPOINT_TOPIC="${VIEWPOINT_TOPIC:-/viewpoint_rep_header}"
START_USB_CAM="${START_USB_CAM:-false}"
USB_VIDEO_DEVICE="${USB_VIDEO_DEVICE:-/dev/video0}"
USB_IMAGE_WIDTH="${USB_IMAGE_WIDTH:-1280}"
USB_IMAGE_HEIGHT="${USB_IMAGE_HEIGHT:-720}"
USB_PIXEL_FORMAT="${USB_PIXEL_FORMAT:-yuyv}"
USB_FRAMERATE="${USB_FRAMERATE:-30.0}"
USB_CAMERA_INFO_URL="${USB_CAMERA_INFO_URL:-}"

BLOCK_LOWER_CONTROLLER="${BLOCK_LOWER_CONTROLLER:-1}"
ENABLE_LOWER_CONTROLLER="${ENABLE_LOWER_CONTROLLER:-0}"
LOWER_CONTROLLER_CMD="${LOWER_CONTROLLER_CMD:-}"
CONTROL_CONTRACT_FILE="${CONTROL_CONTRACT_FILE:-}"
PREFLIGHT_TIMEOUT_S="${PREFLIGHT_TIMEOUT_S:-25}"
WAIT_FOR_LIO="${WAIT_FOR_LIO:-1}"
# Topic discovery alone does not prove that sensor data is flowing.  Fusion
# profiles set this gate so a mapping launch fails closed on stale publishers.
REQUIRE_LIO_SAMPLE="${REQUIRE_LIO_SAMPLE:-0}"
LIO_SAMPLE_TIMEOUT_S="${LIO_SAMPLE_TIMEOUT_S:-5}"
WAIT_FOR_CAMERA="${WAIT_FOR_CAMERA:-0}"
REQUIRE_NO_CMD_VEL_PUBLISHERS="${REQUIRE_NO_CMD_VEL_PUBLISHERS:-1}"
START_SEMANTIC_MAPPING="${START_SEMANTIC_MAPPING:-false}"
MAPPING_CONFIG="${MAPPING_CONFIG:-}"
PROJECTION_CONFIG="${PROJECTION_CONFIG:-}"
DETECTION_TOPIC="${DETECTION_TOPIC:-/detection_result}"
OBJECT_NODES_TOPIC="${OBJECT_NODES_TOPIC:-/object_nodes_list}"

usage() {
  cat <<EOF
Usage: scripts/start_real_robot_framework.sh [ros2 launch args...]

Starts the real-robot STRIVE/SysNav framework inside the container:
  Livox/Point-LIO input topics -> camera -> detection -> semantic mapping.

High-level STRIVE instruction runtime is not started by default. Set
START_STRIVE_RUNTIME=1 to launch it together with detector/mapping.

Safety defaults:
  BLOCK_LOWER_CONTROLLER=1
  ENABLE_LOWER_CONTROLLER=0

This script does not start the chassis bridge, local planner, PD controller, or
any /cmd_vel publisher unless ENABLE_LOWER_CONTROLLER=1 and LOWER_CONTROLLER_CMD
are explicitly provided.
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

section() {
  printf '\n== %s ==\n' "$1"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 2
  fi
}

source_ros() {
  require_file "${ROS_SETUP}"
  require_file "${OVERLAY_SETUP}"
  set +u
  source "${ROS_SETUP}"
  source "${OVERLAY_SETUP}"
  set -u
}

wait_for_topic() {
  local topic="$1"
  local expected_type="${2:-}"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))
  local line

  while ((SECONDS < deadline)); do
    line="$(ros2 topic list -t 2>/dev/null | grep -F "${topic} " || true)"
    if [[ -n "${line}" && ( -z "${expected_type}" || "${line}" == *"[${expected_type}]"* ) ]]; then
      echo "${line}"
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for topic: ${topic}${expected_type:+ [${expected_type}]}" >&2
  return 1
}

wait_for_topic_sample() {
  local topic="$1"
  local timeout_s="$2"
  local output

  if output="$(timeout --kill-after=2s "${timeout_s}" ros2 topic echo --once "${topic}" --field header 2>&1)"; then
    printf '%s\n' "${output}" | sed -n '1,12p'
    return 0
  fi

  printf '%s\n' "${output}" | sed -n '1,12p' >&2
  echo "Timed out waiting for an actual message on ${topic}; a graph publisher is not sufficient." >&2
  return 1
}

cmd_vel_publishers() {
  ros2 topic info /cmd_vel 2>/dev/null | awk '/Publisher count:/ {print $3}'
}

preflight() {
  section "Real-Robot Framework Preflight"
  echo "platform=${PLATFORM}"
  echo "cloud_topic=${CLOUD_TOPIC}"
  echo "odom_topic=${ODOM_TOPIC}"
  echo "camera_topic=${CAMERA_TOPIC}"
  echo "start_usb_cam=${START_USB_CAM}"
  echo "start_semantic_mapping=${START_SEMANTIC_MAPPING}"
  echo "block_lower_controller=${BLOCK_LOWER_CONTROLLER}"
  echo "start_strive_runtime=${START_STRIVE_RUNTIME:-0}"

  if is_true "${WAIT_FOR_LIO}"; then
    wait_for_topic "${CLOUD_TOPIC}" "sensor_msgs/msg/PointCloud2" "${PREFLIGHT_TIMEOUT_S}"
    wait_for_topic "${ODOM_TOPIC}" "nav_msgs/msg/Odometry" "${PREFLIGHT_TIMEOUT_S}"
  fi

  if is_true "${REQUIRE_LIO_SAMPLE}"; then
    section "LIO Data Sample Gate"
    wait_for_topic_sample "${CLOUD_TOPIC}" "${LIO_SAMPLE_TIMEOUT_S}"
    wait_for_topic_sample "${ODOM_TOPIC}" "${LIO_SAMPLE_TIMEOUT_S}"
  fi

  if ! is_true "${START_USB_CAM}" && is_true "${WAIT_FOR_CAMERA}"; then
    wait_for_topic "${CAMERA_TOPIC}" "sensor_msgs/msg/Image" "${PREFLIGHT_TIMEOUT_S}"
  fi

  if is_true "${BLOCK_LOWER_CONTROLLER}" && is_true "${REQUIRE_NO_CMD_VEL_PUBLISHERS}"; then
    local publishers
    publishers="$(cmd_vel_publishers || true)"
    if [[ -n "${publishers}" && "${publishers}" != "0" ]]; then
      echo "Refusing to start in blocked-control mode: /cmd_vel already has ${publishers} publisher(s)." >&2
      echo "Set REQUIRE_NO_CMD_VEL_PUBLISHERS=0 only if this is intentional." >&2
      exit 3
    fi
  fi
}

maybe_start_lower_controller() {
  if is_true "${BLOCK_LOWER_CONTROLLER}"; then
    echo "[control] blocked: lower controller / chassis bridge startup is disabled."
    return 0
  fi

  if ! is_true "${ENABLE_LOWER_CONTROLLER}"; then
    echo "[control] disabled: ENABLE_LOWER_CONTROLLER=0."
    return 0
  fi

  require_motion_contract

  if [[ -z "${LOWER_CONTROLLER_CMD}" ]]; then
    echo "ENABLE_LOWER_CONTROLLER=1 requires LOWER_CONTROLLER_CMD." >&2
    exit 4
  fi

  echo "[control] starting lower controller command: ${LOWER_CONTROLLER_CMD}"
  bash -lc "${LOWER_CONTROLLER_CMD}" &
}

require_motion_contract() {
  if [[ -z "${CONTROL_CONTRACT_FILE}" ]]; then
    echo "Live lower-controller startup requires CONTROL_CONTRACT_FILE." >&2
    exit 4
  fi
  if [[ ! -f "${CONTROL_CONTRACT_FILE}" ]]; then
    echo "Control contract does not exist: ${CONTROL_CONTRACT_FILE}" >&2
    exit 4
  fi
  local required_gate
  for required_gate in \
    '^[[:space:]]+approval_status:[[:space:]]+approved[[:space:]]*$' \
    '^[[:space:]]+allow_strive_waypoint_handoff:[[:space:]]+true[[:space:]]*$' \
    '^[[:space:]]+cmd_vel_direct_publish:[[:space:]]+false[[:space:]]*$' \
    '^[[:space:]]+emergency_stop_verified:[[:space:]]+true[[:space:]]*$'; do
    if ! grep -Eq "${required_gate}" "${CONTROL_CONTRACT_FILE}"; then
      echo "Control contract is missing an approved safety gate: ${required_gate}" >&2
      exit 4
    fi
  done
}

append_launch_arg() {
  local -n args_ref=$1
  local name="$2"
  local value="$3"
  [[ -n "${value}" ]] || return 0
  args_ref+=("${name}:=${value}")
}

main() {
  case "${1:-}" in
    -h|--help|help)
      usage
      return 0
      ;;
  esac

  source_ros
  preflight
  maybe_start_lower_controller

  section "Start STRIVE Real-Robot Framework"
  echo "Launching detection + semantic mapping. High-level runtime starts only when START_STRIVE_RUNTIME=1."
  echo "Control output remains blocked unless explicitly enabled."
  local launch_args=(
    "platform:=${PLATFORM}"
    "cloud_topic:=${CLOUD_TOPIC}"
    "odom_topic:=${ODOM_TOPIC}"
    "camera_topic:=${CAMERA_TOPIC}"
    "start_semantic_mapping:=${START_SEMANTIC_MAPPING}"
    "detection_topic:=${DETECTION_TOPIC}"
    "object_nodes_topic:=${OBJECT_NODES_TOPIC}"
    "viewpoint_topic:=${VIEWPOINT_TOPIC}"
    "start_usb_cam:=${START_USB_CAM}"
    "usb_video_device:=${USB_VIDEO_DEVICE}"
    "usb_image_width:=${USB_IMAGE_WIDTH}"
    "usb_image_height:=${USB_IMAGE_HEIGHT}"
    "usb_pixel_format:=${USB_PIXEL_FORMAT}"
    "usb_framerate:=${USB_FRAMERATE}"
    "usb_camera_info_url:=${USB_CAMERA_INFO_URL}"
  )
  append_launch_arg launch_args "mapping_config" "${MAPPING_CONFIG}"
  append_launch_arg launch_args "projection_config" "${PROJECTION_CONFIG}"
  exec "${REPO_ROOT}/scripts/run_sysnav_detection_mapping.sh" "${launch_args[@]}" "$@"
}

main "$@"
