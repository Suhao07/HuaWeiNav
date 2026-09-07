#!/usr/bin/env bash
set -euo pipefail

# Project-owned sensor entrypoint. It starts only the externally-owned Livox /
# Point-LIO and D435i processes; it never starts mapping, detection, planning,
# waypoint, cmd_vel, or a lower controller.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${1:-orin26_livox_mid360_d435i}"
ACTION="${2:-start}"
PROFILE_FILE="${ROOT_DIR}/real_robot/profiles/${PROFILE%.env}.env"

usage() {
  echo "Usage: $0 [profile] {start|status|check|stop}"
  echo "Example: $0 orin26_livox_mid360_d435i start"
}

[[ -f "${PROFILE_FILE}" ]] || { echo "missing profile: ${PROFILE_FILE}" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "${PROFILE_FILE}"
set +a

ROS_SETUP="${ROS_SETUP_ZSH:-/opt/ros/humble/setup.zsh}"
LIO_HELPER="${LIO_START_SCRIPT:-${ROOT_DIR}/scripts/start_orin_lio_for_strive.sh}"
LIO_SESSION="${LIO_TMUX_SESSION:-livox_odom}"
CAMERA_SESSION="${D435I_TMUX_SESSION:-d435i_camera}"
LIVOX_SETUP="${LIVOX_SETUP_ZSH:-/home/orin26/code/ws_livox/install/setup.zsh}"
POINT_LIO_SETUP="${POINT_LIO_SETUP_ZSH:-/home/orin26/code/point_lio_ws/install/setup.zsh}"
CAMERA_NAMESPACE="${D435I_CAMERA_NAMESPACE:-camera/d435i}"
CAMERA_NAME="${D435I_CAMERA_NAME:-d435i_camera}"
CAMERA_SERIAL="${D435I_SERIAL_NO:-_233522079589}"
CAMERA_FPS="${D435I_FPS:-15}"
RGB_TOPIC="${RGB_TOPIC:-/camera/d435i/d435i_camera/color/image_raw}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/d435i/d435i_camera/aligned_depth_to_color/image_raw}"
INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/d435i/d435i_camera/color/camera_info}"
LIVOX_PUBLISH_FREQ="${LIVOX_PUBLISH_FREQ:-10.0}"

ros_cmd() { zsh -lc "source '${ROS_SETUP}'; source '${LIVOX_SETUP}'; source '${POINT_LIO_SETUP}'; $*"; }
session_exists() { tmux has-session -t "$1" 2>/dev/null; }

start_camera() {
  if session_exists "${CAMERA_SESSION}"; then
    echo "[sensors] reusing D435i tmux session: ${CAMERA_SESSION}"
    return
  fi
  echo "[sensors] starting D435i ${CAMERA_SERIAL} at 1280x720@${CAMERA_FPS} with aligned depth"
  tmux new-session -d -s "${CAMERA_SESSION}" -n camera /bin/zsh -lc \
    "set -e; source '${ROS_SETUP}'; exec ros2 launch realsense2_camera rs_launch.py \
      camera_namespace:='${CAMERA_NAMESPACE}' camera_name:='${CAMERA_NAME}' \
      serial_no:='${CAMERA_SERIAL}' enable_color:=true enable_depth:=true \
      rgb_camera.color_profile:=1280,720,${CAMERA_FPS} depth_module.depth_profile:=1280,720,${CAMERA_FPS} \
      align_depth.enable:=true enable_sync:=true publish_tf:=true"
}

start_lio() {
  if session_exists "${LIO_SESSION}"; then
    echo "[sensors] reusing Point-LIO tmux session: ${LIO_SESSION}"
    return
  fi
  echo "[sensors] starting robot-owned Livox + Point-LIO; runtime cloud/body-cloud overrides enabled"
  ENABLE_CLOUD_PUBLISH=1 ENABLE_BODY_CLOUD_PUBLISH=1 RESTART_EXISTING=0 \
    bash "${LIO_HELPER}" start
  ros_cmd "ros2 param set /livox_lidar_publisher publish_freq ${LIVOX_PUBLISH_FREQ}"
}

check_topics() {
  local t
  for t in "${RGB_TOPIC}" "${DEPTH_TOPIC}" "${INFO_TOPIC}" /livox/lidar /livox/imu /cloud_registered_body /aft_mapped_to_init; do
    echo "[sensors] sample ${t}"
    if ! ros_cmd "timeout 6 ros2 topic echo --once '${t}' >/dev/null"; then
      echo "[sensors] FAIL: no actual message on ${t}" >&2
      return 1
    fi
  done
  echo "[sensors] PASS: all configured sensor/LIO topics have actual messages"
}

case "${ACTION}" in
  start)
    start_lio
    start_camera
    sleep "${SENSOR_START_SETTLE_S:-5}"
    check_topics
    ;;
  check)
    check_topics
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep -E "(${LIO_SESSION}|${CAMERA_SESSION})" || true
    ros_cmd "ros2 node list | grep -E 'laserMapping|livox|d435i' || true"
    ;;
  stop)
    echo "[sensors] stopping only project-started sensor sessions"
    tmux kill-session -t "${CAMERA_SESSION}" 2>/dev/null || true
    tmux kill-session -t "${LIO_SESSION}" 2>/dev/null || true
    ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
