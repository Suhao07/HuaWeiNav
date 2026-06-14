#!/usr/bin/env bash
set -euo pipefail

TMUX_SESSION="${TMUX_SESSION:-livox_odom}"
ROS_SETUP_ZSH="${ROS_SETUP_ZSH:-/opt/ros/humble/setup.zsh}"
LIVOX_SETUP_ZSH="${LIVOX_SETUP_ZSH:-/home/orin26/code/ws_livox/install/setup.zsh}"
POINT_LIO_SETUP_ZSH="${POINT_LIO_SETUP_ZSH:-/home/orin26/code/point_lio_ws/install/setup.zsh}"
POINT_LIO_CONFIG="${POINT_LIO_CONFIG:-/home/orin26/code/point_lio_ws/install/point_lio/share/point_lio/config/mid360_orin.yaml}"
ENABLE_CLOUD_PUBLISH="${ENABLE_CLOUD_PUBLISH:-1}"
ENABLE_BODY_CLOUD_PUBLISH="${ENABLE_BODY_CLOUD_PUBLISH:-0}"
RESTART_EXISTING="${RESTART_EXISTING:-1}"

ros_bool() {
  case "${1,,}" in
    1|true|yes|on) echo true ;;
    *) echo false ;;
  esac
}

if [[ "${RESTART_EXISTING}" == "1" ]]; then
  tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "[start-orin-lio] tmux session already exists: ${TMUX_SESSION}" >&2
  exit 2
fi

if [[ ! -f "${ROS_SETUP_ZSH}" || ! -f "${LIVOX_SETUP_ZSH}" || ! -f "${POINT_LIO_SETUP_ZSH}" ]]; then
  echo "[start-orin-lio] missing ROS/Livox/Point-LIO setup file" >&2
  exit 3
fi
if [[ ! -f "${POINT_LIO_CONFIG}" ]]; then
  echo "[start-orin-lio] missing Point-LIO config: ${POINT_LIO_CONFIG}" >&2
  exit 3
fi

point_lio_params=(
  -p publish.scan_publish_en:="$(ros_bool "${ENABLE_CLOUD_PUBLISH}")"
  -p publish.scan_bodyframe_pub_en:="$(ros_bool "${ENABLE_BODY_CLOUD_PUBLISH}")"
)

tmux new-session -d -s "${TMUX_SESSION}" -n livox /bin/zsh -lc "
set -e
source '${ROS_SETUP_ZSH}'
source '${LIVOX_SETUP_ZSH}'
echo '[livox] ros2 launch livox_ros_driver2 msg_MID360_launch.py'
ros2 launch livox_ros_driver2 msg_MID360_launch.py
exec /bin/zsh
"

tmux split-window -t "${TMUX_SESSION}:0" -h /bin/zsh -lc "
set -e
source '${ROS_SETUP_ZSH}'
source '${LIVOX_SETUP_ZSH}'
source '${POINT_LIO_SETUP_ZSH}'
sleep 2
echo '[odom] ros2 run point_lio pointlio_mapping with STRIVE cloud publish settings'
ros2 run tf2_ros static_transform_publisher \
  --x -0.2 --y 0.0 --z 0.0 \
  --yaw -1.5708 --pitch 0.0 --roll 0.0 \
  --frame-id aft_mapped --child-frame-id base \
  --ros-args -r __node:=tf_aft_mapped_to_base &
TF_PID=\$!
taskset -c 0-3 nice -n -10 ros2 run point_lio pointlio_mapping --ros-args \
  -r __node:=laserMapping \
  --params-file '${POINT_LIO_CONFIG}' \
  ${point_lio_params[*]}
kill \$TF_PID 2>/dev/null || true
exec /bin/zsh
"

tmux list-panes -t "${TMUX_SESSION}" -F 'pane=#{pane_index} cmd=#{pane_current_command} pid=#{pane_pid} active=#{pane_active}'
echo "[start-orin-lio] started ${TMUX_SESSION}; attach with: tmux attach -t ${TMUX_SESSION}"
