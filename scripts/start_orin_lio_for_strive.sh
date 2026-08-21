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
# Keep the robot-owned mapping_mid360_orin launch contract explicit when this
# project starts an observation-only Point-LIO process.  These are parameter
# overrides, not edits to the robot's Point-LIO workspace.
LIO_USE_IMU_AS_INPUT="${LIO_USE_IMU_AS_INPUT:-1}"
LIO_PROP_AT_FREQ_OF_IMU="${LIO_PROP_AT_FREQ_OF_IMU:-1}"
LIO_CHECK_SATU="${LIO_CHECK_SATU:-1}"
LIO_INIT_MAP_SIZE="${LIO_INIT_MAP_SIZE:-10}"
LIO_POINT_FILTER_NUM="${LIO_POINT_FILTER_NUM:-6}"
LIO_SPACE_DOWN_SAMPLE="${LIO_SPACE_DOWN_SAMPLE:-1}"
LIO_FILTER_SIZE_SURF="${LIO_FILTER_SIZE_SURF:-0.5}"
LIO_FILTER_SIZE_MAP="${LIO_FILTER_SIZE_MAP:-0.5}"
LIO_CUBE_SIDE_LENGTH="${LIO_CUBE_SIDE_LENGTH:-1000.0}"
LIO_RUNTIME_POS_LOG_ENABLE="${LIO_RUNTIME_POS_LOG_ENABLE:-0}"
LIO_IVOX_NEARBY_TYPE="${LIO_IVOX_NEARBY_TYPE:-6}"
LIO_LOCATION_MODE="${LIO_LOCATION_MODE:-0}"
LIO_CON_FRAME="${LIO_CON_FRAME:-1}"
LIO_CON_FRAME_NUM="${LIO_CON_FRAME_NUM:-10}"

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
  -p use_sim_time:=false
  -p use_imu_as_input:="$(ros_bool "${LIO_USE_IMU_AS_INPUT}")"
  -p prop_at_freq_of_imu:="$(ros_bool "${LIO_PROP_AT_FREQ_OF_IMU}")"
  -p check_satu:="$(ros_bool "${LIO_CHECK_SATU}")"
  -p init_map_size:="${LIO_INIT_MAP_SIZE}"
  -p point_filter_num:="${LIO_POINT_FILTER_NUM}"
  -p space_down_sample:="$(ros_bool "${LIO_SPACE_DOWN_SAMPLE}")"
  -p filter_size_surf:="${LIO_FILTER_SIZE_SURF}"
  -p filter_size_map:="${LIO_FILTER_SIZE_MAP}"
  -p cube_side_length:="${LIO_CUBE_SIDE_LENGTH}"
  -p runtime_pos_log_enable:="$(ros_bool "${LIO_RUNTIME_POS_LOG_ENABLE}")"
  -p ivox_nearby_type:="${LIO_IVOX_NEARBY_TYPE}"
  -p location_mode:="$(ros_bool "${LIO_LOCATION_MODE}")"
  -p common.con_frame:="$(ros_bool "${LIO_CON_FRAME}")"
  -p common.con_frame_num:="${LIO_CON_FRAME_NUM}"
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
LIO_CMD=(ros2 run point_lio pointlio_mapping --ros-args \
  -r __node:=laserMapping \
  --params-file '${POINT_LIO_CONFIG}' \
  ${point_lio_params[*]})
# Negative nice requires elevated scheduling privileges and previously caused
# the ordinary orin26 tmux pane to exit before Point-LIO started.  Bind to the
# first four CPUs when available, but never make deployment depend on it.
if command -v taskset >/dev/null 2>&1 && taskset -c 0-3 true >/dev/null 2>&1; then
  taskset -c 0-3 "\${LIO_CMD[@]}"
else
  echo '[odom] taskset unavailable; starting Point-LIO without CPU affinity'
  "\${LIO_CMD[@]}"
fi
kill \$TF_PID 2>/dev/null || true
exec /bin/zsh
"

tmux list-panes -t "${TMUX_SESSION}" -F 'pane=#{pane_index} cmd=#{pane_current_command} pid=#{pane_pid} active=#{pane_active}'
echo "[start-orin-lio] started ${TMUX_SESSION}; attach with: tmux attach -t ${TMUX_SESSION}"
