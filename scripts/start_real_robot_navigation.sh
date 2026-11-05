#!/usr/bin/env bash
# Container entry: production runtime -> Action -> native planner/follower -> safety mux.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${STRIVE_EXECUTION_MODE:-preview}"
case "$MODE" in
 preview) PREFIX=/strive/preview; CONTRACT_REQUIRED=false; LIVE=false; AUTO=true ;;
 live)
  [[ "${ALLOW_REAL_MOTION:-false}" == true ]] || { echo 'live requires ALLOW_REAL_MOTION=true' >&2; exit 2; }
  [[ -f "${CONTROL_CONTRACT_FILE:-}" ]] || { echo 'live requires CONTROL_CONTRACT_FILE' >&2; exit 2; }
  PREFIX=/strive/navigation; CONTRACT_REQUIRED=true; LIVE=true; AUTO=false ;;
 *) echo 'STRIVE_EXECUTION_MODE must be preview or live' >&2; exit 2 ;;
esac
python3 "$ROOT/real_robot/navigation_parameters.py"
if [[ "${1:-}" == --check ]]; then
 printf 'mode=%s\nbridge_started=%s\noutput_topic=%s/cmd_vel\ncontract_required=%s\n' "$MODE" "$LIVE" "$PREFIX" "$CONTRACT_REQUIRED"
 exit 0
fi
set +u
source /opt/ros/humble/setup.bash
source "${STRIVE_OVERLAY_SETUP:-${ROOT}/real_robot/ros2_ws/install/setup.bash}"
set -u
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
RUN="${STRIVE_RUN_DIRECTORY:-/tmp/strive-navigation}"
mkdir -p "$RUN"
exec 9>"$RUN/navigation.lock"
flock -n 9 || { echo 'navigation already owns this run directory' >&2; exit 3; }
WORLD="${SYSNAV_WORLD_FRAME:-camera_init}"
BASE_ODOM="$PREFIX/base_odom"
CONTRACT="${CONTROL_CONTRACT_FILE:-}"
CONTRACT_ARGS=()
[[ -z "$CONTRACT" ]] || CONTRACT_ARGS+=("controller_contract_file:=$CONTRACT")
PRIOR_ARGS=()
if [[ -n "${STRIVE_PRIOR_MAP_PATH:-}" ]]; then
 PRIOR_ARGS+=("prior_map_path:=$STRIVE_PRIOR_MAP_PATH" "prior_map_source:=${STRIVE_PRIOR_MAP_SOURCE:-auto}" "prior_map_alignment:=${STRIVE_PRIOR_MAP_ALIGNMENT:-identity}")
fi
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT TERM
start() { local name=$1; shift; setsid "$@" >"$RUN/$name.log" 2>&1 & PIDS+=($!); }
# Perception continues in the sensor/lidar frame; runtime and lower stack consume base pose.
start perception env START_FORMAL_NAVIGATION=0 START_STRIVE_RUNTIME=0 START_LOWER_STACK=0 bash "$ROOT/scripts/run_sysnav_detection_mapping.sh" "$@"
start base_odometry python3 -m strive_sysnav_motion.base_odometry --ros-args \
 -p input_topic:="${ODOM_TOPIC:-/aft_mapped_to_init}" -p output_topic:="$BASE_ODOM" \
 -p world_frame:="$WORLD" -p base_x:="${NAV_BASE_X_M:--0.2}" -p base_y:="${NAV_BASE_Y_M:-0.0}" -p base_yaw:="${NAV_BASE_YAW_RAD:--1.5708}"
start lower ros2 launch strive_sysnav_motion sysnav_lower_stack.launch.py \
 cloud_topic:="${NAV_CLOUD_TOPIC:-/cloud_registered}" odom_topic:="$BASE_ODOM" world_frame:="$WORLD" \
 use_terrain_analysis:=false vehicle_length:="${NAV_VEHICLE_LENGTH_M:-0.6}" vehicle_width:="${NAV_VEHICLE_WIDTH_M:-0.6}" \
 waypoint_topic:="$PREFIX/way_point" path_topic:="$PREFIX/path" planner_status_topic:="$PREFIX/status" \
 action_name:="$PREFIX/execute_waypoint" hold_topic:="$PREFIX/hold" cancel_topic:="$PREFIX/cancel" \
 autonomy_cmd_topic:="$PREFIX/raw_cmd_vel" manual_cmd_topic:="$PREFIX/manual_cmd_vel" output_cmd_topic:="$PREFIX/cmd_vel" \
 autonomy_enable_topic:="$PREFIX/autonomy_enable" manual_takeover_topic:="$PREFIX/manual_takeover" \
 estop_topic:="$PREFIX/estop" estop_reset_topic:="$PREFIX/estop_reset" safety_state_topic:="$PREFIX/safety" \
 chassis_ready_topic:="$PREFIX/chassis_ready" require_chassis_ready:="$LIVE" \
 "${CONTRACT_ARGS[@]}" require_controller_contract:="$CONTRACT_REQUIRED" start_autonomy_enabled:="$AUTO" \
 max_speed:="${NAV_MAX_SPEED_MPS:-0.15}" autonomy_speed:="${NAV_MAX_SPEED_MPS:-0.15}" \
 max_angular_speed:="${NAV_MAX_ANGULAR_RPS:-0.35}" follower_max_yaw_deg_s:="${NAV_FOLLOWER_YAW_DEG_S:-20.0}" \
 follower_stop_distance_m:="${NAV_FOLLOWER_STOP_DISTANCE_M:-0.5}" \
 follower_lookahead_m:="${NAV_FOLLOWER_LOOKAHEAD_M:-1.0}" follower_slowdown_distance_m:="${NAV_FOLLOWER_SLOWDOWN_DISTANCE_M:-1.0}" \
 max_linear_accel:="${NAV_MAX_ACCEL_MPS2:-0.15}" max_angular_accel:="${NAV_MAX_ANGULAR_ACCEL_RPS2:-0.3}" command_watchdog_timeout:="${NAV_COMMAND_TIMEOUT_S:-0.25}" \
 xy_goal_tolerance_m:="${NAV_GOAL_TOLERANCE_M:-1.0}" no_progress_timeout_s:="${NAV_NO_PROGRESS_TIMEOUT_S:-12.0}"
start runtime ros2 launch strive_sysnav_bringup strive_instruction_runtime.launch.py \
 instruction:="${STRIVE_INSTRUCTION:-Find the chair}" dataset_target:="${STRIVE_DATASET_TARGET:-chair}" \
 policy_mode:=semantic_snapshot instruction_plan_backend:=rules motion_backend:=action \
 require_motion_ready:="$LIVE" motion_ready_topic:="$PREFIX/autonomy_enable" \
 goal_resolution:=standoff approach_standoff_m:="${NAV_STANDOFF_M:-0.8}" \
 dry_run:=false lower_controller_enabled:="$LIVE" "${CONTRACT_ARGS[@]}" \
 world_frame:="$WORLD" odom_topic:="$BASE_ODOM" \
 object_topic:="${OBJECT_NODES_TOPIC:-/huawei_vln/d435i_object_nodes_list}" room_topic:="${STRIVE_ROOM_TOPIC:-/huawei_vln/d435i_room_nodes_list}" \
 image_topic:="${CAMERA_TOPIC:-/camera/d435i/d435i_camera/color/image_raw}" detection_topic:="${DETECTION_TOPIC:-/huawei_vln/d435i_detection_result}" \
 pointcloud_topic:="${NAV_CLOUD_TOPIC:-/cloud_registered}" path_topic:="$PREFIX/path" planner_status_topic:="$PREFIX/status" \
 waypoint_topic:="$PREFIX/way_point" test_waypoint_topic:="$PREFIX/way_point" motion_action_name:="$PREFIX/execute_waypoint" \
 hold_topic:="$PREFIX/hold" cancel_topic:="$PREFIX/cancel" enable_final_verifier:=false \
 "${PRIOR_ARGS[@]}" \
 run_directory:="$RUN/runtime" xy_goal_tolerance_m:="${NAV_GOAL_TOLERANCE_M:-1.0}"
if [[ "$LIVE" == true ]]; then
 start bridge ros2 launch strive_sysnav_motion agilex_platform.launch.py \
 input_cmd_topic:="$PREFIX/cmd_vel" uri:="${AGILEX_URI:-ws://192.168.1.102:9090}" remote_control_enabled:=true \
 "${CONTRACT_ARGS[@]}" waypoint_topic:="$PREFIX/way_point" world_frame:="$WORLD" action_name:="$PREFIX/execute_waypoint" \
 autonomy_enable_topic:="$PREFIX/autonomy_enable" manual_takeover_topic:="$PREFIX/manual_takeover" chassis_ready_topic:="$PREFIX/chassis_ready" \
 command_timeout_s:="${NAV_COMMAND_TIMEOUT_S:-0.25}" \
 max_linear_mps:="${NAV_MAX_SPEED_MPS:-0.15}" max_angular_rps:="${NAV_MAX_ANGULAR_RPS:-0.35}"
fi
printf 'mode=%s bridge=%s output=%s/cmd_vel logs=%s\n' "$MODE" "$LIVE" "$PREFIX" "$RUN"
# Any failed child tears down the entire control chain, including the mux.
set +e
wait -n "${PIDS[@]}"
status=$?
for log in "$RUN"/*.log; do echo "Component log: $log"; tail -12 "$log"; done
exit "$status"
