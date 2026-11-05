#!/usr/bin/env bash
# Live perception/localization inputs; no bridge, motion server or physical topics.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
source "${STRIVE_OVERLAY_SETUP:-${ROOT}/real_robot/ros2_ws/install/setup.bash}"
set -u
OUT="${SHADOW_RUN_DIRECTORY:-/tmp/strive-shadow-acceptance}"
mkdir -p "$OUT"
PREFIX="$(ros2 pkg prefix local_planner)"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT TERM
ODOM="${SHADOW_ODOM_TOPIC:-/aft_mapped_to_init}"
CLOUD="${SHADOW_CLOUD_TOPIC:-/cloud_registered}"
"$PREFIX/lib/local_planner/localPlanner" --ros-args -r __node:=strive_shadow_planner \
 -p pathFolder:="$PREFIX/share/local_planner/paths" -p useTerrainAnalysis:=false \
 -p autonomyMode:=true -p autonomySpeed:="${SHADOW_SPEED_MPS:-0.15}" -p maxSpeed:="${SHADOW_SPEED_MPS:-0.15}" \
 -p vehicleLength:="${SHADOW_VEHICLE_LENGTH_M:-0.6}" -p vehicleWidth:="${SHADOW_VEHICLE_WIDTH_M:-0.6}" \
 -p checkObstacle:=true -p checkRotObstacle:=true \
 -p status_topic:=/strive/shadow/status -p cancel_topic:=/strive/shadow/cancel \
 -r /state_estimation:=/strive/shadow/base_odom -r /registered_scan:="$CLOUD" \
 -r /way_point:=/strive/shadow/way_point -r /path:=/strive/shadow/path \
 -r /joy:=/strive/shadow/joy -r /speed:=/strive/shadow/speed \
 >"$OUT/planner.log" 2>&1 & PIDS+=($!)
"$PREFIX/lib/local_planner/pathFollower" --ros-args -r __node:=strive_shadow_follower \
 -p autonomyMode:=true -p autonomySpeed:="${SHADOW_SPEED_MPS:-0.15}" -p maxSpeed:="${SHADOW_SPEED_MPS:-0.15}" \
 -p maxYawRate:="${SHADOW_MAX_YAW_DEG_S:-20.0}" -p maxAccel:="${SHADOW_MAX_ACCEL_MPS2:-0.15}" \
 -p manual_cmd_topic:=/strive/shadow/manual_cmd \
 -r /state_estimation:=/strive/shadow/base_odom -r /path:=/strive/shadow/path \
 -r /cmd_vel/autonomy:=/strive/shadow/raw_cmd_vel -r /joy:=/strive/shadow/joy \
 >"$OUT/follower.log" 2>&1 & PIDS+=($!)
MOTION_PREFIX="$(ros2 pkg prefix strive_sysnav_motion)"
"$MOTION_PREFIX/lib/strive_sysnav_motion/safety_velocity_mux" --ros-args -r __node:=strive_shadow_mux \
 -p require_controller_contract:=false -p start_autonomy_enabled:=true \
 -p autonomy_cmd_topic:=/strive/shadow/raw_cmd_vel -p manual_cmd_topic:=/strive/shadow/manual_cmd \
 -p output_cmd_topic:=/strive/shadow/cmd_vel -p odom_topic:=/strive/shadow/base_odom -p pointcloud_topic:="$CLOUD" \
 -p autonomy_enable_topic:=/strive/shadow/enable -p manual_takeover_topic:=/strive/shadow/takeover \
 -p estop_topic:=/strive/shadow/estop -p estop_reset_topic:=/strive/shadow/estop_reset \
 -p hold_topic:=/strive/shadow/hold -p safety_state_topic:=/strive/shadow/safety \
 -p max_linear_speed_mps:="${SHADOW_SPEED_MPS:-0.15}" -p max_angular_speed_rps:=0.35 \
 -p max_linear_accel_mps2:="${SHADOW_MAX_ACCEL_MPS2:-0.15}" \
 -p max_angular_accel_rps2:="${SHADOW_MAX_ANGULAR_ACCEL_RPS2:-0.3}" \
 >"$OUT/mux.log" 2>&1 & PIDS+=($!)
python3 "$ROOT/scripts/probe_real_robot_shadow.py" \
 --decisions "${SHADOW_DECISIONS:?Set SHADOW_DECISIONS to the active runtime decisions JSONL}" \
 --output "$OUT/result.json" --standoff "${SHADOW_STANDOFF_M:-0.8}" \
 --base-in-lidar "${SHADOW_BASE_X_M:-0.0}" "${SHADOW_BASE_Y_M:-0.0}" "${SHADOW_BASE_YAW_RAD:-0.0}" \
 --odom "$ODOM" --cloud "$CLOUD" --duration "${SHADOW_DURATION_S:-35}" "$@"
