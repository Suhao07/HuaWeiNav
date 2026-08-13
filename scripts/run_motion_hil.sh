#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${ROOT_DIR}/real_robot/ros2_ws}"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
SCENARIO="${1:-${STRIVE_HIL_SCENARIO:-reached}}"
OVERLAY_SETUP="${STRIVE_OVERLAY_SETUP:-${WS_DIR}/install/setup.bash}"
HIL_ARTIFACT_DIR="${STRIVE_HIL_ARTIFACT_DIR:-/tmp}"

if [[ ! -f "${ROS_SETUP}" || ! -f "${OVERLAY_SETUP}" ]]; then
  echo "ROS or STRIVE overlay setup is missing; build the ROS workspace first." >&2
  exit 2
fi
mkdir -p "${HIL_ARTIFACT_DIR}"

set +u
source "${ROS_SETUP}"
source "${OVERLAY_SETUP}"
set -u

SERVER_PID=""
PLANNER_PID=""
FOLLOWER_PID=""
MUX_PID=""
cleanup() {
  if [[ -n "${PLANNER_PID}" ]] && kill -0 "${PLANNER_PID}" >/dev/null 2>&1; then
    kill "${PLANNER_PID}" >/dev/null 2>&1 || true
    wait "${PLANNER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  for pid in "${FOLLOWER_PID:-}" "${MUX_PID:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

# 中文说明：HIL 只启动任务级 MotionServer 和反馈模拟器，不启动
# localPlanner/pathFollower/SafetyVelocityMux，也不会产生任何 /cmd_vel。
ros2 run strive_sysnav_motion sysnav_motion_server \
  --ros-args \
  -p planner_status_topic:=/local_planner/status \
  -p safety_state_topic:=/platform/safety_state \
  -p stable_reach_time_s:=0.0 \
  -p require_controller_contract:=false \
  >/tmp/strive_motion_hil_server.log 2>&1 &
SERVER_PID=$!

sleep 1
if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
  echo "MotionServer exited before HIL client startup:" >&2
  sed -n '1,240p' /tmp/strive_motion_hil_server.log >&2 || true
  exit 1
fi

native_planner_flag=false
native_safety_flag=false
if [[ "${SCENARIO}" == "native_planner" || "${SCENARIO}" == "native_safety" ]]; then
  planner_prefix="$(ros2 pkg prefix local_planner)"
  # 中文说明：该场景启动迁移后的真实 localPlanner。HIL 只提供
  # odom/registered_scan，不伪造 /path 或 planner status。
  ros2 run local_planner localPlanner \
    --ros-args \
    -p pathFolder:="${planner_prefix}/share/local_planner/paths" \
    -p useTerrainAnalysis:=false \
    -p autonomyMode:=true \
    -p autonomySpeed:=0.5 \
    -p maxSpeed:=0.5 \
    -p cancel_topic:=/local_planner/cancel \
    -p status_topic:=/local_planner/status \
    -r /state_estimation:=/aft_mapped_to_init \
    -r /registered_scan:=/hil/registered_scan \
    -r /way_point:=/way_point \
    -r /path:=/path \
    >/tmp/strive_motion_hil_planner.log 2>&1 &
  PLANNER_PID=$!
  sleep 1
  if ! kill -0 "${PLANNER_PID}" >/dev/null 2>&1; then
    echo "localPlanner exited before native HIL client startup:" >&2
    sed -n '1,240p' /tmp/strive_motion_hil_planner.log >&2 || true
    exit 1
  fi
  native_planner_flag=true
fi

if [[ "${SCENARIO}" == "native_safety" ]]; then
  # 中文说明：此场景继续使用迁移后的 pathFollower，并将其候选速度
  # 交给唯一 SafetyVelocityMux；HIL 只观测最终命令，不拥有底盘输出。
  ros2 run local_planner pathFollower \
    --ros-args \
    -p autonomyMode:=true \
    -p autonomySpeed:=0.3 \
    -p maxSpeed:=0.5 \
    -p manual_cmd_topic:=/cmd_vel/manual \
    -r /state_estimation:=/aft_mapped_to_init \
    -r /path:=/path \
    -r /cmd_vel/autonomy:=/cmd_vel/autonomy \
    >/tmp/strive_motion_hil_follower.log 2>&1 &
  FOLLOWER_PID=$!
  ros2 run strive_sysnav_motion safety_velocity_mux \
    --ros-args \
    -p require_controller_contract:=false \
    -p autonomy_cmd_topic:=/cmd_vel/autonomy \
    -p manual_cmd_topic:=/cmd_vel/manual \
    -p output_cmd_topic:=/cmd_vel \
    -p autonomy_enable_topic:=/platform/autonomy_enable \
    -p manual_takeover_topic:=/platform/manual_takeover \
    -p estop_topic:=/platform/estop_active \
    -p odom_topic:=/aft_mapped_to_init \
    -p pointcloud_topic:=/hil/registered_scan \
    -p start_autonomy_enabled:=false \
    -p max_linear_speed_mps:=0.5 \
    -p max_angular_speed_rps:=1.0 \
    -p max_linear_accel_mps2:=0.5 \
    -p max_angular_accel_rps2:=1.0 \
    >/tmp/strive_motion_hil_mux.log 2>&1 &
  MUX_PID=$!
  sleep 1
  if ! kill -0 "${FOLLOWER_PID}" >/dev/null 2>&1 || ! kill -0 "${MUX_PID}" >/dev/null 2>&1; then
    echo "pathFollower or SafetyVelocityMux exited before native safety HIL startup:" >&2
    sed -n '1,240p' /tmp/strive_motion_hil_follower.log >&2 || true
    sed -n '1,240p' /tmp/strive_motion_hil_mux.log >&2 || true
    exit 1
  fi
  native_safety_flag=true
fi

hil_status=0
ros2 run strive_sysnav_motion motion_hil \
  --ros-args \
  -p scenario:="${SCENARIO}" \
  -p native_planner:="${native_planner_flag}" \
  -p native_safety:="${native_safety_flag}" \
  -p artifact_path:="${HIL_ARTIFACT_DIR}/strive_motion_hil_${SCENARIO}.json" \
  -p timeout_s:="$([[ "${SCENARIO}" == "native_safety" ]] && echo 10.0 || echo 1.0)" || hil_status=$?
if [[ "${hil_status}" -ne 0 ]]; then
  echo "Motion HIL failed; MotionServer log:" >&2
  sed -n '1,240p' /tmp/strive_motion_hil_server.log >&2 || true
fi
exit "${hil_status}"
