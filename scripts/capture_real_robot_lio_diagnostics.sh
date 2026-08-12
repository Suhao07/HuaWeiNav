#!/usr/bin/env bash
# Capture a bounded, read-only LIO/DDS evidence report for a real robot.
# It never starts, stops, or reconfigures external sensor or LIO processes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
LIVOX_SETUP="${LIVOX_SETUP:-/home/orin26/code/ws_livox/install/setup.bash}"
POINT_LIO_SETUP="${POINT_LIO_SETUP:-/home/orin26/code/point_lio_ws/install/setup.bash}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/cloud_registered}"
ODOM_TOPIC="${ODOM_TOPIC:-/aft_mapped_to_init}"
POSE_TOPIC="${POSE_TOPIC:-${ODOM_TOPIC}}"
LIO_INPUT_MODE="${LIO_INPUT_MODE:-cloud_and_pose}"
LIO_LIDAR_TOPIC="${LIO_LIDAR_TOPIC:-/livox/lidar}"
LIO_IMU_TOPIC="${LIO_IMU_TOPIC:-/livox/imu}"
POINT_LIO_NODE_NAME="${POINT_LIO_NODE_NAME:-/laserMapping}"
HOST_FASTDDS_BUILTIN_TRANSPORTS="${HOST_FASTDDS_BUILTIN_TRANSPORTS:-}"
SAMPLE_TIMEOUT_S="${LIO_DIAGNOSTIC_TIMEOUT_S:-8}"
OUTPUT_DIR="${LIO_DIAGNOSTIC_OUTPUT_DIR:-${REPO_ROOT}/logs/diagnostics}"

usage() {
  cat <<'EOF'
Usage: scripts/capture_real_robot_lio_diagnostics.sh [REPORT_PATH]

Writes a bounded, read-only report of Fast DDS/ROS settings, discovered topic
QoS, and one header sample for each configured Livox/Point-LIO topic.  The
default report path is under this deployment workspace's logs/diagnostics/.

No external process is started, stopped, or modified.  A sample timeout proves
only that this independent subscriber did not receive data; it must keep
semantic mapping disabled until resolved.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source_ros() {
  if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS setup not found: ${ROS_SETUP}" >&2
    return 2
  fi
  set +u
  source "${ROS_SETUP}"
  [[ -f "${LIVOX_SETUP}" ]] && source "${LIVOX_SETUP}"
  [[ -f "${POINT_LIO_SETUP}" ]] && source "${POINT_LIO_SETUP}"
  set -u
}

capture() {
  local label="$1"
  shift
  local output rc
  set +e
  output="$("$@" 2>&1)"
  rc=$?
  set -e
  printf '\n## %s (rc=%s)\n%s\n' "${label}" "${rc}" "${output}"
  return 0
}

capture_sample() {
  local topic="$1"
  local reliability="$2"
  local output rc
  set +e
  output="$(ROS_DISABLE_DAEMON=1 timeout --kill-after=2s "${SAMPLE_TIMEOUT_S}" \
    ros2 topic echo --once "${topic}" --field header \
      --qos-reliability "${reliability}" 2>&1)"
  rc=$?
  set -e
  printf '\n### header sample: %s (qos=%s, rc=%s)\n%s\n' \
    "${topic}" "${reliability}" "${rc}" "${output}"
  if [[ "${rc}" == "0" && "${output}" == *"stamp:"* ]]; then
    printf 'RESULT: PASS (actual header received)\n'
  else
    printf 'RESULT: FAIL (no actual header received; do not enable semantic mapping)\n'
  fi
}

mkdir -p "${OUTPUT_DIR}"
report_path="${1:-${OUTPUT_DIR}/lio_dds_$(date -u +%Y%m%dT%H%M%SZ).md}"
if [[ "${report_path}" != "${REPO_ROOT}"/* ]]; then
  echo "Report path must remain inside REPO_ROOT: ${report_path}" >&2
  exit 2
fi

source_ros
case "${LIO_INPUT_MODE}" in
  pose_only|cloud_and_pose|disabled) ;;
  *) echo "Unsupported LIO_INPUT_MODE=${LIO_INPUT_MODE}" >&2; exit 2 ;;
esac
# FASTDDS_BUILTIN_TRANSPORTS=UDPv4 is deliberately set for the deployment
# container.  Do not leak that container workaround into a host-side LIO
# diagnostic: the externally owned LIO processes on this robot use Fast DDS'
# host default transport.  A profile may set HOST_FASTDDS_BUILTIN_TRANSPORTS
# explicitly if a future robot requires a host override.
if [[ -n "${HOST_FASTDDS_BUILTIN_TRANSPORTS}" ]]; then
  export FASTDDS_BUILTIN_TRANSPORTS="${HOST_FASTDDS_BUILTIN_TRANSPORTS}"
else
  unset FASTDDS_BUILTIN_TRANSPORTS
fi
export ROS_DISABLE_DAEMON=1
lio_pids="$(pgrep -d, -f '[/]livox_ros_driver2_node|[/]pointlio_mapping' || true)"
lio_pids="${lio_pids:-0}"

{
  printf '# Real-robot LIO / DDS diagnostic report\n\n'
  printf -- '- generated_utc: `%s`\n' "$(date -u +%FT%TZ)"
  printf -- '- repo_root: `%s`\n' "${REPO_ROOT}"
  printf -- '- read_only: `true`\n'
  printf -- '- ros_domain_id: `%s`\n' "${ROS_DOMAIN_ID:-<unset>}"
  printf -- '- rmw_implementation: `%s`\n' "${RMW_IMPLEMENTATION:-<unset>}"
  printf -- '- fastdds_builtin_transports: `%s`\n' "${FASTDDS_BUILTIN_TRANSPORTS:-<unset>}"
  printf -- '- ros_localhost_only: `%s`\n' "${ROS_LOCALHOST_ONLY:-<unset>}"
  printf -- '- sample_timeout_s: `%s`\n' "${SAMPLE_TIMEOUT_S}"
  printf -- '- point_lio_node: `%s`\n' "${POINT_LIO_NODE_NAME}"
  printf -- '- lio_input_mode: `%s`\n' "${LIO_INPUT_MODE}"
  printf -- '- pose_topic: `%s`\n' "${POSE_TOPIC}"

  capture "LIO process snapshot" ps -o pid,etime,pcpu,pmem,stat,command -p "${lio_pids}"

  for topic in "${LIO_LIDAR_TOPIC}" "${LIO_IMU_TOPIC}" "${POSE_TOPIC}"; do
    capture "topic info: ${topic}" ros2 topic info -v "${topic}"
  done
  if [[ "${LIO_INPUT_MODE}" == "cloud_and_pose" ]]; then
    capture "topic info: ${CLOUD_TOPIC}" ros2 topic info -v "${CLOUD_TOPIC}"
  else
    printf '\n## topic info: %s (optional in pose_only mode)\n' "${CLOUD_TOPIC}"
    capture "topic info: ${CLOUD_TOPIC}" ros2 topic info -v "${CLOUD_TOPIC}"
  fi

  # A ROS publisher endpoint alone does not mean Point-LIO emits a cloud.  In
  # particular, scan_publish_en=false leaves /cloud_registered discoverable
  # while suppressing its actual samples.  Querying parameters is read-only.
  for parameter in \
    publish.scan_publish_en \
    publish.path_en \
    publish.scan_bodyframe_pub_en \
    publish.odometry.publish_odometry_without_downsample; do
    capture "Point-LIO parameter: ${parameter}" \
      timeout --kill-after=2s "${SAMPLE_TIMEOUT_S}" \
      ros2 param get "${POINT_LIO_NODE_NAME}" "${parameter}"
  done

  # Point-LIO and Livox endpoint QoS on this robot are inspected above.  The
  # chosen profiles deliberately cover the actual endpoint reliability modes.
  if [[ "${LIO_INPUT_MODE}" != "disabled" ]]; then
    capture_sample "${POSE_TOPIC}" "reliable"
  fi
  if [[ "${LIO_INPUT_MODE}" == "cloud_and_pose" ]]; then
    capture_sample "${CLOUD_TOPIC}" "reliable"
  else
    printf '\n### header sample: %s (optional in pose_only mode)\n' "${CLOUD_TOPIC}"
    printf 'RESULT: SKIP (registered cloud is not consumed by this profile)\n'
  fi
} >"${report_path}"

printf 'LIO diagnostics report: %s\n' "${report_path}"
grep -E '^RESULT:' "${report_path}" || true
