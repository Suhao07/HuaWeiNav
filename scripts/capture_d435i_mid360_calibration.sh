#!/usr/bin/env bash
set -Eeuo pipefail

# Record-only D435i + MID-360 calibration data.  This script never publishes a
# waypoint or velocity command.  Existing host-owned sensor processes are
# observed by default; optional driver startup requires an explicit command.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE_PATH="${REPO_ROOT}/real_robot/profiles/orin26_livox_mid360_d435i.env"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

duration_s="${CALIBRATION_CAPTURE_DURATION_S:-144}"
phase_duration_s="${CALIBRATION_PHASE_DURATION_S:-8}"
output_root="${CALIBRATION_OUTPUT_ROOT:-${REPO_ROOT}/real_robot/calibration/raw_bags}"
output_dir=""
bag_name=""
allow_nonaligned_depth="${ALLOW_NONALIGNED_DEPTH:-0}"
start_lio="${START_LIO:-0}"
start_d435i="${START_D435I:-0}"
lio_start_cmd="${LIO_START_CMD:-}"
d435i_start_cmd="${D435I_START_CMD:-}"
sample_timeout_s="${CALIBRATION_SAMPLE_TIMEOUT_S:-8}"
non_interactive="${CALIBRATION_NON_INTERACTIVE:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/capture_d435i_mid360_calibration.sh [options]

Record a synchronized RGB-D + CameraInfo + Livox + IMU + odometry bag for
D435i--MID-360 calibration.  The default is observe-only: it does not start
drivers, semantic mapping, a waypoint adapter, a planner, or any controller.

Options:
  --duration SEC              Total capture duration (default: 144)
  --phase-duration SEC        Duration of each of 18 coverage phases (default: 8)
  --output DIR                Output directory containing bag and evidence
  --bag-name NAME             Bag directory name (default: timestamped)
  --allow-nonaligned-depth    Explicitly allow a depth topic whose name is not
                              aligned_depth_to_color; the manifest records this
                              as operator-declared and it remains a calibration risk.
  --start-lio                 Run LIO_START_CMD before recording
  --start-d435i               Run D435I_START_CMD before recording
  --non-interactive           Do not wait for operator confirmation between phases
  -h, --help                  Show this help

Topic overrides are environment variables:
  RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO_TOPIC, LIDAR_TOPIC, IMU_TOPIC,
  CLOUD_TOPIC, ODOM_TOPIC, TF_TOPIC, TF_STATIC_TOPIC.

Optional startup commands (never inferred):
  LIO_START_CMD='RESTART_EXISTING=0 ENABLE_CLOUD_PUBLISH=1 ENABLE_BODY_CLOUD_PUBLISH=1 bash scripts/start_orin_lio_for_strive.sh'
  D435I_START_CMD='ros2 launch realsense2_camera rs_launch.py ...'

The 18 phases cover 3 distances x 3 target orientations x 2 target poses.
The operator moves/holds the calibration target; this script only sleeps while
ros2 bag record captures data.
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

die() {
  echo "[calibration-capture] $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --duration)
      (($# >= 2)) || die "--duration requires seconds"
      duration_s="$2"; shift 2 ;;
    --phase-duration)
      (($# >= 2)) || die "--phase-duration requires seconds"
      phase_duration_s="$2"; shift 2 ;;
    --output)
      (($# >= 2)) || die "--output requires a directory"
      output_dir="$2"; shift 2 ;;
    --bag-name)
      (($# >= 2)) || die "--bag-name requires a name"
      bag_name="$2"; shift 2 ;;
    --allow-nonaligned-depth)
      allow_nonaligned_depth=1; shift ;;
    --start-lio)
      start_lio=1; shift ;;
    --start-d435i)
      start_d435i=1; shift ;;
    --non-interactive)
      non_interactive=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "unknown argument: $1 (use --help)" ;;
  esac
done

[[ "${duration_s}" =~ ^[1-9][0-9]*$ ]] || die "duration must be a positive integer"
[[ "${phase_duration_s}" =~ ^[1-9][0-9]*$ ]] || die "phase-duration must be a positive integer"
((duration_s >= phase_duration_s * 18)) || die "duration must cover all 18 phases"

if [[ -f "${PROFILE_PATH}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${PROFILE_PATH}"
  set +a
fi

# Profile values are defaults only; explicit capture variables win.
RGB_TOPIC="${RGB_TOPIC:-${CAMERA_TOPIC:-/camera/d435i/d435i_camera/color/image_raw}}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/d435i/d435i_camera/color/camera_info}"
DEPTH_TOPIC="${DEPTH_TOPIC:-}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
IMU_TOPIC="${IMU_TOPIC:-/livox/imu}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/cloud_registered_body}"
ODOM_TOPIC="${ODOM_TOPIC:-/aft_mapped_to_init}"
TF_TOPIC="${TF_TOPIC:-/tf}"
TF_STATIC_TOPIC="${TF_STATIC_TOPIC:-/tf_static}"

if [[ -f "${ROS_SETUP}" ]]; then
  # ROS setup is idempotent and gives ros2 bag the message/type environment.
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
fi
command -v ros2 >/dev/null 2>&1 || die "ros2 is not available; run this on the ROS 2 robot environment"
command -v timeout >/dev/null 2>&1 || die "timeout is required"

topic_line() {
  local topic="$1"
  ros2 topic list -t 2>/dev/null | awk -v wanted="${topic}" '$1 == wanted { print; exit }'
}

topic_type() {
  local line
  line="$(topic_line "$1")"
  [[ -n "${line}" ]] || return 1
  printf '%s\n' "${line}" | sed -n 's/.*\[\(.*\)\].*/\1/p'
}

require_topic() {
  local topic="$1"
  local expected_type="$2"
  local line type
  line="$(topic_line "${topic}")"
  [[ -n "${line}" ]] || die "topic is not advertised: ${topic}"
  type="$(topic_type "${topic}")"
  [[ "${type}" == "${expected_type}" ]] || die "${topic} has type '${type}', expected '${expected_type}'"
  printf '[calibration-capture] %s [%s]\n' "${topic}" "${type}"
}

sample_topic() {
  local topic="$1"
  local sample_file="$2"
  if ! timeout --kill-after=2s "${sample_timeout_s}" ros2 topic echo --once "${topic}" >"${sample_file}" 2>&1; then
    sed -n '1,12p' "${sample_file}" >&2 || true
    die "no actual message received on ${topic} within ${sample_timeout_s}s"
  fi
}

choose_depth_topic() {
  [[ -n "${DEPTH_TOPIC}" ]] && return 0
  local candidate
  for candidate in \
    /camera/d435i/d435i_camera/aligned_depth_to_color/image_raw \
    /camera/veocc_d435i/aligned_depth_to_color/image_raw \
    /camera/d435i/d435i_camera/depth/image_rect_raw; do
    if [[ -n "$(topic_line "${candidate}")" ]]; then
      DEPTH_TOPIC="${candidate}"
      return 0
    fi
  done
  die "no D435i depth image topic found; set DEPTH_TOPIC explicitly"
}

if is_true "${start_lio}"; then
  [[ -n "${lio_start_cmd}" ]] || die "--start-lio requires LIO_START_CMD; no LIO command is inferred"
fi
if is_true "${start_d435i}"; then
  [[ -n "${d435i_start_cmd}" ]] || die "--start-d435i requires D435I_START_CMD; no camera command is inferred"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${output_root}/d435i_mid360_${timestamp}"
fi
if [[ -z "${bag_name}" ]]; then
  bag_name="d435i_mid360_${timestamp}"
fi
mkdir -p "${output_dir}"
bag_dir="${output_dir}/${bag_name}"
[[ ! -e "${bag_dir}" ]] || die "bag output already exists: ${bag_dir}"

lio_pid=""
d435i_pid=""
bag_pid=""
cleanup() {
  set +e
  if [[ -n "${bag_pid}" ]] && kill -0 "${bag_pid}" 2>/dev/null; then
    kill -INT "${bag_pid}" 2>/dev/null || true
    wait "${bag_pid}" 2>/dev/null || true
  fi
  for pid in "${d435i_pid}" "${lio_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if is_true "${start_lio}"; then
  echo "[calibration-capture] starting explicitly supplied LIO command"
  bash -lc "${lio_start_cmd}" >"${output_dir}/lio_start.log" 2>&1 &
  lio_pid=$!
  sleep 3
fi
if is_true "${start_d435i}"; then
  echo "[calibration-capture] starting explicitly supplied D435i command"
  bash -lc "${d435i_start_cmd}" >"${output_dir}/d435i_start.log" 2>&1 &
  d435i_pid=$!
  sleep 3
fi

choose_depth_topic
require_topic "${RGB_TOPIC}" sensor_msgs/msg/Image
require_topic "${DEPTH_TOPIC}" sensor_msgs/msg/Image
require_topic "${CAMERA_INFO_TOPIC}" sensor_msgs/msg/CameraInfo
require_topic "${LIDAR_TOPIC}" livox_ros_driver2/msg/CustomMsg
require_topic "${IMU_TOPIC}" sensor_msgs/msg/Imu
require_topic "${CLOUD_TOPIC}" sensor_msgs/msg/PointCloud2
require_topic "${ODOM_TOPIC}" nav_msgs/msg/Odometry
rgb_type="$(topic_type "${RGB_TOPIC}")"
depth_type="$(topic_type "${DEPTH_TOPIC}")"
camera_info_type="$(topic_type "${CAMERA_INFO_TOPIC}")"
lidar_type="$(topic_type "${LIDAR_TOPIC}")"
imu_type="$(topic_type "${IMU_TOPIC}")"
cloud_type="$(topic_type "${CLOUD_TOPIC}")"
odom_type="$(topic_type "${ODOM_TOPIC}")"

depth_alignment="aligned_depth_to_color"
if [[ "${DEPTH_TOPIC}" != *aligned_depth_to_color* ]]; then
  depth_alignment="operator_declared_alignment"
  if ! is_true "${allow_nonaligned_depth}"; then
    die "depth topic is not named aligned_depth_to_color; use --allow-nonaligned-depth only after confirming the driver alignment setting"
  fi
fi

echo "[calibration-capture] checking actual samples"
sample_topic "${RGB_TOPIC}" "${output_dir}/sample_rgb.txt"
sample_topic "${DEPTH_TOPIC}" "${output_dir}/sample_depth.txt"
sample_topic "${CAMERA_INFO_TOPIC}" "${output_dir}/camera_info_sample.yaml"
sample_topic "${LIDAR_TOPIC}" "${output_dir}/sample_livox.txt"
sample_topic "${IMU_TOPIC}" "${output_dir}/sample_imu.txt"
sample_topic "${CLOUD_TOPIC}" "${output_dir}/sample_cloud.txt"
sample_topic "${ODOM_TOPIC}" "${output_dir}/sample_odom.txt"

if [[ -n "$(topic_line /cmd_vel)" ]]; then
  echo "[calibration-capture] warning: /cmd_vel is advertised; this script will not publish it" >&2
fi

mapfile -t record_topics <<EOF
${RGB_TOPIC}
${DEPTH_TOPIC}
${CAMERA_INFO_TOPIC}
${LIDAR_TOPIC}
${IMU_TOPIC}
${CLOUD_TOPIC}
${ODOM_TOPIC}
${TF_TOPIC}
${TF_STATIC_TOPIC}
EOF

cat >"${output_dir}/capture_manifest.yaml" <<EOF
capture:
  type: d435i_mid360_calibration
  started_utc: ${timestamp}
  duration_s: ${duration_s}
  phase_duration_s: ${phase_duration_s}
  coverage_grid: 3_distances_x_3_orientations_x_2_poses
  depth_alignment: ${depth_alignment}
  bag_path: ${bag_dir}
topics:
  rgb: ${RGB_TOPIC}
  depth: ${DEPTH_TOPIC}
  camera_info: ${CAMERA_INFO_TOPIC}
  lidar: ${LIDAR_TOPIC}
  imu: ${IMU_TOPIC}
  cloud_registered_body: ${CLOUD_TOPIC}
  odom: ${ODOM_TOPIC}
  tf: ${TF_TOPIC}
  tf_static: ${TF_STATIC_TOPIC}
types:
  rgb: ${rgb_type}
  depth: ${depth_type}
  camera_info: ${camera_info_type}
  lidar: ${lidar_type}
  imu: ${imu_type}
  cloud_registered_body: ${cloud_type}
  odom: ${odom_type}
EOF
ros2 topic list -t >"${output_dir}/topic_list_before_record.txt"

echo "[calibration-capture] recording to ${bag_dir}"
ros2 bag record --output "${bag_dir}" "${record_topics[@]}" >"${output_dir}/ros2_bag_record.log" 2>&1 &
bag_pid=$!
sleep 2
kill -0 "${bag_pid}" 2>/dev/null || die "ros2 bag record exited before capture"

phases=(
  "near/front/pose-A" "near/left/pose-A" "near/right/pose-A"
  "near/front/pose-B" "near/left/pose-B" "near/right/pose-B"
  "middle/front/pose-A" "middle/left/pose-A" "middle/right/pose-A"
  "middle/front/pose-B" "middle/left/pose-B" "middle/right/pose-B"
  "far/front/pose-A" "far/left/pose-A" "far/right/pose-A"
  "far/front/pose-B" "far/left/pose-B" "far/right/pose-B"
)
printf 'phase_index,phase_label,started_utc,duration_s\n' >"${output_dir}/phase_log.csv"
for index in "${!phases[@]}"; do
  phase_label="${phases[$index]}"
  printf '[calibration-capture] phase %02d/18: %s — position target, then hold it still\n' "$((index + 1))" "${phase_label}"
  if ! is_true "${non_interactive}"; then
    read -r -p "Press Enter to record this phase (${phase_duration_s}s)... " || true
  fi
  phase_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%d,%s,%s,%s\n' "$((index + 1))" "${phase_label}" "${phase_start}" "${phase_duration_s}" >>"${output_dir}/phase_log.csv"
  sleep "${phase_duration_s}"
done

# If duration has deliberately been extended, preserve the requested total.
minimum_s=$((phase_duration_s * 18))
if ((duration_s > minimum_s)); then
  sleep "$((duration_s - minimum_s))"
fi

echo "[calibration-capture] stopping bag"
kill -INT "${bag_pid}" 2>/dev/null || true
wait "${bag_pid}" 2>/dev/null || true
bag_pid=""
ros2 bag info "${bag_dir}" >"${output_dir}/bag_info.txt" 2>&1 || {
  echo "[calibration-capture] ros2 bag info failed; inspect ros2_bag_record.log" >&2
  exit 3
}

ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'ended_utc: %s\n' "${ended}" >>"${output_dir}/capture_manifest.yaml"
printf 'bag_info: %s/bag_info.txt\n' "${output_dir}" >>"${output_dir}/capture_manifest.yaml"
echo "[calibration-capture] complete"
echo "bag=${bag_dir}"
echo "manifest=${output_dir}/capture_manifest.yaml"
echo "phase_log=${output_dir}/phase_log.csv"
