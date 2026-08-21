#!/usr/bin/env bash
set -euo pipefail

# Observation-only soak monitor.  It reads ROS rates, process usage and
# tegrastats; it never publishes a ROS message.
DURATION_S="${RESOURCE_MONITOR_DURATION_S:-60}"
INTERVAL_S="${RESOURCE_MONITOR_INTERVAL_S:-1}"
OUT_DIR="${RESOURCE_MONITOR_DIR:-${PWD}/logs/diagnostics/resources}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/cloud_registered_body}"
ODOM_TOPIC="${ODOM_TOPIC:-/aft_mapped_to_init}"
LIO_PID="${POINT_LIO_PID:-$(pgrep -n -f '[p]ointlio_mapping' || true)}"
mkdir -p "${OUT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUT_DIR}/${STAMP}"
mkdir -p "${RUN_DIR}"

if [[ -f "${ROS_SETUP}" ]]; then
  # The monitor is also intended to be called directly from SSH, outside the
  # container entrypoint, so make standard ROS CLI tools available here.
  set +u
  source "${ROS_SETUP}"
  set -u
fi

echo "timestamp_utc=${STAMP}" | tee "${RUN_DIR}/run.info"
echo "duration_s=${DURATION_S}" | tee -a "${RUN_DIR}/run.info"
echo "cloud_topic=${CLOUD_TOPIC}" | tee -a "${RUN_DIR}/run.info"
echo "odom_topic=${ODOM_TOPIC}" | tee -a "${RUN_DIR}/run.info"
echo "lio_pid=${LIO_PID:-<not-found>}" | tee -a "${RUN_DIR}/run.info"
cat /proc/sys/kernel/core_pattern > "${RUN_DIR}/core_pattern.txt" 2>/dev/null || true

if command -v tegrastats >/dev/null 2>&1; then
  timeout --kill-after=3s "${DURATION_S}" tegrastats --interval "$((INTERVAL_S * 1000))" > "${RUN_DIR}/tegrastats.log" 2>&1 &
  TEGRA_PID=$!
else
  TEGRA_PID=""
  echo "tegrastats=absent" > "${RUN_DIR}/tegrastats.log"
fi

timeout --kill-after=3s "${DURATION_S}" ros2 topic hz --window 30 "${CLOUD_TOPIC}" > "${RUN_DIR}/cloud_hz.log" 2>&1 &
CLOUD_HZ_PID=$!
timeout --kill-after=3s "${DURATION_S}" ros2 topic hz --window 100 "${ODOM_TOPIC}" > "${RUN_DIR}/odom_hz.log" 2>&1 &
ODOM_HZ_PID=$!

deadline=$((SECONDS + DURATION_S))
{
  printf 'epoch pid cpu_percent mem_percent rss_kb stat\n'
  while ((SECONDS < deadline)); do
    if [[ -n "${LIO_PID}" ]]; then
      ps -o etime=,pid=,pcpu=,pmem=,rss=,stat= -p "${LIO_PID}" || true
    fi
    sleep "${INTERVAL_S}"
  done
} > "${RUN_DIR}/process.log"

wait "${CLOUD_HZ_PID}" || true
wait "${ODOM_HZ_PID}" || true
if [[ -n "${TEGRA_PID}" ]]; then wait "${TEGRA_PID}" || true; fi

echo "run_dir=${RUN_DIR}"
echo "--- cloud rate tail ---"
tail -n 8 "${RUN_DIR}/cloud_hz.log" || true
echo "--- odom rate tail ---"
tail -n 8 "${RUN_DIR}/odom_hz.log" || true
echo "--- resource run ---"
cat "${RUN_DIR}/run.info"
