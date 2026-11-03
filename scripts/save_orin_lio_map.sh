#!/usr/bin/env bash
set -euo pipefail

# 结束 LIO_PCD_SAVE_EN=1 的建图会话并取出地图。Point-LIO 只在 SIGINT 正常退出时
# 写 scans.pcd；tmux kill-session（SIGHUP）会直接丢图，所以这里只向其窗格发 Ctrl-C。
# 地图坐标系 = 本次建图会话启动时雷达的位姿（camera_init）。

TMUX_SESSION="${TMUX_SESSION:-livox_odom}"
LIO_PANE="${LIO_PANE:-${TMUX_SESSION}:0.1}"
POINT_LIO_PCD="${POINT_LIO_PCD:-/home/orin26/code/point_lio_ws/src/point_lio_ros2/PCD/scans.pcd}"
SAVE_TIMEOUT_S="${SAVE_TIMEOUT_S:-180}"

out_dir="${1:-}"
if [[ -z "${out_dir}" ]]; then
  echo "Usage: $0 <output_dir>" >&2
  exit 2
fi
if [[ -e "${out_dir}/lio_map.pcd" ]]; then
  echo "[save-lio-map] refuse to overwrite ${out_dir}/lio_map.pcd" >&2
  exit 2
fi
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "[save-lio-map] tmux session not found: ${TMUX_SESSION}" >&2
  exit 3
fi

session_created="$(tmux display-message -p -t "${TMUX_SESSION}" '#{session_created}')"
sent_at="$(date +%s)"
tmux send-keys -t "${LIO_PANE}" C-c
echo "[save-lio-map] sent SIGINT to ${LIO_PANE}; waiting for ${POINT_LIO_PCD}"

# 新图须晚于发信号时刻，且大小连续两次一致才算写完。
last_size=-1
deadline=$((sent_at + SAVE_TIMEOUT_S))
while :; do
  if [[ -f "${POINT_LIO_PCD}" ]] && (( $(stat -c %Y "${POINT_LIO_PCD}") >= sent_at )); then
    size="$(stat -c %s "${POINT_LIO_PCD}")"
    if (( size > 0 && size == last_size )); then
      break
    fi
    last_size="${size}"
  fi
  if (( $(date +%s) > deadline )); then
    echo "[save-lio-map] timeout: no new PCD (was LIO started with LIO_PCD_SAVE_EN=1?)" >&2
    exit 4
  fi
  sleep 2
done

mkdir -p "${out_dir}"
cp "${POINT_LIO_PCD}" "${out_dir}/lio_map.pcd"
sha="$(sha256sum "${out_dir}/lio_map.pcd" | cut -d' ' -f1)"
python3 - "${out_dir}/lio_map.json" <<EOF
import json, sys
json.dump({
    "pcd": "lio_map.pcd",
    "sha256": "${sha}",
    "size_bytes": ${last_size},
    "frame_id": "camera_init",
    "frame_origin": "lidar pose when the mapping LIO session started",
    "mapping_session_created_unix": ${session_created},
    "saved_unix": ${sent_at},
    "source": "${POINT_LIO_PCD}",
}, open(sys.argv[1], "w"), indent=2)
EOF
echo "[save-lio-map] saved ${out_dir}/lio_map.pcd sha256=${sha}"
# 图已落盘；关掉残留的 livox 驱动窗格，否则 start_real_robot_sensors.sh 会复用旧会话而不进定位模式。
tmux kill-session -t "${TMUX_SESSION}"
echo "[save-lio-map] ${TMUX_SESSION} closed; restart with LIO_LOCATION_MODE=1 LIO_MAP_PATH=$(cd "${out_dir}" && pwd)/lio_map.pcd"
