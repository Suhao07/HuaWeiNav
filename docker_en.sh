#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
ENV_FILE="${SYSNAV_ENV_FILE:-${HOME}/.huawei_nav_real.env}"

IMAGE_TAG="${IMAGE_TAG:-huawei-nav-real:orin}"
CONTAINER_NAME="${CONTAINER_NAME:-huawei-nav-real}"
ASSET_DIR="${SYSNAV_ASSET_DIR:-${REPO_ROOT}/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external}"

PLATFORM="${PLATFORM:-mecanum}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/cloud_registered}"
ODOM_TOPIC="${ODOM_TOPIC:-/aft_mapped_to_init}"
CAMERA_TOPIC="${CAMERA_TOPIC:-/camera/image}"
START_USB_CAM="${START_USB_CAM:-true}"
USB_VIDEO_DEVICE="${USB_VIDEO_DEVICE:-/dev/video0}"
USB_IMAGE_WIDTH="${USB_IMAGE_WIDTH:-1280}"
USB_IMAGE_HEIGHT="${USB_IMAGE_HEIGHT:-720}"
USB_PIXEL_FORMAT="${USB_PIXEL_FORMAT:-yuyv}"
START_LIO="${START_LIO:-1}"
FRAMEWORK_SCRIPT="${FRAMEWORK_SCRIPT:-/workspace/STRIVE/scripts/start_real_robot_framework.sh}"
BLOCK_LOWER_CONTROLLER="${BLOCK_LOWER_CONTROLLER:-1}"
ENABLE_LOWER_CONTROLLER="${ENABLE_LOWER_CONTROLLER:-0}"
LOWER_CONTROLLER_CMD="${LOWER_CONTROLLER_CMD:-}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

SYSNAV_DETECTOR_MODEL_TYPE="${SYSNAV_DETECTOR_MODEL_TYPE:-yoloe}"
SYSNAV_DETECTOR_MODEL_PATH="${SYSNAV_DETECTOR_MODEL_PATH:-${ASSET_DIR}/yoloe-11s-seg.pt}"
SYSNAV_SAM2_CHECKPOINT="${SYSNAV_SAM2_CHECKPOINT:-${ASSET_DIR}/sam2/checkpoints/sam2.1_hiera_base_plus.pt}"
SYSNAV_MOBILECLIP_BLT_PATH="${SYSNAV_MOBILECLIP_BLT_PATH:-${ASSET_DIR}/mobileclip_blt.pt}"
SYSNAV_MOBILECLIP_BLT_TS_PATH="${SYSNAV_MOBILECLIP_BLT_TS_PATH:-${ASSET_DIR}/mobileclip_blt.ts}"
SYSNAV_CLIP_VIT_B32_PATH="${SYSNAV_CLIP_VIT_B32_PATH:-${ASSET_DIR}/ViT-B-32.pt}"
FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
SYSNAV_MAPPING_EXECUTOR_THREADS="${SYSNAV_MAPPING_EXECUTOR_THREADS:-4}"

usage() {
  cat <<EOF
Usage: ./docker_en.sh <command>

Commands:
  start       Start the real-robot container; starts LIO first unless START_LIO=0.
  enter       Enter the running container with bash.
  exec CMD    Run CMD inside the running container.
  stop        Stop and remove the real-robot container.
  restart     Stop then start.
  logs        Follow container logs.
  status      Show image/container/LIO status.
  smoke       Run bounded real-robot smoke checks on the host.
  start-lio   Start/restart host Livox + Point-LIO helper only.
  stop-lio    Stop the host livox_odom tmux session.

Configuration:
  Image/container: IMAGE_TAG=${IMAGE_TAG}, CONTAINER_NAME=${CONTAINER_NAME}
  Env file       : SYSNAV_ENV_FILE=${ENV_FILE}
  Framework      : FRAMEWORK_SCRIPT=${FRAMEWORK_SCRIPT}
  Control mode   : BLOCK_LOWER_CONTROLLER=${BLOCK_LOWER_CONTROLLER}
EOF
}

sudo_run() {
  if [[ -n "${SUDO_STDIN_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_STDIN_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sudo_run docker "$@"
  fi
}

section() {
  printf '\n== %s ==\n' "$1"
}

ensure_image() {
  if ! docker_cmd image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Docker image not found: ${IMAGE_TAG}" >&2
    echo "Build or tag the final Orin image first." >&2
    exit 2
  fi
}

ensure_assets() {
  local missing=0
  for path in \
    "${SYSNAV_DETECTOR_MODEL_PATH}" \
    "${SYSNAV_SAM2_CHECKPOINT}" \
    "${SYSNAV_MOBILECLIP_BLT_TS_PATH}" \
    "${SYSNAV_CLIP_VIT_B32_PATH}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing required asset: ${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    exit 3
  fi
}

append_jetson_nvidia_library_args() {
  local -n args_ref=$1
  local cuda_home="${CUDA_HOME_HOST:-}"
  local ld_paths=()

  if [[ -z "${cuda_home}" && -e /usr/local/cuda ]]; then
    cuda_home="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
  fi
  if [[ -n "${cuda_home}" && -d "${cuda_home}" ]]; then
    args_ref+=(-v "${cuda_home}:${cuda_home}:ro")
    ld_paths+=("${cuda_home}/lib64" "${cuda_home}/targets/aarch64-linux/lib")
  fi

  for lib in /usr/lib/aarch64-linux-gnu/libcudnn*.so*; do
    [[ -e "${lib}" ]] || continue
    args_ref+=(-v "${lib}:${lib}:ro")
  done
  ld_paths+=("/usr/lib/aarch64-linux-gnu" "/usr/lib/aarch64-linux-gnu/tegra")

  local cudss_dir="${CUDSS_HOST_DIR:-/opt/nvidia/cudss/lib}"
  if [[ -d "${cudss_dir}" ]]; then
    args_ref+=(-v "${cudss_dir}:/opt/nvidia/cudss/lib:ro")
    ld_paths+=("/opt/nvidia/cudss/lib")
  fi

  if ((${#ld_paths[@]})); then
    local joined
    joined="$(IFS=:; echo "${ld_paths[*]}")"
    args_ref+=(-e "LD_LIBRARY_PATH=${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")
  fi
}

append_unique_dir_mount() {
  local -n args_ref=$1
  local -n mounted_ref=$2
  local path="$3"
  [[ -n "${path}" && -f "${path}" ]] || return 0
  local dir
  dir="$(cd "$(dirname "${path}")" && pwd)"
  if [[ -z "${mounted_ref[${dir}]:-}" ]]; then
    args_ref+=(-v "${dir}:${dir}:ro")
    mounted_ref["${dir}"]=1
  fi
}

docker_args() {
  DOCKER_RUN_ARGS=(
    --name "${CONTAINER_NAME}"
    --network host
    --ipc=host
    --restart unless-stopped
    -e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}"
    -e "PYTHONUNBUFFERED=1"
    -e "SYSNAV_DETECTOR_MODEL_TYPE=${SYSNAV_DETECTOR_MODEL_TYPE}"
    -e "SYSNAV_DETECTOR_MODEL_PATH=${SYSNAV_DETECTOR_MODEL_PATH}"
    -e "SYSNAV_SAM2_CHECKPOINT=${SYSNAV_SAM2_CHECKPOINT}"
    -e "SYSNAV_MOBILECLIP_BLT_PATH=${SYSNAV_MOBILECLIP_BLT_PATH}"
    -e "SYSNAV_MOBILECLIP_BLT_TS_PATH=${SYSNAV_MOBILECLIP_BLT_TS_PATH}"
    -e "SYSNAV_CLIP_VIT_B32_PATH=${SYSNAV_CLIP_VIT_B32_PATH}"
    -e "SYSNAV_MAPPING_EXECUTOR_THREADS=${SYSNAV_MAPPING_EXECUTOR_THREADS}"
    -e "FRAMEWORK_SCRIPT=${FRAMEWORK_SCRIPT}"
    -e "BLOCK_LOWER_CONTROLLER=${BLOCK_LOWER_CONTROLLER}"
    -e "ENABLE_LOWER_CONTROLLER=${ENABLE_LOWER_CONTROLLER}"
    -e "LOWER_CONTROLLER_CMD=${LOWER_CONTROLLER_CMD}"
  )

  local runtime_json
  runtime_json="$(docker_cmd info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  if grep -q '"nvidia"' <<< "${runtime_json}"; then
    DOCKER_RUN_ARGS+=(
      --runtime=nvidia
      -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
      -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}"
    )
    append_jetson_nvidia_library_args DOCKER_RUN_ARGS
  elif command -v nvidia-smi >/dev/null 2>&1; then
    DOCKER_RUN_ARGS+=(--gpus all)
  fi

  for device in /dev/video0 /dev/video1; do
    [[ -e "${device}" ]] && DOCKER_RUN_ARGS+=(--device "${device}:${device}")
  done
  if [[ -n "${REAL_ROBOT_EXTRA_DEVICES:-}" ]]; then
    IFS=',' read -r -a extra_devices <<< "${REAL_ROBOT_EXTRA_DEVICES}"
    for device in "${extra_devices[@]}"; do
      [[ -n "${device}" && -e "${device}" ]] && DOCKER_RUN_ARGS+=(--device "${device}:${device}")
    done
  fi

  mkdir -p "${REPO_ROOT}/output" "${REPO_ROOT}/logs/ros" "${REPO_ROOT}/logs/runtime"
  DOCKER_RUN_ARGS+=(
    -v "${REPO_ROOT}/output:/workspace/STRIVE/output:rw"
    -v "${REPO_ROOT}/logs/runtime:/workspace/STRIVE/logs:rw"
    -v "${REPO_ROOT}/logs/ros:/root/.ros/log:rw"
    -v "${REPO_ROOT}:/workspace/STRIVE_HOST:ro"
  )

  declare -A mounted_dirs=()
  append_unique_dir_mount DOCKER_RUN_ARGS mounted_dirs "${SYSNAV_DETECTOR_MODEL_PATH}"
  append_unique_dir_mount DOCKER_RUN_ARGS mounted_dirs "${SYSNAV_SAM2_CHECKPOINT}"
  DOCKER_RUN_ARGS+=(
    -v "${SYSNAV_MOBILECLIP_BLT_PATH}:/workspace/STRIVE/mobileclip_blt.pt:ro"
    -v "${SYSNAV_MOBILECLIP_BLT_TS_PATH}:/workspace/STRIVE/mobileclip_blt.ts:ro"
    -v "${SYSNAV_CLIP_VIT_B32_PATH}:/root/.cache/clip/ViT-B-32.pt:ro"
  )

  for name in \
    MAP_PROVIDER AMAP_KEY \
    LLM_PROVIDER LLM_MODEL LLM_API_BASE_URL ARK_API_KEY GEMINI_API_KEY \
    STRIVE_LLM_CLIENT COGNAV_OBJNAV_PATH; do
    if [[ -n "${!name:-}" ]]; then
      DOCKER_RUN_ARGS+=(-e "${name}=${!name}")
    fi
  done
}

launch_args() {
  LAUNCH_ARGS=(
    "platform:=${PLATFORM}"
    "cloud_topic:=${CLOUD_TOPIC}"
    "odom_topic:=${ODOM_TOPIC}"
    "start_usb_cam:=${START_USB_CAM}"
    "usb_video_device:=${USB_VIDEO_DEVICE}"
    "usb_image_width:=${USB_IMAGE_WIDTH}"
    "usb_image_height:=${USB_IMAGE_HEIGHT}"
    "usb_pixel_format:=${USB_PIXEL_FORMAT}"
    "camera_topic:=${CAMERA_TOPIC}"
  )
}

start_lio() {
  bash "${REPO_ROOT}/scripts/start_orin_lio_for_strive.sh"
}

stop_lio() {
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t livox_odom 2>/dev/null; then
    tmux kill-session -t livox_odom
  fi
}

start_container() {
  ensure_image
  ensure_assets

  if [[ "${START_LIO}" == "1" ]]; then
    section "Start LIO"
    start_lio
    sleep "${LIO_STARTUP_SLEEP:-6}"
  fi

  local existing
  existing="$(docker_cmd ps -aq -f "name=^/${CONTAINER_NAME}$" || true)"
  if [[ -n "${existing}" ]]; then
    docker_cmd rm -f "${CONTAINER_NAME}" >/dev/null
  fi

  docker_args
  launch_args
  section "Start Container"
  docker_cmd run -d "${DOCKER_RUN_ARGS[@]}" "${IMAGE_TAG}" \
    bash -lc 'exec "${FRAMEWORK_SCRIPT:-/workspace/STRIVE/scripts/start_real_robot_framework.sh}" "$@"' \
    bash "${LAUNCH_ARGS[@]}"
  docker_cmd ps --filter "name=^/${CONTAINER_NAME}$"
}

enter_container() {
  if ! docker_cmd ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "Container is not running: ${CONTAINER_NAME}" >&2
    echo "Run: ./docker_en.sh start" >&2
    exit 4
  fi
  docker_cmd exec -it "${CONTAINER_NAME}" bash
}

exec_container() {
  if ! docker_cmd ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "Container is not running: ${CONTAINER_NAME}" >&2
    exit 4
  fi
  docker_cmd exec "${CONTAINER_NAME}" bash -lc "$*"
}

stop_container() {
  docker_cmd rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

status() {
  section "Image"
  docker_cmd images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E "^(${IMAGE_TAG}|REPOSITORY)" || true
  section "Container"
  docker_cmd ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}} {{.Image}} {{.Status}}' || true
  section "LIO"
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t livox_odom 2>/dev/null; then
    tmux list-panes -t livox_odom -F 'pane=#{pane_index} cmd=#{pane_current_command} pid=#{pane_pid} active=#{pane_active}'
  else
    echo "livox_odom tmux session is not running"
  fi
}

smoke() {
  IMAGE_TAG="${IMAGE_TAG}" \
  SYSNAV_DETECTOR_MODEL_TYPE="${SYSNAV_DETECTOR_MODEL_TYPE}" \
  SYSNAV_DETECTOR_MODEL_PATH="${SYSNAV_DETECTOR_MODEL_PATH}" \
  SYSNAV_SAM2_CHECKPOINT="${SYSNAV_SAM2_CHECKPOINT}" \
  SYSNAV_MOBILECLIP_BLT_PATH="${SYSNAV_MOBILECLIP_BLT_PATH}" \
  SYSNAV_MOBILECLIP_BLT_TS_PATH="${SYSNAV_MOBILECLIP_BLT_TS_PATH}" \
  SYSNAV_CLIP_VIT_B32_PATH="${SYSNAV_CLIP_VIT_B32_PATH}" \
  REQUIRE_ASSETS="${REQUIRE_ASSETS:-1}" \
  REQUIRE_LIO="${REQUIRE_LIO:-1}" \
  REQUIRE_ML="${REQUIRE_ML:-1}" \
  CHECK_CAMERA="${CHECK_CAMERA:-1}" \
  CHECK_DETECTOR_INIT="${CHECK_DETECTOR_INIT:-0}" \
  SUDO_STDIN_PASSWORD="${SUDO_STDIN_PASSWORD:-}" \
  bash "${REPO_ROOT}/scripts/smoke_real_robot_orin.sh"
}

cmd="${1:-}"
shift || true
case "${cmd}" in
  start) start_container "$@" ;;
  enter) enter_container ;;
  exec) exec_container "$@" ;;
  stop) stop_container ;;
  restart) stop_container; start_container "$@" ;;
  logs) docker_cmd logs -f --tail "${LOG_TAIL:-200}" "${CONTAINER_NAME}" ;;
  status) status ;;
  smoke) smoke ;;
  start-lio) start_lio ;;
  stop-lio) stop_lio ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: ${cmd}" >&2; usage; exit 1 ;;
esac
