#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
ENV_FILE="${SYSNAV_ENV_FILE:-${REPO_ROOT}/.env.realworld}"

IMAGE_TAG="${IMAGE_TAG:-huawei-vln-realworld:orin}"
CONTAINER_NAME="${CONTAINER_NAME:-huawei-vln-realworld}"
ASSET_DIR="${SYSNAV_ASSET_DIR:-${REPO_ROOT}/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external}"

PLATFORM="${PLATFORM:-mecanum}"
CLOUD_TOPIC="${CLOUD_TOPIC:-/cloud_registered}"
ODOM_TOPIC="${ODOM_TOPIC:-/aft_mapped_to_init}"
CAMERA_TOPIC="${CAMERA_TOPIC:-/camera/image}"
START_USB_CAM="${START_USB_CAM:-false}"
USB_VIDEO_DEVICE="${USB_VIDEO_DEVICE:-/dev/video0}"
USB_IMAGE_WIDTH="${USB_IMAGE_WIDTH:-1280}"
USB_IMAGE_HEIGHT="${USB_IMAGE_HEIGHT:-720}"
USB_PIXEL_FORMAT="${USB_PIXEL_FORMAT:-yuyv}"
USB_FRAMERATE="${USB_FRAMERATE:-30.0}"
USB_CAMERA_INFO_URL="${USB_CAMERA_INFO_URL:-}"
START_LIO="${START_LIO:-0}"
FRAMEWORK_SCRIPT="${FRAMEWORK_SCRIPT:-/workspace/STRIVE/scripts/start_real_robot_framework.sh}"
WAIT_FOR_LIO="${WAIT_FOR_LIO:-1}"
REQUIRE_LIO_SAMPLE="${REQUIRE_LIO_SAMPLE:-0}"
LIO_SAMPLE_TIMEOUT_S="${LIO_SAMPLE_TIMEOUT_S:-5}"
BLOCK_LOWER_CONTROLLER="${BLOCK_LOWER_CONTROLLER:-1}"
ENABLE_LOWER_CONTROLLER="${ENABLE_LOWER_CONTROLLER:-0}"
LOWER_CONTROLLER_CMD="${LOWER_CONTROLLER_CMD:-}"
CONTROL_CONTRACT_FILE="${CONTROL_CONTRACT_FILE:-}"
RESTART_POLICY="${RESTART_POLICY:-no}"
START_SEMANTIC_MAPPING="${START_SEMANTIC_MAPPING:-false}"
MAPPING_CONFIG="${MAPPING_CONFIG:-}"
PROJECTION_CONFIG="${PROJECTION_CONFIG:-}"
DETECTION_TOPIC="${DETECTION_TOPIC:-/huawei_vln/detection_result}"
OBJECT_NODES_TOPIC="${OBJECT_NODES_TOPIC:-/huawei_vln/object_nodes_list}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-}"

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
START_STRIVE_RUNTIME="${START_STRIVE_RUNTIME:-0}"
STRIVE_POLICY_MODE="${STRIVE_POLICY_MODE:-wait}"
STRIVE_INSTRUCTION_PLAN_BACKEND="${STRIVE_INSTRUCTION_PLAN_BACKEND:-rules}"
STRIVE_DRY_RUN="${STRIVE_DRY_RUN:-true}"
STRIVE_RUN_DIRECTORY="${STRIVE_RUN_DIRECTORY:-/tmp/strive_real_robot_runtime}"
STRIVE_LOWER_CONTROLLER_ENABLED="${STRIVE_LOWER_CONTROLLER_ENABLED:-false}"

usage() {
  cat <<EOF
Usage: ./docker_en.sh <command>

Commands:
  start       Start the isolated real-robot container. Host LIO remains untouched unless explicitly authorized.
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
  Runtime node   : START_STRIVE_RUNTIME=${START_STRIVE_RUNTIME}, STRIVE_DRY_RUN=${STRIVE_DRY_RUN}
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
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

require_host_lio_authority() {
  if ! is_true "${MANAGE_HOST_LIO:-false}"; then
    echo "Host LIO management is disabled. Set MANAGE_HOST_LIO=true only after confirming ownership of the livox_odom session." >&2
    exit 5
  fi
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
  local cuda_container_mount="${CUDA_CONTAINER_MOUNT:-/opt/strive/host-cuda}"
  local ld_paths=()

  if [[ -z "${cuda_home}" && -e /usr/local/cuda ]]; then
    cuda_home="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
  fi
  if [[ -n "${cuda_home}" && -d "${cuda_home}" ]]; then
    # nvidia-container-runtime may mount an empty path over /usr/local/cuda.
    # A private container path avoids that collision while keeping host CUDA RO.
    args_ref+=(-v "${cuda_home}:${cuda_container_mount}:ro")
    ld_paths+=("${cuda_container_mount}/lib64" "${cuda_container_mount}/targets/aarch64-linux/lib")
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
  case "${RESTART_POLICY}" in
    no|always|unless-stopped|on-failure|on-failure:*) ;;
    *) echo "Unsupported RESTART_POLICY: ${RESTART_POLICY}" >&2; exit 2 ;;
  esac
  DOCKER_RUN_ARGS=(
    --name "${CONTAINER_NAME}"
    --network host
    --ipc=host
    --restart "${RESTART_POLICY}"
    -e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}"
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
    -e "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
    -e "PYTHONUNBUFFERED=1"
    # Pass every physical binding into the framework environment as well as
    # launch arguments.  Preflight must validate the same values that launch
    # will ultimately use; otherwise a profile override can be masked by a
    # script default before the ROS graph starts.
    -e "PLATFORM=${PLATFORM}"
    -e "CLOUD_TOPIC=${CLOUD_TOPIC}"
    -e "ODOM_TOPIC=${ODOM_TOPIC}"
    -e "CAMERA_TOPIC=${CAMERA_TOPIC}"
    -e "VIEWPOINT_TOPIC=${VIEWPOINT_TOPIC:-/viewpoint_rep_header}"
    -e "START_USB_CAM=${START_USB_CAM}"
    -e "USB_VIDEO_DEVICE=${USB_VIDEO_DEVICE}"
    -e "USB_IMAGE_WIDTH=${USB_IMAGE_WIDTH}"
    -e "USB_IMAGE_HEIGHT=${USB_IMAGE_HEIGHT}"
    -e "USB_PIXEL_FORMAT=${USB_PIXEL_FORMAT}"
    -e "USB_FRAMERATE=${USB_FRAMERATE}"
    -e "USB_CAMERA_INFO_URL=${USB_CAMERA_INFO_URL}"
    -e "SYSNAV_DETECTOR_MODEL_TYPE=${SYSNAV_DETECTOR_MODEL_TYPE}"
    -e "SYSNAV_DETECTOR_MODEL_PATH=${SYSNAV_DETECTOR_MODEL_PATH}"
    -e "SYSNAV_SAM2_CHECKPOINT=${SYSNAV_SAM2_CHECKPOINT}"
    -e "SYSNAV_MOBILECLIP_BLT_PATH=${SYSNAV_MOBILECLIP_BLT_PATH}"
    -e "SYSNAV_MOBILECLIP_BLT_TS_PATH=${SYSNAV_MOBILECLIP_BLT_TS_PATH}"
    -e "SYSNAV_CLIP_VIT_B32_PATH=${SYSNAV_CLIP_VIT_B32_PATH}"
    -e "SYSNAV_MAPPING_EXECUTOR_THREADS=${SYSNAV_MAPPING_EXECUTOR_THREADS}"
    -e "FRAMEWORK_SCRIPT=${FRAMEWORK_SCRIPT}"
    -e "WAIT_FOR_LIO=${WAIT_FOR_LIO}"
    -e "REQUIRE_LIO_SAMPLE=${REQUIRE_LIO_SAMPLE}"
    -e "LIO_SAMPLE_TIMEOUT_S=${LIO_SAMPLE_TIMEOUT_S}"
    -e "BLOCK_LOWER_CONTROLLER=${BLOCK_LOWER_CONTROLLER}"
    -e "ENABLE_LOWER_CONTROLLER=${ENABLE_LOWER_CONTROLLER}"
    -e "LOWER_CONTROLLER_CMD=${LOWER_CONTROLLER_CMD}"
    -e "CONTROL_CONTRACT_FILE=${CONTROL_CONTRACT_FILE}"
    -e "START_STRIVE_RUNTIME=${START_STRIVE_RUNTIME}"
    -e "START_LOWER_STACK=${START_LOWER_STACK:-0}"
    -e "STRIVE_POLICY_MODE=${STRIVE_POLICY_MODE}"
    -e "STRIVE_INSTRUCTION_PLAN_BACKEND=${STRIVE_INSTRUCTION_PLAN_BACKEND}"
    -e "STRIVE_DRY_RUN=${STRIVE_DRY_RUN}"
    -e "STRIVE_RUN_DIRECTORY=${STRIVE_RUN_DIRECTORY}"
    -e "STRIVE_LOWER_CONTROLLER_ENABLED=${STRIVE_LOWER_CONTROLLER_ENABLED}"
    -e "START_SEMANTIC_MAPPING=${START_SEMANTIC_MAPPING}"
    -e "MAPPING_CONFIG=${MAPPING_CONFIG}"
    -e "PROJECTION_CONFIG=${PROJECTION_CONFIG}"
    -e "DETECTION_TOPIC=${DETECTION_TOPIC}"
    -e "OBJECT_NODES_TOPIC=${OBJECT_NODES_TOPIC}"
  )
  if [[ -n "${RMW_IMPLEMENTATION}" ]]; then
    DOCKER_RUN_ARGS+=(-e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}")
  fi

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

  if is_true "${START_USB_CAM}"; then
    for device in /dev/video0 /dev/video1; do
      [[ -e "${device}" ]] && DOCKER_RUN_ARGS+=(--device "${device}:${device}")
    done
    if [[ -n "${REAL_ROBOT_EXTRA_DEVICES:-}" ]]; then
      IFS=',' read -r -a extra_devices <<< "${REAL_ROBOT_EXTRA_DEVICES}"
      for device in "${extra_devices[@]}"; do
        [[ -n "${device}" && -e "${device}" ]] && DOCKER_RUN_ARGS+=(--device "${device}:${device}")
      done
    fi
  fi

  mkdir -p "${REPO_ROOT}/output" "${REPO_ROOT}/logs/ros" "${REPO_ROOT}/logs/runtime"
  DOCKER_RUN_ARGS+=(
    -v "${REPO_ROOT}/output:/workspace/STRIVE/output:rw"
    -v "${REPO_ROOT}/logs/runtime:/workspace/STRIVE/logs:rw"
    -v "${REPO_ROOT}/logs/ros:/root/.ros/log:rw"
  )

  # Camera intrinsics remain specific to this workspace and are mounted only
  # read-only.  A calibrated file can therefore be swapped without rebuilding
  # the application image or exposing another robot's calibration directory.
  local calibration_dir="${CAMERA_CALIBRATION_HOST_DIR:-${REPO_ROOT}/real_robot/calibration}"
  if [[ -d "${calibration_dir}" ]]; then
    DOCKER_RUN_ARGS+=(-v "${calibration_dir}:/workspace/STRIVE/real_robot/calibration:ro")
  fi
  local control_contract_dir="${CONTROL_CONTRACT_HOST_DIR:-${REPO_ROOT}/real_robot/control}"
  if [[ -d "${control_contract_dir}" ]]; then
    DOCKER_RUN_ARGS+=(-v "${control_contract_dir}:/workspace/STRIVE/real_robot/control:ro")
  fi

  declare -A mounted_dirs=()
  append_unique_dir_mount DOCKER_RUN_ARGS mounted_dirs "${SYSNAV_DETECTOR_MODEL_PATH}"
  append_unique_dir_mount DOCKER_RUN_ARGS mounted_dirs "${SYSNAV_SAM2_CHECKPOINT}"
  append_unique_dir_mount DOCKER_RUN_ARGS mounted_dirs "${STRIVE_PRIOR_MAP_PATH:-}"
  DOCKER_RUN_ARGS+=(
    -v "${SYSNAV_MOBILECLIP_BLT_PATH}:/workspace/STRIVE/mobileclip_blt.pt:ro"
    -v "${SYSNAV_MOBILECLIP_BLT_TS_PATH}:/workspace/STRIVE/mobileclip_blt.ts:ro"
    -v "${SYSNAV_CLIP_VIT_B32_PATH}:/root/.cache/clip/ViT-B-32.pt:ro"
  )

  for name in \
    MAP_PROVIDER AMAP_KEY \
    LLM_PROVIDER LLM_MODEL LLM_API_BASE_URL ARK_API_KEY GEMINI_API_KEY \
    STRIVE_LLM_CLIENT COGNAV_OBJNAV_PATH \
    STRIVE_INSTRUCTION STRIVE_DATASET_TARGET STRIVE_VLM \
    STRIVE_ENABLE_FINAL_VERIFIER STRIVE_EVIDENCE_MODE STRIVE_PRIOR_MAP_PATH \
    STRIVE_DRY_RUN_STATUS STRIVE_WAYPOINT_TOPIC STRIVE_TEST_WAYPOINT_TOPIC \
    STRIVE_HOLD_TOPIC STRIVE_CANCEL_TOPIC STRIVE_EMERGENCY_STOP_TOPIC \
    STRIVE_ALLOW_EMERGENCY_STOP_PUBLISH \
    STRIVE_OBJECT_TOPIC STRIVE_ROOM_TOPIC STRIVE_ODOM_TOPIC STRIVE_PATH_TOPIC \
    STRIVE_PLANNER_STATUS_TOPIC STRIVE_IMAGE_TOPIC STRIVE_DETECTION_TOPIC \
    STRIVE_DEPTH_TOPIC STRIVE_POINTCLOUD_TOPIC \
    STRIVE_PERSIST_OBSERVATION_IMAGES STRIVE_OBSERVATION_IMAGE_DIRECTORY \
    STRIVE_DECISION_PERIOD_S STRIVE_USE_SIM_TIME; do
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
    "usb_framerate:=${USB_FRAMERATE}"
    "usb_camera_info_url:=${USB_CAMERA_INFO_URL}"
    "camera_topic:=${CAMERA_TOPIC}"
  )
}

start_lio() {
  require_host_lio_authority
  bash "${REPO_ROOT}/scripts/start_orin_lio_for_strive.sh"
}

stop_lio() {
  require_host_lio_authority
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
  CHECK_CAMERA="${CHECK_CAMERA:-0}" \
  USB_IMAGE_WIDTH="${USB_IMAGE_WIDTH}" \
  USB_IMAGE_HEIGHT="${USB_IMAGE_HEIGHT}" \
  USB_PIXEL_FORMAT="${USB_PIXEL_FORMAT}" \
  USB_FRAMERATE="${USB_FRAMERATE}" \
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
