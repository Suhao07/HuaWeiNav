#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-huawei-nav-real:orin}"
CONTAINER_NAME="${CONTAINER_NAME:-huawei-nav-real-sysnav}"
GPU_ARGS=()
DEVICE_ARGS=()
TTY_ARGS=()

if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

append_jetson_nvidia_library_args() {
  local -n args_ref=$1
  local cuda_home="${CUDA_HOME_HOST:-}"
  local cuda_container_mount="${CUDA_CONTAINER_MOUNT:-/opt/strive/host-cuda}"
  local ld_paths=()

  if [[ -z "${cuda_home}" && -e /usr/local/cuda ]]; then
    cuda_home="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
  fi
  if [[ -n "${cuda_home}" && -d "${cuda_home}" ]]; then
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

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  GPU_ARGS=(
    --runtime=nvidia
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}"
  )
  append_jetson_nvidia_library_args GPU_ARGS
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

for device in /dev/video0 /dev/video1; do
  if [[ -e "${device}" ]]; then
    DEVICE_ARGS+=(--device "${device}:${device}")
  fi
done
if [[ -n "${REAL_ROBOT_EXTRA_DEVICES:-}" ]]; then
  IFS=',' read -r -a extra_devices <<< "${REAL_ROBOT_EXTRA_DEVICES}"
  for device in "${extra_devices[@]}"; do
    [[ -n "${device}" && -e "${device}" ]] || continue
    DEVICE_ARGS+=(--device "${device}:${device}")
  done
fi

if [[ -z "${SYSNAV_MOBILECLIP_BLT_TS_PATH:-}" && -n "${SYSNAV_MOBILECLIP_BLT_PATH:-}" ]]; then
  mobileclip_ts_candidate="${SYSNAV_MOBILECLIP_BLT_PATH%.*}.ts"
  if [[ -f "${mobileclip_ts_candidate}" ]]; then
    SYSNAV_MOBILECLIP_BLT_TS_PATH="${mobileclip_ts_candidate}"
  fi
fi

MODEL_ENVS=()
for name in SYSNAV_DETECTOR_MODEL_TYPE SYSNAV_DETECTOR_MODEL_PATH SYSNAV_SAM2_CHECKPOINT SYSNAV_MOBILECLIP_BLT_PATH SYSNAV_MOBILECLIP_BLT_TS_PATH SYSNAV_CLIP_VIT_B32_PATH; do
  if [[ -n "${!name:-}" ]]; then
    MODEL_ENVS+=(-e "${name}=${!name}")
  fi
done

RUNTIME_ENVS=()
for name in \
  MAP_PROVIDER AMAP_KEY \
  LLM_PROVIDER LLM_MODEL LLM_API_BASE_URL ARK_API_KEY GEMINI_API_KEY \
  STRIVE_LLM_CLIENT COGNAV_OBJNAV_PATH SYSNAV_MAPPING_EXECUTOR_THREADS \
  START_STRIVE_RUNTIME START_LOWER_STACK STRIVE_INSTRUCTION STRIVE_DATASET_TARGET \
  STRIVE_POLICY_MODE STRIVE_INSTRUCTION_PLAN_BACKEND STRIVE_VLM \
  STRIVE_MOTION_BACKEND STRIVE_MOTION_ACTION_NAME \
  CONTROL_CONTRACT_FILE \
  STRIVE_ENABLE_FINAL_VERIFIER STRIVE_EVIDENCE_MODE STRIVE_PRIOR_MAP_PATH \
  STRIVE_DRY_RUN STRIVE_DRY_RUN_STATUS \
  STRIVE_LOWER_CONTROLLER_ENABLED STRIVE_WAYPOINT_TOPIC STRIVE_TEST_WAYPOINT_TOPIC \
  STRIVE_HOLD_TOPIC STRIVE_CANCEL_TOPIC STRIVE_EMERGENCY_STOP_TOPIC \
  STRIVE_ALLOW_EMERGENCY_STOP_PUBLISH \
  STRIVE_OBJECT_TOPIC STRIVE_ROOM_TOPIC STRIVE_ODOM_TOPIC STRIVE_PATH_TOPIC \
  STRIVE_PLANNER_STATUS_TOPIC STRIVE_IMAGE_TOPIC STRIVE_DETECTION_TOPIC \
  STRIVE_DEPTH_TOPIC STRIVE_POINTCLOUD_TOPIC \
  STRIVE_PERSIST_OBSERVATION_IMAGES STRIVE_OBSERVATION_IMAGE_DIRECTORY \
  STRIVE_DECISION_PERIOD_S STRIVE_USE_SIM_TIME; do
  if [[ -n "${!name:-}" ]]; then
    RUNTIME_ENVS+=(-e "${name}=${!name}")
  fi
done

DDS_ENVS=(-e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}")

VOLUME_ARGS=()
declare -A _mounted_dirs=()
for path_var in SYSNAV_DETECTOR_MODEL_PATH SYSNAV_SAM2_CHECKPOINT; do
  model_path="${!path_var:-}"
  if [[ -n "${model_path}" && -f "${model_path}" ]]; then
    model_dir="$(cd "$(dirname "${model_path}")" && pwd)"
    if [[ -z "${_mounted_dirs[${model_dir}]:-}" ]]; then
      # 核心：权重保留在宿主机，容器内使用相同绝对路径读取，便于迁移到真机。
      VOLUME_ARGS+=(-v "${model_dir}:${model_dir}:ro")
      _mounted_dirs["${model_dir}"]=1
    fi
  fi
done
if [[ -n "${STRIVE_PRIOR_MAP_PATH:-}" && -f "${STRIVE_PRIOR_MAP_PATH}" ]]; then
  prior_map_dir="$(cd "$(dirname "${STRIVE_PRIOR_MAP_PATH}")" && pwd)"
  if [[ -z "${_mounted_dirs[${prior_map_dir}]:-}" ]]; then
    VOLUME_ARGS+=(-v "${prior_map_dir}:${prior_map_dir}:ro")
    _mounted_dirs["${prior_map_dir}"]=1
  fi
fi
if [[ -n "${SYSNAV_MOBILECLIP_BLT_PATH:-}" && -f "${SYSNAV_MOBILECLIP_BLT_PATH}" ]]; then
  VOLUME_ARGS+=(-v "${SYSNAV_MOBILECLIP_BLT_PATH}:/workspace/STRIVE/mobileclip_blt.pt:ro")
fi
if [[ -n "${SYSNAV_MOBILECLIP_BLT_TS_PATH:-}" && -f "${SYSNAV_MOBILECLIP_BLT_TS_PATH}" ]]; then
  VOLUME_ARGS+=(-v "${SYSNAV_MOBILECLIP_BLT_TS_PATH}:/workspace/STRIVE/mobileclip_blt.ts:ro")
fi
if [[ -n "${SYSNAV_CLIP_VIT_B32_PATH:-}" && -f "${SYSNAV_CLIP_VIT_B32_PATH}" ]]; then
  VOLUME_ARGS+=(-v "${SYSNAV_CLIP_VIT_B32_PATH}:/root/.cache/clip/ViT-B-32.pt:ro")
fi
if [[ -n "${CONTROL_CONTRACT_FILE:-}" ]]; then
  control_contract_dir="${CONTROL_CONTRACT_HOST_DIR:-${REPO_ROOT}/real_robot/control}"
  if [[ ! -d "${control_contract_dir}" ]]; then
    echo "CONTROL_CONTRACT_HOST_DIR does not exist: ${control_contract_dir}" >&2
    exit 2
  fi
  # 只读挂载 robot-specific contract；容器内路径与 profile 保持一致，
  # 防止审批文件被镜像中的模板或旧配置遮蔽。
  VOLUME_ARGS+=(-v "${control_contract_dir}:/workspace/STRIVE/real_robot/control:ro")
fi

docker run --rm "${TTY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc=host \
  "${GPU_ARGS[@]}" \
  "${DEVICE_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  "${MODEL_ENVS[@]}" \
  "${DDS_ENVS[@]}" \
  "${RUNTIME_ENVS[@]}" \
  "${IMAGE_TAG}" \
  bash -lc 'exec /workspace/STRIVE/scripts/run_sysnav_detection_mapping.sh "$@"' \
  bash "$@"
