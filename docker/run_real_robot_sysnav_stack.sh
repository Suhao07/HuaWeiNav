#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-strive-real-robot:humble}"
CONTAINER_NAME="${CONTAINER_NAME:-strive-real-robot-sysnav}"
GPU_ARGS=()
DEVICE_ARGS=()
TTY_ARGS=()

if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

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
  STRIVE_LLM_CLIENT COGNAV_OBJNAV_PATH SYSNAV_MAPPING_EXECUTOR_THREADS; do
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
if [[ -n "${SYSNAV_MOBILECLIP_BLT_PATH:-}" && -f "${SYSNAV_MOBILECLIP_BLT_PATH}" ]]; then
  VOLUME_ARGS+=(-v "${SYSNAV_MOBILECLIP_BLT_PATH}:/workspace/STRIVE/mobileclip_blt.pt:ro")
fi
if [[ -n "${SYSNAV_MOBILECLIP_BLT_TS_PATH:-}" && -f "${SYSNAV_MOBILECLIP_BLT_TS_PATH}" ]]; then
  VOLUME_ARGS+=(-v "${SYSNAV_MOBILECLIP_BLT_TS_PATH}:/workspace/STRIVE/mobileclip_blt.ts:ro")
fi
if [[ -n "${SYSNAV_CLIP_VIT_B32_PATH:-}" && -f "${SYSNAV_CLIP_VIT_B32_PATH}" ]]; then
  VOLUME_ARGS+=(-v "${SYSNAV_CLIP_VIT_B32_PATH}:/root/.cache/clip/ViT-B-32.pt:ro")
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
