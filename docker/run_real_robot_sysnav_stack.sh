#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-strive-real-robot:humble}"
CONTAINER_NAME="${CONTAINER_NAME:-strive-real-robot-sysnav}"
GPU_ARGS=()

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

MODEL_ENVS=()
for name in SYSNAV_DETECTOR_MODEL_TYPE SYSNAV_DETECTOR_MODEL_PATH SYSNAV_SAM2_CHECKPOINT; do
  if [[ -n "${!name:-}" ]]; then
    MODEL_ENVS+=(-e "${name}=${!name}")
  fi
done

RUNTIME_ENVS=()
for name in \
  MAP_PROVIDER AMAP_KEY \
  LLM_PROVIDER LLM_MODEL LLM_API_BASE_URL ARK_API_KEY GEMINI_API_KEY \
  STRIVE_LLM_CLIENT COGNAV_OBJNAV_PATH; do
  if [[ -n "${!name:-}" ]]; then
    RUNTIME_ENVS+=(-e "${name}=${!name}")
  fi
done

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

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --network host \
  "${GPU_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  "${MODEL_ENVS[@]}" \
  "${RUNTIME_ENVS[@]}" \
  "${IMAGE_TAG}" \
  bash -lc 'source /opt/ros/humble/setup.bash && source /workspace/STRIVE/real_robot/ros2_ws/install/setup.bash && /workspace/STRIVE/scripts/run_sysnav_detection_mapping.sh'
