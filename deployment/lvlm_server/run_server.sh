#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/opt/vln/ms-swift}"
MODEL_PATH="${VLN_LVLM_MODEL_PATH:-${STRIVE_LVLM_MODEL_PATH:-/models/Qwen2.5-VL-7B-Instruct}}"
SERVED_MODEL="${VLN_LVLM_SERVED_MODEL:-${STRIVE_LVLM_SERVED_MODEL:-vln-qwen2.5-vl-7b}}"
SERVER_HOST="${VLN_LVLM_HOST:-${STRIVE_LVLM_HOST:-0.0.0.0}}"
SERVER_PORT="${VLN_LVLM_PORT:-${STRIVE_LVLM_PORT:-8000}}"
API_KEY="${VLN_LVLM_API_KEY:-${STRIVE_LVLM_API_KEY:-}}"

export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export PYTHONPATH="${MS_SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "${MS_SWIFT_ROOT}/swift" ]]; then
  echo "[lvlm-server] ms-swift source is unavailable: ${MS_SWIFT_ROOT}" >&2
  exit 2
fi

python3 "${SCRIPT_DIR}/preflight.py" --model "${MODEL_PATH}"

args=(
  --model "${MODEL_PATH}"
  --infer_backend vllm
  --served_model_name "${SERVED_MODEL}"
  --host "${SERVER_HOST}"
  --port "${SERVER_PORT}"
  --vllm_gpu_memory_utilization "${VLN_LVLM_GPU_MEMORY_UTILIZATION:-${STRIVE_LVLM_GPU_MEMORY_UTILIZATION:-0.85}}"
  --vllm_max_model_len "${VLN_LVLM_MAX_MODEL_LEN:-${STRIVE_LVLM_MAX_MODEL_LEN:-8192}}"
  --vllm_limit_mm_per_prompt "${VLN_LVLM_LIMIT_MM_PER_PROMPT:-${STRIVE_LVLM_LIMIT_MM_PER_PROMPT:-{\"image\":8,\"video\":0}}}"
  --max_new_tokens "${VLN_LVLM_MAX_NEW_TOKENS:-${STRIVE_LVLM_MAX_NEW_TOKENS:-1024}}"
)

if [[ -n "${API_KEY}" ]]; then
  args+=(--api_key "${API_KEY}")
fi

echo "[lvlm-server] model : ${MODEL_PATH}"
echo "[lvlm-server] API   : http://${SERVER_HOST}:${SERVER_PORT}/v1"
echo "[lvlm-server] name  : ${SERVED_MODEL}"

exec python3 -m swift.cli.deploy "${args[@]}" "$@"
