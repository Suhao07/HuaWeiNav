#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-strive-hm3d:local}"
CONTAINER_NAME="${CONTAINER_NAME:-strive-hm3d-baseline}"
STRIVE_ROOT="${STRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COGNAV_ROOT="${COGNAV_ROOT:-/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav}"
MODELS_DIR="${STRIVE_MODELS_DIR:-$COGNAV_ROOT/model/pretrained_model}"
HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"

mkdir -p "$MODELS_DIR" "$STRIVE_ROOT/logs"

# 权重优先复用本机和 CogNav 仓库已有文件；只有显式设置
# STRIVE_DOWNLOAD_WEIGHTS=1 时才下载，避免每次 benchmark 都访问外网。
find_first() {
  for path in "$@"; do
    if [ -f "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

download_if_enabled() {
  local url="$1"
  local out="$2"
  if [ "${STRIVE_DOWNLOAD_WEIGHTS:-0}" != "1" ]; then
    return 1
  fi
  echo "[weights] downloading $(basename "$out")"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$out" "$url"
  else
    curl -L --progress-bar -o "$out" "$url"
  fi
}

SAM_HOST="${SAM_CHECKPOINT:-}"
if [ -z "$SAM_HOST" ]; then
  SAM_HOST="$(find_first \
    "$MODELS_DIR/sam_vit_h_4b8939.pth" \
    "$COGNAV_ROOT/model/pretrained_model/sam_vit_h_4b8939.pth" \
    "/home/ubuntu/WorkSpace/research/code/CoRL2025/SG-Nav/segment_anything/sam_vit_h_4b8939.pth" \
    "/home/ubuntu/WorkSpace/research/code/CoRL2025/AKGVP/data/models/sam_vit_h_4b8939.pth" \
    2>/dev/null || true)"
fi
if [ -z "$SAM_HOST" ]; then
  SAM_HOST="$MODELS_DIR/sam_vit_h_4b8939.pth"
  download_if_enabled "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" "$SAM_HOST" || {
    echo "[weights] missing SAM checkpoint. Set SAM_CHECKPOINT or run with STRIVE_DOWNLOAD_WEIGHTS=1." >&2
    exit 2
  }
fi

DINO_HOST="${GROUNDING_DINO_CHECKPOINT:-}"
if [ -z "$DINO_HOST" ]; then
  DINO_HOST="$(find_first \
    "$MODELS_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
    "$COGNAV_ROOT/model/pretrained_model/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
    "$STRIVE_ROOT/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
    2>/dev/null || true)"
fi
if [ -z "$DINO_HOST" ]; then
  DINO_HOST="$MODELS_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
  download_if_enabled \
    "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
    "$DINO_HOST" || {
    echo "[weights] missing MMDetection GroundingDINO Swin-L checkpoint." >&2
    echo "[weights] Set GROUNDING_DINO_CHECKPOINT or run with STRIVE_DOWNLOAD_WEIGHTS=1." >&2
    echo "[weights] The local groundingdino_swint_ogc.pth files are not used because STRIVE's mmdet config expects Swin-L." >&2
    exit 2
  }
fi

if [ ! -d "$COGNAV_ROOT/data/scene_datasets/hm3d_v0.2" ]; then
  echo "[data] HM3D scene data not found under $COGNAV_ROOT/data/scene_datasets/hm3d_v0.2" >&2
  exit 2
fi
if [ ! -f "$COGNAV_ROOT/data/objectgoal_hm3d/val/val.json.gz" ] \
   && [ ! -f "$COGNAV_ROOT/data/objectnav_hm3d_v2/val/val.json.gz" ]; then
  echo "[data] HM3D ObjectNav episodes not found under $COGNAV_ROOT/data" >&2
  exit 2
fi

ENV_FILE_FLAG=()
if [ -f "$STRIVE_ROOT/docker/.env" ]; then
  ENV_FILE_FLAG=(--env-file "$STRIVE_ROOT/docker/.env")
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

RUN_ARGS=(
  docker run --rm "${TTY_ARGS[@]}"
  --name "$CONTAINER_NAME"
  --gpus all
  --shm-size=12g
  --network host
  --ipc host
  "${ENV_FILE_FLAG[@]}"
  -e "STRIVE_LLM_CLIENT=cognav"
  -e "COGNAV_OBJNAV_PATH=/workspace/CogNav_ObjNav"
  -e "COGNAV_APIKEY_FILE=/workspace/CogNav_ObjNav/apikey.txt"
  -e "ARK_API_KEY=${ARK_API_KEY:-}"
  -e "OPENAI_API_KEY=${OPENAI_API_KEY:-}"
  -e "MAP_PROVIDER=${MAP_PROVIDER:-}"
  -e "AMAP_KEY=${AMAP_KEY:-}"
  -e "LLM_PROVIDER=${LLM_PROVIDER:-ark}"
  -e "LLM_MODEL=${LLM_MODEL:-doubao-seed-2-0-pro-260215}"
  -e "LLM_API_BASE_URL=${LLM_API_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
  -e "LLM_OFFLINE=${LLM_OFFLINE:-0}"
  -e "STRIVE_LLM_FALLBACK=${STRIVE_LLM_FALLBACK:-0}"
  -e "HF_HOME=/root/.cache/huggingface"
  -e "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}"
  -e "HABITAT_LAB_PATH=/workspace/CogNav_ObjNav"
  -e "HM3D_DATA_PATH=/workspace/CogNav_ObjNav/data"
  -e "MP3D_DATA_PATH=/workspace/CogNav_ObjNav/data"
  -e "SAM_CHECKPOINT=/weights/sam_vit_h_4b8939.pth"
  -e "GROUNDING_DINO_PATH=/opt/mmdetection"
  -e "GROUNDING_DINO_CONFIG=/opt/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py"
  -e "GROUNDING_DINO_CHECKPOINT=/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
  # STRIVE 可写挂载，CogNav 只读挂载；数据和 LLM client 都从 CogNav 复用。
  -v "$STRIVE_ROOT":/workspace/STRIVE
  -v "$COGNAV_ROOT":/workspace/CogNav_ObjNav:ro
  -v "$HF_HOME_HOST":/root/.cache/huggingface:ro
  -v "$SAM_HOST":/weights/sam_vit_h_4b8939.pth:ro
  -v "$DINO_HOST":/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth:ro
  "$IMAGE_TAG"
)

if [ "${1:-}" = "bash" ]; then
  shift
  exec "${RUN_ARGS[@]}" bash "$@"
fi

if [ "$#" -eq 0 ]; then
  set -- --eval_episodes 1 --start_episode 0 --save_dir hm3d_cognav_smoke --vlm cognav
fi

exec "${RUN_ARGS[@]}" bash -lc '
  set -e
  cd /workspace/STRIVE
  PYTHONNOUSERSITE=1 python objnav_benchmark_with_process_obs.py "$@"
' bash "$@"
