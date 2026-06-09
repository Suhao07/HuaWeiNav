#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash docker/run_scene_object_nav.sh --scene_id SCENE --object_category OBJECT [options]

Example:
  bash docker/run_scene_object_nav.sh \
    --scene_id wcojb4TFT35 \
    --object_category tv \
    --save_dir hm3d_wcojb4TFT35_tv_real_llm \
    --vlm cognav

Options:
  --episode_rank N      Use the N-th matched episode in this scene/object pair. Default: 0.
  --save_dir NAME      Output directory under logs/. Default: hm3d_${scene}_${object}.
  --vlm NAME           VLM/LLM backend passed to benchmark. Default: cognav.
  --max_steps N        Optional cap for quick debugging. Default benchmark cap is 500.
  -h, --help           Show this help.
  ...                  Any remaining args are forwarded to objnav_benchmark_with_process_obs.py.
EOF
}

SCENE_ID=""
OBJECT_CATEGORY=""
EPISODE_RANK="0"
SAVE_DIR=""
VLM="cognav"
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --scene_id)
      SCENE_ID="${2:?--scene_id requires a value}"
      shift 2
      ;;
    --object_category)
      OBJECT_CATEGORY="${2:?--object_category requires a value}"
      shift 2
      ;;
    --episode_rank)
      EPISODE_RANK="${2:?--episode_rank requires a value}"
      shift 2
      ;;
    --save_dir)
      SAVE_DIR="${2:?--save_dir requires a value}"
      shift 2
      ;;
    --vlm)
      VLM="${2:?--vlm requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$SCENE_ID" ] || [ -z "$OBJECT_CATEGORY" ]; then
  usage >&2
  exit 2
fi

safe_name() {
  printf '%s' "$1" | tr -c '[:alnum:]_-' '_'
}

if [ -z "$SAVE_DIR" ]; then
  SAVE_DIR="hm3d_$(safe_name "$SCENE_ID")_$(safe_name "$OBJECT_CATEGORY")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_hm3d_baseline.sh" \
  --scene_id "$SCENE_ID" \
  --object_category "$OBJECT_CATEGORY" \
  --episode_rank "$EPISODE_RANK" \
  --eval_episodes 1 \
  --start_episode 0 \
  --dataset_episodes 1 \
  --save_dir "$SAVE_DIR" \
  --vlm "$VLM" \
  "${EXTRA_ARGS[@]}"
