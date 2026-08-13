#!/usr/bin/env bash
set -euo pipefail

# This entrypoint only builds static HM3D room/layout priors. It intentionally
# does not require detector checkpoints or ObjectNav episode annotations.
IMAGE_TAG="${IMAGE_TAG:-strive-hm3d:local}"
CONTAINER_NAME="${CONTAINER_NAME:-strive-hm3d-floorplan-layout}"
STRIVE_ROOT="${STRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${STRIVE_DATA_ROOT:-}" ]]; then
  DATA_ROOT="$STRIVE_DATA_ROOT"
elif [[ -n "${HM3D_DATA_ROOT:-}" ]]; then
  DATA_ROOT="$HM3D_DATA_ROOT"
elif [[ -n "${COGNAV_ROOT:-}" ]]; then
  DATA_ROOT="$COGNAV_ROOT/data"
else
  DATA_ROOT="$STRIVE_ROOT/data"
fi

if [[ $# -lt 1 || "${1:-}" == -* ]]; then
  cat >&2 <<'EOF'
Usage:
  bash docker/run_hm3d_floorplan_layout.sh \
    /workspace/data/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35 \
    --scene_id wcojb4TFT35 \
    --output logs/prior_maps/wcojb4TFT35_floorplan.json \
    --quality_output logs/prior_maps/wcojb4TFT35_floorplan_quality.json

The scene path is inside the container. The host data directory is selected by
STRIVE_DATA_ROOT, HM3D_DATA_ROOT, COGNAV_ROOT/data, or STRIVE_ROOT/data.
EOF
  exit 2
fi

SCENE_DIR="$1"
shift

if [[ "$SCENE_DIR" == /workspace/data/* ]]; then
  HOST_SCENE_DIR="$DATA_ROOT/${SCENE_DIR#/workspace/data/}"
else
  echo "[hm3d-layout] scene_dir must be under /workspace/data inside the container: $SCENE_DIR" >&2
  exit 2
fi

for required in "*.basis.glb" "*.semantic.glb" "*.semantic.txt" "*.basis.navmesh"; do
  if ! compgen -G "$HOST_SCENE_DIR/$required" >/dev/null; then
    echo "[hm3d-layout] missing $required under $HOST_SCENE_DIR" >&2
    echo "[hm3d-layout] check STRIVE_DATA_ROOT/COGNAV_ROOT and the HM3D split" >&2
    exit 2
  fi
done

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size=8g \
  --ipc host \
  -e "HM3D_DATA_PATH=/workspace/data" \
  -v "$STRIVE_ROOT:/workspace/STRIVE" \
  -v "$DATA_ROOT:/workspace/data:ro" \
  "$IMAGE_TAG" \
  bash -lc '
    set -euo pipefail
    cd /workspace/STRIVE
    PYTHONNOUSERSITE=1 python scripts/build_hm3d_floorplan_layout.py "$@"
  ' bash "$SCENE_DIR" "$@"
