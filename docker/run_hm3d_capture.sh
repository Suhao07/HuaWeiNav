#!/usr/bin/env bash
set -euo pipefail
STRIVE_ROOT="${STRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${STRIVE_DATA_ROOT:?Set STRIVE_DATA_ROOT to the existing Habitat dataset root}"
IMAGE_TAG="${IMAGE_TAG:-strive-hm3d:local}"
docker run --rm --gpus all --user "$(id -u):$(id -g)" --shm-size=4g \
  -e PYTHONNOUSERSITE=1 -e MAGNUM_LOG=quiet -e HABITAT_SIM_LOG=quiet \
  -v "$STRIVE_ROOT:/workspace/STRIVE" -v "$STRIVE_DATA_ROOT:/workspace/data:ro" \
  -w /workspace/STRIVE "$IMAGE_TAG" bash -lc 'cd /workspace/STRIVE && PYTHONNOUSERSITE=1 python scripts/collect_hm3d_observations.py "$@"' bash "$@"
