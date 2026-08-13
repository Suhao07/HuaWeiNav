#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-strive-hm3d:local}"
BASE_IMAGE="${BASE_IMAGE:-habitat-hm3d:local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[build] STRIVE image : $IMAGE_TAG"
echo "[build] base image   : $BASE_IMAGE"

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "[build] base image not found: $BASE_IMAGE" >&2
  echo "[build] HM3D simulation needs a Habitat/PyTorch CUDA base image." >&2
  echo "[build] Real-robot ROS2 Humble deployment uses docker/build_real_robot.sh instead." >&2
  exit 2
fi

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$SCRIPT_DIR/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"

echo "[build] OK"
docker images "$IMAGE_TAG"
