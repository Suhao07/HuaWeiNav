#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-strive-hm3d:local}"
COGNAV_BASE_IMAGE="${COGNAV_BASE_IMAGE:-cognav-vln:1.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[build] STRIVE image : $IMAGE_TAG"
echo "[build] base image   : $COGNAV_BASE_IMAGE"
docker build \
  --build-arg "COGNAV_BASE_IMAGE=$COGNAV_BASE_IMAGE" \
  -f "$SCRIPT_DIR/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"

echo "[build] OK"
docker images "$IMAGE_TAG"
