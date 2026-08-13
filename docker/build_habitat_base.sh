#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-habitat-hm3d:local}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-strive}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
HABITAT_LAB_REPO="${HABITAT_LAB_REPO:-https://github.com/facebookresearch/habitat-lab.git}"
HABITAT_LAB_REF="${HABITAT_LAB_REF:-v0.3.2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[build-base] image           : $IMAGE_TAG"
echo "[build-base] conda env       : $CONDA_ENV_NAME"
echo "[build-base] python          : $PYTHON_VERSION"
echo "[build-base] habitat-lab repo: $HABITAT_LAB_REPO"
echo "[build-base] habitat-lab ref : $HABITAT_LAB_REF"

docker build \
  --build-arg "CONDA_ENV_NAME=$CONDA_ENV_NAME" \
  --build-arg "PYTHON_VERSION=$PYTHON_VERSION" \
  --build-arg "HABITAT_LAB_REPO=$HABITAT_LAB_REPO" \
  --build-arg "HABITAT_LAB_REF=$HABITAT_LAB_REF" \
  -f "$SCRIPT_DIR/Dockerfile.habitat_base" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"

echo "[build-base] OK"
docker images "$IMAGE_TAG"
