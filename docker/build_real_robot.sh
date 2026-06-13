#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-strive-real-robot:humble}"
INSTALL_ML_DEPS="${INSTALL_ML_DEPS:-0}"
INSTALL_LLM_DEPS="${INSTALL_LLM_DEPS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[build-real-robot] image          : ${IMAGE_TAG}"
echo "[build-real-robot] install ML deps: ${INSTALL_ML_DEPS}"
echo "[build-real-robot] install LLM deps: ${INSTALL_LLM_DEPS}"
echo "[build-real-robot] context        : ${REPO_ROOT}"

docker build \
  --build-arg "INSTALL_ML_DEPS=${INSTALL_ML_DEPS}" \
  --build-arg "INSTALL_LLM_DEPS=${INSTALL_LLM_DEPS}" \
  -f "${SCRIPT_DIR}/Dockerfile.real_robot" \
  -t "${IMAGE_TAG}" \
  "${REPO_ROOT}"

echo "[build-real-robot] OK"
docker images "${IMAGE_TAG}"
