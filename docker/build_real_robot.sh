#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-huawei-vln-realworld:orin}"
INSTALL_ML_DEPS="${INSTALL_ML_DEPS:-0}"
INSTALL_LLM_DEPS="${INSTALL_LLM_DEPS:-1}"
UBUNTU_PORTS_MIRROR="${UBUNTU_PORTS_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports}"
JETSON_PYTORCH_INDEX="${JETSON_PYTORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
REAL_ROBOT_BASE_IMAGE="${REAL_ROBOT_BASE_IMAGE:-ros:humble-ros-base}"
SKIP_BASE_DEPS="${SKIP_BASE_DEPS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[build-real-robot] image          : ${IMAGE_TAG}"
echo "[build-real-robot] install ML deps: ${INSTALL_ML_DEPS}"
echo "[build-real-robot] install LLM deps: ${INSTALL_LLM_DEPS}"
echo "[build-real-robot] ubuntu mirror  : ${UBUNTU_PORTS_MIRROR}"
echo "[build-real-robot] Jetson PyTorch : ${JETSON_PYTORCH_INDEX}"
echo "[build-real-robot] base image     : ${REAL_ROBOT_BASE_IMAGE}"
echo "[build-real-robot] skip base deps : ${SKIP_BASE_DEPS}"
echo "[build-real-robot] context        : ${REPO_ROOT}"

docker build \
  --build-arg "INSTALL_ML_DEPS=${INSTALL_ML_DEPS}" \
  --build-arg "INSTALL_LLM_DEPS=${INSTALL_LLM_DEPS}" \
  --build-arg "UBUNTU_PORTS_MIRROR=${UBUNTU_PORTS_MIRROR}" \
  --build-arg "JETSON_PYTORCH_INDEX=${JETSON_PYTORCH_INDEX}" \
  --build-arg "REAL_ROBOT_BASE_IMAGE=${REAL_ROBOT_BASE_IMAGE}" \
  --build-arg "SKIP_BASE_DEPS=${SKIP_BASE_DEPS}" \
  -f "${SCRIPT_DIR}/Dockerfile.real_robot" \
  -t "${IMAGE_TAG}" \
  "${REPO_ROOT}"

echo "[build-real-robot] OK"
docker images "${IMAGE_TAG}"
