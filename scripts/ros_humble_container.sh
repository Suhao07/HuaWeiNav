#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="${ROS_HUMBLE_IMAGE_TAG:-strive-ros-humble:local}"
CONTAINER_NAME="${ROS_HUMBLE_CONTAINER_NAME:-strive-ros-humble-dev}"

usage() {
  cat <<EOF
Usage: scripts/ros_humble_container.sh <command> [args...]

Commands:
  build       Build a local ROS2 Humble image for STRIVE real_robot overlay tests.
  shell       Open an interactive shell in the image.
  test        Run offline acceptance checks inside the image.
  build-ws    Build real_robot/ros2_ws inside the image.
  clean       Remove the dev container if it exists.

Notes:
  The host is not modified. This is the supported path on Ubuntu 24.04 hosts,
  because ROS2 Humble deb packages target Ubuntu 22.04 Jammy.

Environment:
  ROS_HUMBLE_IMAGE_TAG=${IMAGE_TAG}
  ROS_HUMBLE_RUN_AS_ROOT=1  # optional, defaults to current host uid/gid
  STRIVE_COLCON_BUILD_BASE=/tmp/strive_colcon_build
  STRIVE_COLCON_INSTALL_BASE=/tmp/strive_colcon_install
  STRIVE_COLCON_LOG_BASE=/tmp/strive_colcon_log
  SUDO_STDIN_PASSWORD=...  # optional, used only when Docker requires sudo
EOF
}

sudo_run() {
  if [[ -n "${SUDO_STDIN_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_STDIN_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sudo_run docker "$@"
  fi
}

build_image() {
  docker_cmd build \
    --build-arg INSTALL_ML_DEPS="${INSTALL_ML_DEPS:-0}" \
    --build-arg INSTALL_LLM_DEPS="${INSTALL_LLM_DEPS:-0}" \
    -f "${REPO_ROOT}/docker/Dockerfile.real_robot" \
    -t "${IMAGE_TAG}" \
    "${REPO_ROOT}"
}

run_in_image() {
  local tty_args=()
  local user_args=()
  if [[ -t 0 && -t 1 ]]; then
    tty_args=(-it)
  fi
  if [[ "${ROS_HUMBLE_RUN_AS_ROOT:-0}" != "1" ]]; then
    user_args=(--user "$(id -u):$(id -g)" -e "HOME=/tmp/strive_humble_home")
  fi
  docker_cmd rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker_cmd run --rm "${tty_args[@]}" \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    -e "PYTHONUNBUFFERED=1" \
    "${user_args[@]}" \
    -v "${REPO_ROOT}:/workspace/STRIVE_LIVE:rw" \
    "${IMAGE_TAG}" \
    bash -lc "mkdir -p \"\${HOME}\" && $*"
}

cmd="${1:-}"
shift || true
case "${cmd}" in
  build)
    build_image
    ;;
  shell)
    run_in_image 'cd /workspace/STRIVE_LIVE && source /opt/ros/humble/setup.bash && exec bash'
    ;;
  test)
    run_in_image 'cd /workspace/STRIVE_LIVE && bash scripts/check_real_robot_acceptance.sh'
    ;;
  build-ws)
    run_in_image 'cd /workspace/STRIVE_LIVE && set +u && source /opt/ros/humble/setup.bash && set -u && STRIVE_COLCON_LOG_BASE="${STRIVE_COLCON_LOG_BASE:-/tmp/strive_colcon_log}" bash scripts/build_real_robot_ros_ws.sh --build-base "${STRIVE_COLCON_BUILD_BASE:-/tmp/strive_colcon_build}" --install-base "${STRIVE_COLCON_INSTALL_BASE:-/tmp/strive_colcon_install}"'
    ;;
  clean)
    docker_cmd rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
