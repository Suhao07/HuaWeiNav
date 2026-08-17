#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DEST_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST_DIR="${1:-${DEFAULT_DEST_PARENT}/STRIVE_code_only_${STAMP}}"

usage() {
  cat <<EOF
Usage: scripts/export_code_only.sh [DEST_DIR]

Exports the repository as code only. The export excludes:
  .git and local env files
  weights/checkpoints/model artifacts
  rosbag files and recorded datasets
  ROS build/install/log products
  Python/test/cache/runtime output directories
  the STRIVE paper PDF in docs/

Default destination:
  ${DEFAULT_DEST_PARENT}/STRIVE_code_only_${STAMP}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

mkdir -p "$(dirname "${DEST_DIR}")"
DEST_PARENT_ABS="$(cd "$(dirname "${DEST_DIR}")" && pwd)"
DEST_ABS="${DEST_PARENT_ABS}/$(basename "${DEST_DIR}")"
REPO_ABS="$(cd "${REPO_ROOT}" && pwd)"
case "${DEST_ABS}" in
  "${REPO_ABS}"|"${REPO_ABS}"/*)
    echo "Destination must be outside the repository: ${DEST_DIR}" >&2
    exit 2
    ;;
esac

mkdir -p "${DEST_DIR}"

RSYNC_EXCLUDES=(
  "--exclude=.git/"
  "--exclude=.agents/"
  "--exclude=.codex/"
  "--exclude=.env"
  "--exclude=.env.*"
  "--exclude=*.env"
  "--exclude=__pycache__/"
  "--exclude=.pytest_cache/"
  "--exclude=.mypy_cache/"
  "--exclude=.ruff_cache/"
  "--exclude=logs/"
  "--exclude=output/"
  "--exclude=outputs/"
  "--exclude=data/"
  "--exclude=datasets/"
  "--exclude=weights/"
  "--exclude=checkpoints/"
  "--exclude=tmp/"
  "--exclude=tmp*/"
  "--exclude=real_robot/ros2_ws/build/"
  "--exclude=real_robot/ros2_ws/install/"
  "--exclude=real_robot/ros2_ws/log/"
  "--exclude=*.pyc"
  "--exclude=*.pt"
  "--exclude=*.pth"
  "--exclude=*.engine"
  "--exclude=*.onnx"
  "--exclude=*.npy"
  "--exclude=*.npz"
  "--exclude=*.bag"
  "--exclude=*.db3"
  "--exclude=*.mcap"
  "--exclude=*.mp4"
  "--exclude=*.avi"
  "--exclude=*.mov"
  "--exclude=*.ply"
  "--exclude=references/papers/Zhu et al. - 2025 - STRIVE Structured Representation Integrating VLM Reasoning for Efficient Object Navigation.pdf"
)

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${REPO_ROOT}/" "${DEST_DIR}/"
else
  echo "rsync is required for code-only export." >&2
  exit 2
fi

echo "Exported code-only repository to: ${DEST_DIR}"
