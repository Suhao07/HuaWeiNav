#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${STRIVE_REAL_ROBOT_WS:-${REPO_ROOT}/real_robot/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
OVERLAY_SETUP="${WS_DIR}/install/setup.bash"

usage() {
  cat <<EOF
Usage: scripts/run_real_robot_instruction_runtime.sh [launch args...]

Starts only the STRIVE high-level instruction runtime node.

Safe defaults are defined in the launch file:
  dry_run:=true
  policy_mode:=wait

Example:
  scripts/run_real_robot_instruction_runtime.sh \\
    instruction:="find a book" \\
    dry_run:=true \\
    policy_mode:=wait \\
    run_directory:=/tmp/strive_real_robot_runtime

Semantic snapshot dry-run:
  scripts/run_real_robot_instruction_runtime.sh \\
    instruction:="find a book" \\
    dataset_target:=book \\
    dry_run:=true \\
    policy_mode:=semantic_snapshot \\
    instruction_plan_backend:=llm \\
    enable_final_verifier:=false \\
    run_directory:=/tmp/strive_real_robot_runtime

Final verifier dry-run:
  scripts/run_real_robot_instruction_runtime.sh \\
    instruction:="find a book" \\
    dataset_target:=book \\
    dry_run:=true \\
    dry_run_status:=reached \\
    policy_mode:=semantic_snapshot \\
    instruction_plan_backend:=llm \\
    enable_final_verifier:=true \\
    run_directory:=/tmp/strive_real_robot_runtime_verifier
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 2
fi
if [[ ! -f "${OVERLAY_SETUP}" ]]; then
  echo "Overlay setup not found: ${OVERLAY_SETUP}" >&2
  echo "Run scripts/build_real_robot_ros_ws.sh first." >&2
  exit 2
fi

set +u
source "${ROS_SETUP}"
source "${OVERLAY_SETUP}"
set -u

# The ROS package lives inside real_robot/ros2_ws, while STRIVE contracts and
# adapters live in the repository-level real_robot Python package.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec ros2 launch strive_sysnav_bringup strive_instruction_runtime.launch.py "$@"
