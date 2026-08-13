#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE_DIR="${REPO_ROOT}/real_robot/profiles"

profile_name="${1:-}"
command_name="${2:-check}"
shift $(( $# >= 2 ? 2 : $# )) || true

usage() {
  cat <<EOF
Usage: scripts/run_real_robot_profile.sh PROFILE {check|build|smoke|lio-diagnostics|runtime-smoke|start|status|stop|logs} [args...]

PROFILE is a basename in real_robot/profiles, with or without .env.
All start operations retain safe defaults unless the profile has passed its
calibration and lower-controller gates.
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "${profile_name}" || "${profile_name}" == "-h" || "${profile_name}" == "--help" ]]; then
  usage
  exit 1
fi

profile_path="${PROFILE_DIR}/${profile_name%.env}.env"
if [[ ! -f "${profile_path}" ]]; then
  echo "Profile does not exist: ${profile_path}" >&2
  exit 2
fi

set -a
# Profiles are repository-maintained deployment configuration, not untrusted input.
# shellcheck disable=SC1090
source "${profile_path}"
set +a

# A selected robot profile is a complete, version-controlled hardware contract.
# Do not silently cascade a generic .env.realworld afterwards: it may contain
# stale values from a different robot (including its container name, camera,
# or topics).  Advanced users can still opt in to a supplementary env file by
# setting SYSNAV_ENV_FILE explicitly before invoking this helper.
if [[ -z "${SYSNAV_ENV_FILE:-}" ]]; then
  export SYSNAV_ENV_FILE="/dev/null"
fi

container_to_host_path() {
  local path="$1"
  if [[ "${path}" == /workspace/STRIVE/* ]]; then
    printf '%s/%s\n' "${REPO_ROOT}" "${path#/workspace/STRIVE/}"
  else
    printf '%s\n' "${path}"
  fi
}

check_profile() {
  local missing=0
  local path projection_host_path camera_info_host_path control_contract_host_path
  printf 'profile=%s\nversion=%s\n' "${ROBOT_PROFILE_NAME:-unknown}" "${ROBOT_PROFILE_VERSION:-unknown}"
  printf 'image=%s\ncontainer=%s\n' "${IMAGE_TAG}" "${CONTAINER_NAME}"
  printf 'lio=%s\ncloud=%s\nodom=%s\n' "${START_LIO}" "${CLOUD_TOPIC}" "${ODOM_TOPIC}"
  printf 'camera=%s\ndevice=%s\nformat=%sx%s %s @ %s FPS\n' \
    "${START_USB_CAM}" "${USB_VIDEO_DEVICE}" "${USB_IMAGE_WIDTH}" "${USB_IMAGE_HEIGHT}" \
    "${USB_PIXEL_FORMAT}" "${USB_FRAMERATE:-30.0}"
  printf 'camera_info_url=%s\n' "${USB_CAMERA_INFO_URL:-<unset>}"
  printf 'semantic_mapping=%s\ndry_run=%s\nlower_controller=%s\n' \
    "${START_SEMANTIC_MAPPING}" "${STRIVE_DRY_RUN}" "${STRIVE_LOWER_CONTROLLER_ENABLED}"

  for path in \
    "${SYSNAV_DETECTOR_MODEL_PATH}" \
    "${SYSNAV_SAM2_CHECKPOINT}" \
    "${SYSNAV_MOBILECLIP_BLT_TS_PATH}" \
    "${SYSNAV_CLIP_VIT_B32_PATH}"; do
    if [[ ! -f "${path}" ]]; then
      echo "missing asset: ${path}" >&2
      missing=1
    fi
  done

  if is_true "${START_USB_CAM}" && [[ ! -e "${USB_VIDEO_DEVICE}" ]]; then
    echo "camera device does not exist: ${USB_VIDEO_DEVICE}" >&2
    missing=1
  fi
  if [[ -n "${CAMERA_EXPECTED_HOST_PATH:-}" && ! -e "${CAMERA_EXPECTED_HOST_PATH}" ]]; then
    echo "expected stable camera device is missing: ${CAMERA_EXPECTED_HOST_PATH}" >&2
    missing=1
  fi
  if is_true "${START_USB_CAM}" && [[ -n "${USB_CAMERA_INFO_URL:-}" ]]; then
    if [[ "${USB_CAMERA_INFO_URL}" == file:///workspace/STRIVE/* ]]; then
      camera_info_host_path="$(container_to_host_path "${USB_CAMERA_INFO_URL#file://}")"
      if [[ ! -f "${camera_info_host_path}" ]]; then
        echo "camera_info_url points to a missing workspace file: ${camera_info_host_path}" >&2
        missing=1
      fi
    else
      echo "USB_CAMERA_INFO_URL must be a file:///workspace/STRIVE/... URL for this isolated profile." >&2
      missing=1
    fi
  fi

  if is_true "${START_SEMANTIC_MAPPING}"; then
    projection_host_path="$(container_to_host_path "${PROJECTION_CONFIG}")"
    if [[ ! -f "${projection_host_path}" ]]; then
      echo "projection profile is missing: ${projection_host_path}" >&2
      missing=1
    elif ! grep -Eq '^  calibration_status:[[:space:]]+calibrated[[:space:]]*$' "${projection_host_path}"; then
      echo "semantic mapping requires an approved calibration profile: ${projection_host_path}" >&2
      missing=1
    fi
    if ! is_true "${WAIT_FOR_LIO:-0}" || ! is_true "${REQUIRE_LIO_SAMPLE:-0}"; then
      echo "semantic mapping requires WAIT_FOR_LIO=true and REQUIRE_LIO_SAMPLE=true." >&2
      missing=1
    fi
  fi

  if is_true "${ALLOW_REAL_MOTION}"; then
    if ! is_true "${STRIVE_LOWER_CONTROLLER_ENABLED}" || ! is_true "${ENABLE_LOWER_CONTROLLER}" || is_true "${BLOCK_LOWER_CONTROLLER}" || is_true "${STRIVE_DRY_RUN}"; then
      echo "ALLOW_REAL_MOTION requires dry_run=false, lower_controller_enabled=true, ENABLE_LOWER_CONTROLLER=1, and BLOCK_LOWER_CONTROLLER=0" >&2
      missing=1
    fi
    if [[ -z "${CONTROL_CONTRACT_FILE:-}" ]]; then
      echo "ALLOW_REAL_MOTION requires CONTROL_CONTRACT_FILE." >&2
      missing=1
    else
      control_contract_host_path="$(container_to_host_path "${CONTROL_CONTRACT_FILE}")"
      if [[ ! -f "${control_contract_host_path}" ]]; then
        echo "Control contract does not exist: ${control_contract_host_path}" >&2
        missing=1
      else
        for required_gate in \
          '^[[:space:]]+approval_status:[[:space:]]+approved[[:space:]]*$' \
          '^[[:space:]]+allow_strive_waypoint_handoff:[[:space:]]+true[[:space:]]*$' \
          '^[[:space:]]+cmd_vel_direct_publish:[[:space:]]+false[[:space:]]*$' \
          '^[[:space:]]+emergency_stop_verified:[[:space:]]+true[[:space:]]*$'; do
          if ! grep -Eq "${required_gate}" "${control_contract_host_path}"; then
            echo "Control contract is missing an approved safety gate: ${required_gate}" >&2
            missing=1
          fi
        done
      fi
    fi
  fi
  [[ "${missing}" == "0" ]]
}

runtime_smoke() {
  check_profile

  local duration_s="${RUNTIME_SMOKE_DURATION_S:-8}"
  if ! [[ "${duration_s}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNTIME_SMOKE_DURATION_S must be a positive integer: ${duration_s}" >&2
    return 2
  fi

  local stamp container_name run_directory host_decision_file
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  container_name="${CONTAINER_NAME}-runtime-smoke-${stamp}"
  run_directory="/workspace/STRIVE/output/runtime/${ROBOT_PROFILE_NAME:-robot}/runtime-smoke-${stamp}"
  host_decision_file="${REPO_ROOT}/output/runtime/${ROBOT_PROFILE_NAME:-robot}/runtime-smoke-${stamp}/runtime_decisions.jsonl"

  # A runtime smoke is deliberately a closed-loop *logic* verification only:
  # no sensor driver, LIO manager, semantic mapping, waypoint, or lower
  # controller is allowed to start.  The result must be a persisted WAIT
  # decision until real sensor contracts are supplied.
  export CONTAINER_NAME="${container_name}"
  export START_LIO="0"
  export MANAGE_HOST_LIO="false"
  export START_USB_CAM="false"
  export START_SEMANTIC_MAPPING="false"
  export WAIT_FOR_LIO="0"
  export REQUIRE_LIO_SAMPLE="0"
  export START_STRIVE_RUNTIME="1"
  export STRIVE_POLICY_MODE="wait"
  export STRIVE_DRY_RUN="true"
  export STRIVE_LOWER_CONTROLLER_ENABLED="false"
  export ALLOW_REAL_MOTION="false"
  export BLOCK_LOWER_CONTROLLER="1"
  export ENABLE_LOWER_CONTROLLER="0"
  export STRIVE_RUN_DIRECTORY="${run_directory}"

  cleanup_runtime_smoke() {
    bash "${REPO_ROOT}/docker_en.sh" stop >/dev/null 2>&1 || true
  }
  trap cleanup_runtime_smoke EXIT INT TERM

  echo "runtime_smoke_container=${CONTAINER_NAME}"
  echo "runtime_smoke_duration_s=${duration_s}"
  echo "runtime_smoke_decisions=${host_decision_file}"
  bash "${REPO_ROOT}/docker_en.sh" start
  sleep "${duration_s}"

  if [[ ! -s "${host_decision_file}" ]]; then
    echo "Runtime smoke did not write a decision file: ${host_decision_file}" >&2
    return 3
  fi
  if ! grep -q '"mode": "wait"' "${host_decision_file}"; then
    echo "Runtime smoke must emit a dry-run WAIT decision: ${host_decision_file}" >&2
    return 3
  fi
  tail -n 1 "${host_decision_file}"
  cleanup_runtime_smoke
  trap - EXIT INT TERM
}

lio_diagnostics() {
  # This command is intentionally host-side and read-only.  It captures
  # endpoint QoS and bounded samples from an externally owned LIO graph, then
  # writes evidence only under this deployment workspace.
  export CLOUD_TOPIC ODOM_TOPIC LIO_LIDAR_TOPIC LIO_IMU_TOPIC POINT_LIO_NODE_NAME
  export HOST_FASTDDS_BUILTIN_TRANSPORTS="${HOST_FASTDDS_BUILTIN_TRANSPORTS:-}"
  export LIO_DIAGNOSTIC_TIMEOUT_S="${LIO_DIAGNOSTIC_TIMEOUT_S:-8}"
  exec bash "${REPO_ROOT}/scripts/capture_real_robot_lio_diagnostics.sh" "$@"
}

case "${command_name}" in
  check)
    check_profile
    ;;
  build)
    exec bash "${REPO_ROOT}/docker/build_real_robot.sh" "$@"
    ;;
  smoke|start)
    check_profile
    exec bash "${REPO_ROOT}/docker_en.sh" "${command_name}" "$@"
    ;;
  lio-diagnostics)
    lio_diagnostics "$@"
    ;;
  runtime-smoke)
    runtime_smoke
    ;;
  status|stop|logs|enter|exec)
    exec bash "${REPO_ROOT}/docker_en.sh" "${command_name}" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
