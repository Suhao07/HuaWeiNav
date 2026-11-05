#!/usr/bin/env bash
# Host one-click entry; sensor drivers/LIO remain externally owned.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${STRIVE_EXECUTION_MODE:-preview}"
motion="${ALLOW_REAL_MOTION:-false}"
contract="${CONTROL_CONTRACT_FILE:-}"
requested_uri="${AGILEX_URI:-}"
source "${STRIVE_ROBOT_PROFILE:-$ROOT/real_robot/profiles/orin26_livox_mid360_d435i_yoloworld_candidate.env}"
[[ -z "$requested_uri" ]] || export AGILEX_URI="$requested_uri"
export STRIVE_EXECUTION_MODE="$mode" ALLOW_REAL_MOTION="$motion" CONTROL_CONTRACT_FILE="$contract"
if [[ "${1:-}" == --check ]]; then exec bash "$ROOT/scripts/start_real_robot_navigation.sh" --check; fi
python3 "$ROOT/real_robot/navigation_parameters.py"
export START_FORMAL_NAVIGATION=1 START_SEMANTIC_MAPPING=true
export IMAGE_TAG="${NAV_IMAGE_TAG:-huawei-vln-realworld:orin-r36.5-navigation}"
export CONTAINER_NAME="${NAV_CONTAINER_NAME:-huawei-vln-navigation}"
export STRIVE_RUN_DIRECTORY="${NAV_RUN_DIRECTORY:-/workspace/STRIVE/output/runtime/navigation}"
export LLM_OFFLINE=1
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/STRIVE/real_robot/config/fastdds_udp.xml
if docker ps --format '{{.Names}}' | grep -Eq '^huawei-vln-(navigation|shadow-acceptance|static-correspondence-candidate)$'; then
  echo 'A STRIVE stack is already running. Export its runtime data and stop it before starting another.' >&2
  exit 3
fi
exec bash "$ROOT/docker/run_real_robot_sysnav_stack.sh" \
 platform:=mecanum camera_topic:="$CAMERA_TOPIC" cloud_topic:="$CLOUD_TOPIC" odom_topic:="$ODOM_TOPIC" \
 detector_device:="$SYSNAV_DETECTOR_DEVICE" detector_imgsz:="$SYSNAV_DETECTOR_IMGSZ" \
 start_usb_cam:=false usb_camera_info_url:="$USB_CAMERA_INFO_URL" "$@"
