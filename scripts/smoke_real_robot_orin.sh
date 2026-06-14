#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-strive-real-robot:humble}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
LIVOX_SETUP="${LIVOX_SETUP:-/home/orin26/code/ws_livox/install/setup.bash}"
POINT_LIO_SETUP="${POINT_LIO_SETUP:-/home/orin26/code/point_lio_ws/install/setup.bash}"
CHECK_CAMERA="${CHECK_CAMERA:-1}"
CHECK_DETECTOR_INIT="${CHECK_DETECTOR_INIT:-0}"
DETECTOR_INIT_TIMEOUT="${DETECTOR_INIT_TIMEOUT:-180}"
REQUIRE_ML="${REQUIRE_ML:-0}"
REQUIRE_ASSETS="${REQUIRE_ASSETS:-0}"
REQUIRE_LIO="${REQUIRE_LIO:-0}"
HZ_TIMEOUT="${HZ_TIMEOUT:-7}"
ECHO_TIMEOUT="${ECHO_TIMEOUT:-5}"

if [[ -z "${SYSNAV_MOBILECLIP_BLT_TS_PATH:-}" && -n "${SYSNAV_MOBILECLIP_BLT_PATH:-}" ]]; then
  mobileclip_ts_candidate="${SYSNAV_MOBILECLIP_BLT_PATH%.*}.ts"
  if [[ -f "${mobileclip_ts_candidate}" ]]; then
    SYSNAV_MOBILECLIP_BLT_TS_PATH="${mobileclip_ts_candidate}"
  fi
fi

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

section() {
  printf '\n== %s ==\n' "$1"
}

run_maybe_sudo() {
  if "$@" >/tmp/strive_sudo_probe.out 2>/tmp/strive_sudo_probe.err; then
    cat /tmp/strive_sudo_probe.out
    return 0
  fi
  if have_cmd sudo; then
    if [[ -n "${SUDO_STDIN_PASSWORD:-}" ]]; then
      printf '%s\n' "${SUDO_STDIN_PASSWORD}" | sudo -S -p '' "$@"
    else
      sudo "$@"
    fi
  else
    cat /tmp/strive_sudo_probe.err >&2
    return 1
  fi
}

source_ros() {
  if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS setup not found: ${ROS_SETUP}" >&2
    return 1
  fi
  set +u
  source "${ROS_SETUP}"
  [[ -f "${LIVOX_SETUP}" ]] && source "${LIVOX_SETUP}"
  [[ -f "${POINT_LIO_SETUP}" ]] && source "${POINT_LIO_SETUP}"
  set -u
}

append_jetson_nvidia_library_args() {
  local -n args_ref=$1
  local cuda_home="${CUDA_HOME_HOST:-}"
  local ld_paths=()

  if [[ -z "${cuda_home}" && -e /usr/local/cuda ]]; then
    cuda_home="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
  fi
  if [[ -n "${cuda_home}" && -d "${cuda_home}" ]]; then
    args_ref+=(-v "${cuda_home}:${cuda_home}:ro")
    ld_paths+=("${cuda_home}/lib64" "${cuda_home}/targets/aarch64-linux/lib")
  fi

  for lib in /usr/lib/aarch64-linux-gnu/libcudnn*.so*; do
    [[ -e "${lib}" ]] || continue
    args_ref+=(-v "${lib}:${lib}:ro")
  done
  ld_paths+=("/usr/lib/aarch64-linux-gnu" "/usr/lib/aarch64-linux-gnu/tegra")

  local cudss_dir="${CUDSS_HOST_DIR:-/opt/nvidia/cudss/lib}"
  if [[ -d "${cudss_dir}" ]]; then
    args_ref+=(-v "${cudss_dir}:/opt/nvidia/cudss/lib:ro")
    ld_paths+=("/opt/nvidia/cudss/lib")
  fi

  if ((${#ld_paths[@]})); then
    local joined
    joined="$(IFS=:; echo "${ld_paths[*]}")"
    args_ref+=(-e "LD_LIBRARY_PATH=${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")
  fi
}

append_model_asset_args() {
  local -n args_ref=$1
  local mounted_dirs=":"

  for path_var in SYSNAV_DETECTOR_MODEL_PATH SYSNAV_SAM2_CHECKPOINT; do
    local model_path="${!path_var:-}"
    [[ -n "${model_path}" ]] || continue
    args_ref+=(-e "${path_var}=${model_path}")
    if [[ -f "${model_path}" ]]; then
      local model_dir
      model_dir="$(cd "$(dirname "${model_path}")" && pwd)"
      if [[ "${mounted_dirs}" != *":${model_dir}:"* ]]; then
        args_ref+=(-v "${model_dir}:${model_dir}:ro")
        mounted_dirs+="${model_dir}:"
      fi
    fi
  done
  if [[ -n "${SYSNAV_DETECTOR_MODEL_TYPE:-}" ]]; then
    args_ref+=(-e "SYSNAV_DETECTOR_MODEL_TYPE=${SYSNAV_DETECTOR_MODEL_TYPE}")
  fi
  if [[ -n "${SYSNAV_MOBILECLIP_BLT_PATH:-}" ]]; then
    if [[ -f "${SYSNAV_MOBILECLIP_BLT_PATH}" ]]; then
      args_ref+=(-v "${SYSNAV_MOBILECLIP_BLT_PATH}:/workspace/STRIVE/mobileclip_blt.pt:ro")
      if [[ -z "${SYSNAV_MOBILECLIP_BLT_TS_PATH:-}" && -f "${SYSNAV_MOBILECLIP_BLT_PATH%.*}.ts" ]]; then
        SYSNAV_MOBILECLIP_BLT_TS_PATH="${SYSNAV_MOBILECLIP_BLT_PATH%.*}.ts"
      fi
    else
      echo "SYSNAV_MOBILECLIP_BLT_PATH does not exist: ${SYSNAV_MOBILECLIP_BLT_PATH}" >&2
    fi
  fi
  if [[ -n "${SYSNAV_MOBILECLIP_BLT_TS_PATH:-}" ]]; then
    if [[ -f "${SYSNAV_MOBILECLIP_BLT_TS_PATH}" ]]; then
      args_ref+=(-e "SYSNAV_MOBILECLIP_BLT_TS_PATH=${SYSNAV_MOBILECLIP_BLT_TS_PATH}")
      args_ref+=(-v "${SYSNAV_MOBILECLIP_BLT_TS_PATH}:/workspace/STRIVE/mobileclip_blt.ts:ro")
    else
      echo "SYSNAV_MOBILECLIP_BLT_TS_PATH does not exist: ${SYSNAV_MOBILECLIP_BLT_TS_PATH}" >&2
    fi
  fi
  if [[ -n "${SYSNAV_CLIP_VIT_B32_PATH:-}" ]]; then
    if [[ -f "${SYSNAV_CLIP_VIT_B32_PATH}" ]]; then
      args_ref+=(-e "SYSNAV_CLIP_VIT_B32_PATH=${SYSNAV_CLIP_VIT_B32_PATH}")
      args_ref+=(-v "${SYSNAV_CLIP_VIT_B32_PATH}:/root/.cache/clip/ViT-B-32.pt:ro")
    else
      echo "SYSNAV_CLIP_VIT_B32_PATH does not exist: ${SYSNAV_CLIP_VIT_B32_PATH}" >&2
    fi
  fi
}

section "Host"
hostname || true
cat /etc/nv_tegra_release 2>/dev/null || true
apt show nvidia-jetpack 2>/dev/null | sed -n '1,12p' || true
python3 --version || true

section "Docker"
run_maybe_sudo docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E "(${IMAGE_TAG}|REPOSITORY)" || true
run_maybe_sudo docker info --format '{{json .Runtimes}}' | grep -o 'nvidia' || true
run_maybe_sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | grep strive || true

section "Devices"
ls -l /dev/video* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
if have_cmd v4l2-ctl; then
  v4l2-ctl --list-devices 2>/dev/null || true
  v4l2-ctl -d /dev/video0 --list-formats-ext 2>/dev/null | sed -n '1,80p' || true
fi

section "Deployment Assets"
asset_missing=0
for var_name in SYSNAV_DETECTOR_MODEL_PATH SYSNAV_SAM2_CHECKPOINT; do
  asset_path="${!var_name:-}"
  if [[ -z "${asset_path}" ]]; then
    echo "${var_name}: unset"
    asset_missing=1
  elif [[ -f "${asset_path}" ]]; then
    echo "${var_name}: OK ${asset_path}"
  else
    echo "${var_name}: MISSING ${asset_path}"
    asset_missing=1
  fi
done
for var_name in SYSNAV_MOBILECLIP_BLT_PATH SYSNAV_MOBILECLIP_BLT_TS_PATH SYSNAV_CLIP_VIT_B32_PATH; do
  asset_path="${!var_name:-}"
  if [[ -z "${asset_path}" ]]; then
    echo "${var_name}: unset"
  elif [[ -f "${asset_path}" ]]; then
    echo "${var_name}: OK ${asset_path}"
  else
    echo "${var_name}: MISSING ${asset_path}"
    [[ "${REQUIRE_ASSETS}" == "1" ]] && asset_missing=1
  fi
done
if [[ "${REQUIRE_ASSETS}" == "1" && "${asset_missing}" != "0" ]]; then
  echo "Required deployment assets are missing." >&2
  exit 3
fi

section "ROS Graph"
source_ros
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true
sleep 1
ros2 node list | sort || true
ros2 topic list -t | sort || true

section "LIO Process Snapshot"
ps -o pid,etime,pcpu,pmem,stat,command \
  -p "$(pgrep -d, -f 'livox_ros_driver2_node|pointlio_mapping|static_transform_publisher' || echo 0)" \
  2>/dev/null || true
tmux capture-pane -pt livox_odom -S -80 2>/dev/null | sed -n '1,120p' || true

section "Hardware Topic Info"
for topic in \
  /livox/lidar \
  /livox/imu \
  /cloud_registered \
  /cloud_registered_body \
  /aft_mapped_to_init \
  /base_odom \
  /path \
  /registered_scan \
  /state_estimation \
  /camera/image \
  /way_point \
  /cmd_vel; do
  echo "--- ${topic}"
  ros2 topic info -v "${topic}" 2>&1 | sed -n '1,70p' || true
done

section "Hardware Topic Samples"
for topic in /livox/imu /cloud_registered /aft_mapped_to_init /path; do
  echo "--- hz ${topic}"
  ROS_DISABLE_DAEMON=1 timeout "${HZ_TIMEOUT}" ros2 topic hz "${topic}" --window 5 2>&1 | sed -n '1,30p' || true
done
echo "--- /aft_mapped_to_init once"
ROS_DISABLE_DAEMON=1 timeout "${ECHO_TIMEOUT}" ros2 topic echo --once /aft_mapped_to_init 2>&1 | sed -n '1,80p' || true
echo "--- /cloud_registered header once"
ROS_DISABLE_DAEMON=1 timeout "${ECHO_TIMEOUT}" ros2 topic echo --once /cloud_registered --field header 2>&1 | sed -n '1,60p' || true

if [[ "${REQUIRE_LIO}" == "1" ]]; then
  section "Required LIO Sample Gate"
  lio_missing=0
  for spec in \
    "/livox/lidar header" \
    "/livox/imu header" \
    "/cloud_registered header" \
    "/aft_mapped_to_init header"; do
    topic="${spec% *}"
    field="${spec##* }"
    echo "--- required ${topic} ${field}"
    sample_output="$(
      ROS_DISABLE_DAEMON=1 timeout "${ECHO_TIMEOUT}" \
        ros2 topic echo --once "${topic}" --field "${field}" 2>&1 || true
    )"
    printf '%s\n' "${sample_output}" | sed -n '1,40p'
    if [[ -z "${sample_output//[[:space:]]/}" ]] \
      || grep -Eq 'Unknown topic|does not appear to be published|Could not determine the type' <<< "${sample_output}" \
      || ! grep -q '^stamp:' <<< "${sample_output}"; then
      echo "Missing required LIO sample: ${topic}" >&2
      lio_missing=1
    fi
  done
  if [[ "${lio_missing}" != "0" ]]; then
    echo "Required LIO samples are missing. Start/restart LIO with scripts/start_orin_lio_for_strive.sh." >&2
    exit 5
  fi
fi

section "Container ROS Smoke"
DOCKER_GPU_ARGS=()
if run_maybe_sudo docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  DOCKER_GPU_ARGS=(
    --runtime=nvidia
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}"
  )
  append_jetson_nvidia_library_args DOCKER_GPU_ARGS
elif have_cmd nvidia-smi; then
  DOCKER_GPU_ARGS=(--gpus all)
fi
DOCKER_DDS_ARGS=(-e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}")

run_maybe_sudo docker run --rm --network host --ipc=host "${DOCKER_GPU_ARGS[@]}" "${DOCKER_DDS_ARGS[@]}" "${IMAGE_TAG}" bash -lc '
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /workspace/STRIVE/real_robot/ros2_ws/install/setup.bash
set -u
ros2 pkg prefix strive_sysnav_bringup
ros2 pkg prefix usb_cam
ros2 pkg prefix topic_tools
python3 -c "import real_robot.contracts, real_robot.sysnav_ros_adapters; print(\"real_robot imports ok\")"
ros2 launch strive_sysnav_bringup sysnav_detection_mapping.launch.py --show-args >/tmp/strive_launch_args.txt
grep -E "camera_topic|cloud_topic|odom_topic|start_usb_cam|usb_video_device" /tmp/strive_launch_args.txt
'

if [[ "${REQUIRE_LIO}" == "1" ]]; then
  section "Container LIO Receive Smoke"
  container_lio_output="$(
    run_maybe_sudo docker run --rm --network host --ipc=host \
      "${DOCKER_GPU_ARGS[@]}" "${DOCKER_DDS_ARGS[@]}" \
      "${IMAGE_TAG}" bash -lc '
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
echo "--- container /aft_mapped_to_init header"
timeout 8 ros2 topic echo --once /aft_mapped_to_init --field header
echo "--- container /cloud_registered header"
timeout 8 ros2 topic echo --once /cloud_registered --field header
'
  )"
  printf '%s\n' "${container_lio_output}" | sed -n '1,80p'
  if ! grep -q '^stamp:' <<< "${container_lio_output}"; then
    echo "Container could not receive required LIO topics. Check Docker DDS settings." >&2
    exit 6
  fi
fi

section "Container ML Imports"
ml_import_output="$(
run_maybe_sudo docker run --rm --ipc=host "${DOCKER_GPU_ARGS[@]}" "${DOCKER_DDS_ARGS[@]}" "${IMAGE_TAG}" bash -lc '
set +u
source /opt/ros/humble/setup.bash
source /workspace/STRIVE/real_robot/ros2_ws/install/setup.bash
set -u
python3 - <<'"'"'PY'"'"'
import importlib
import importlib.metadata as md

mods = [
    "torch",
    "torchvision",
    "ultralytics",
    "supervision",
    "open3d",
    "sklearn",
    "shapely",
    "hydra",
    "omegaconf",
    "iopath",
    "rerun",
    "cv2",
    "semantic_mapping.semantic_map_new",
    "semantic_mapping.semantic_mapping_node",
]
dist_names = {
    "cv2": "opencv-python",
    "hydra": "hydra-core",
    "rerun": "rerun-sdk",
    "sklearn": "scikit-learn",
    "semantic_mapping.semantic_map_new": None,
    "semantic_mapping.semantic_mapping_node": None,
}

def version_for(name):
    dist_name = dist_names.get(name, name)
    if not dist_name:
        return ""
    try:
        return f" version={md.version(dist_name)}"
    except md.PackageNotFoundError:
        return ""

for name in mods:
    try:
        mod = importlib.import_module(name)
        extra = ""
        if name == "torch":
            extra = f" cuda={mod.cuda.is_available()}{version_for(name)}"
        else:
            extra = version_for(name)
        print(f"{name}: OK{extra}")
    except Exception as exc:
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")
PY'
)"
printf '%s\n' "${ml_import_output}"
if [[ "${REQUIRE_ML}" == "1" ]] && grep -q ': FAIL ' <<< "${ml_import_output}"; then
  echo "Required ML imports are missing from ${IMAGE_TAG}." >&2
  exit 4
fi

if [[ "${CHECK_DETECTOR_INIT}" == "1" ]]; then
  section "Container Detector Init Smoke"
  MODEL_ASSET_ARGS=()
  append_model_asset_args MODEL_ASSET_ARGS
  run_maybe_sudo timeout "${DETECTOR_INIT_TIMEOUT}" docker run --rm --network host --ipc=host "${DOCKER_GPU_ARGS[@]}" "${DOCKER_DDS_ARGS[@]}" "${MODEL_ASSET_ARGS[@]}" "${IMAGE_TAG}" bash -lc '
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /workspace/STRIVE/real_robot/ros2_ws/install/setup.bash
set -u
python3 - <<'"'"'PY'"'"'
import os

import rclpy
from semantic_mapping.detection_node import DetectNode

model_path = os.environ.get("SYSNAV_DETECTOR_MODEL_PATH")
if not model_path:
    raise RuntimeError("SYSNAV_DETECTOR_MODEL_PATH is required for detector init smoke")
args = ["--ros-args", "-p", f"detector_model_path:={model_path}"]
model_type = os.environ.get("SYSNAV_DETECTOR_MODEL_TYPE")
if model_type:
    args.extend(["-p", f"detector_model_type:={model_type}"])
rclpy.init(args=args)
node = DetectNode()
print(f"detector init ok: {type(node.grounding_model).__name__}")
node.destroy_node()
rclpy.shutdown()
PY'
fi

if [[ "${CHECK_CAMERA}" == "1" && -e /dev/video0 ]]; then
  section "Container Camera Smoke"
  CAMERA_DEVICE_ARGS=(--device /dev/video0:/dev/video0)
  [[ -e /dev/video1 ]] && CAMERA_DEVICE_ARGS+=(--device /dev/video1:/dev/video1)
  run_maybe_sudo docker run --rm --network host --ipc=host "${DOCKER_GPU_ARGS[@]}" "${DOCKER_DDS_ARGS[@]}" \
    "${CAMERA_DEVICE_ARGS[@]}" \
    "${IMAGE_TAG}" bash -lc '
set -eo pipefail
v4l2-ctl --list-devices 2>&1 || true
set +u
source /opt/ros/humble/setup.bash
set -u
timeout 8 ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video0 \
  -p image_width:=640 \
  -p image_height:=480 \
  -p pixel_format:=yuyv \
  -r image_raw:=/camera/image 2>&1 | sed -n "1,120p" || true
'
fi

section "Expected Real-Robot Bringup"
cat <<'EOF'
sudo bash docker/run_real_robot_sysnav_stack.sh \
  platform:=mecanum \
  cloud_topic:=/cloud_registered \
  odom_topic:=/aft_mapped_to_init \
  start_usb_cam:=true \
  usb_video_device:=/dev/video0 \
  camera_topic:=/camera/image
EOF

section "Safety Note"
cat <<'EOF'
This smoke script does not publish /way_point or /cmd_vel and does not start the
AgileX/WebSocket bridge. It only observes the ROS graph and runs bounded camera
and container checks.
EOF
