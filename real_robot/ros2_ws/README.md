# STRIVE Real-Robot ROS2 Overlay

This workspace vendors the first SysNav detector and semantic mapping stack
needed by STRIVE real-robot deployment.

## Packages

```text
src/tare_planner
  Message-only compatibility package. It provides the SysNav message types
  consumed by semantic_mapping, including DetectionResult, ObjectNode,
  ObjectNodeList, RoomNode, RoomNodeList, TargetObjectInstruction, and related
  VLM/navigation query messages.

src/semantic_mapping
  Vendored SysNav detector_node and semantic_mapping_node.
  It subscribes /camera/image, /registered_scan, /state_estimation, and
  publishes /detection_result and /object_nodes_list.

src/strive_sysnav_bringup
  Launch-only package for starting detection_node and semantic_mapping_node
  inside the STRIVE overlay.
```

The full SysNav C++ exploration/local-planner package is intentionally not
compiled in this overlay yet. STRIVE publishes `/way_point`; a real robot can
consume that topic through an existing SysNav/Nav2/local-planner stack.

## Build

```bash
cd /home/ubuntu/WorkSpace/project/Huawei\ Nav/Code/STRIVE
bash scripts/build_real_robot_ros_ws.sh
```

The script builds:

```text
tare_planner
semantic_mapping
strive_sysnav_bringup
```

## Runtime Assets

Detector and SAM2 weights are deployment assets and are not committed to the
repository. Configure them explicitly:

```bash
export SYSNAV_DETECTOR_MODEL_TYPE=yoloe
export SYSNAV_DETECTOR_MODEL_PATH=/path/to/yoloe-26x-seg.engine
export SYSNAV_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_base_plus.pt
```

If the model path is omitted, `detection_node` falls back to the package-local
default under `semantic_mapping/external`. The run script checks explicitly
provided paths before launching.

## Run

For the real robot, use the guarded framework entrypoint from inside the
container. It checks the live LIO topics, keeps lower controller startup blocked
by default, then starts camera, detector, and semantic mapping:

```bash
bash scripts/start_real_robot_framework.sh
```

The robot-side Docker entrypoint calls this script automatically:

```bash
SUDO_STDIN_PASSWORD=1 ./docker_en.sh start
```

The lower controller remains blocked unless these are set explicitly:

```bash
BLOCK_LOWER_CONTROLLER=0 \
ENABLE_LOWER_CONTROLLER=1 \
LOWER_CONTROLLER_CMD='<controller launch command>' \
bash scripts/start_real_robot_framework.sh
```

```bash
cd /home/ubuntu/WorkSpace/project/Huawei\ Nav/Code/STRIVE
bash scripts/run_sysnav_detection_mapping.sh
```

Extra launch arguments can be passed through:

```bash
bash scripts/run_sysnav_detection_mapping.sh \
  platform:=mecanum \
  use_sim_time:=false
```

On the current Orin/Mid-360 robot, Point-LIO publishes the registered cloud
and odometry under its native topic names. Start the STRIVE overlay with
explicit remaps:

```bash
bash scripts/run_sysnav_detection_mapping.sh \
  platform:=mecanum \
  use_sim_time:=false \
  cloud_topic:=/cloud_registered \
  odom_topic:=/aft_mapped_to_init
```

If no camera driver is already publishing `/camera/image`, the bringup launch
can start `usb_cam` from the USB camera device and remap it into STRIVE:

```bash
bash docker/run_real_robot_sysnav_stack.sh \
  platform:=mecanum \
  cloud_topic:=/cloud_registered \
  odom_topic:=/aft_mapped_to_init \
  start_usb_cam:=true \
  usb_video_device:=/dev/video0 \
  camera_topic:=/camera/image
```

Expected topics:

```text
Input:
  /camera/image
  /registered_scan
  /state_estimation
  /viewpoint_rep_header

Output:
  /detection_result
  /object_nodes_list
  /annotated_image_detection
  /annotated_image
  /cloud_image
```

STRIVE consumes `/object_nodes_list` and `/room_nodes_list` through
`real_robot.sysnav_runtime.SysNavSemanticMapBridge`, then publishes waypoint
goals with `real_robot.sysnav_ros_adapters.RosWaypointController`.

Observed hardware topics on the Orin robot:

```text
Livox driver:
  /livox/lidar                 livox_ros_driver2/msg/CustomMsg
  /livox/imu                   sensor_msgs/msg/Imu

Point-LIO:
  /cloud_registered            sensor_msgs/msg/PointCloud2
  /cloud_registered_body       sensor_msgs/msg/PointCloud2
  /aft_mapped_to_init          nav_msgs/msg/Odometry
  /base_odom                   nav_msgs/msg/Odometry
  /path                        nav_msgs/msg/Path

USB camera device:
  /dev/video0, /dev/video1
```

The robot did not have `/registered_scan`, `/state_estimation`, `/camera/image`,
`/way_point`, or `/cmd_vel` active during the first smoke pass, so STRIVE bringup
must either use the launch remaps above or start the missing camera/local-planner
nodes before running the full stack.

Point-LIO's installed `mapping_mid360_orin.launch.py` loads
`publish.scan_publish_en: false` from its config, so `/cloud_registered` can
exist in the ROS graph without emitting live `PointCloud2` samples. For STRIVE,
start the Livox/LIO tmux session through the HuaWeiNav host helper, which keeps
the external repositories unchanged and applies runtime parameter overrides:

```bash
cd /home/orin26/code/HuaWeiNav
bash scripts/start_orin_lio_for_strive.sh
```

The helper starts `livox_ros_driver2` and runs `point_lio` with:

```text
publish.scan_publish_en:=true
```

`/cloud_registered_body` is optional for STRIVE and is disabled by default to
reduce Point-LIO load. Enable it only when debugging body-frame clouds:

```bash
ENABLE_BODY_CLOUD_PUBLISH=1 bash scripts/start_orin_lio_for_strive.sh
```

After this, the observed smoke rates were roughly:

```text
/livox/lidar          ~100 Hz
/aft_mapped_to_init   ~100 Hz
/cloud_registered     ~96-102 Hz
/cloud_registered_body ~9 Hz when ENABLE_BODY_CLOUD_PUBLISH=1
```

## Orin Smoke Check

Run the bounded smoke script on the robot before starting the full stack:

```bash
cd /home/orin26/code/HuaWeiNav
bash scripts/smoke_real_robot_orin.sh
```

For the hardware-topic gate, run:

```bash
IMAGE_TAG=huawei-nav-real:orin REQUIRE_LIO=1 CHECK_CAMERA=1 \
  bash scripts/smoke_real_robot_orin.sh
```

The smoke script only observes the ROS graph and starts short-lived container
checks. It does not publish `/way_point` or `/cmd_vel`, and it does not start the
AgileX/WebSocket bridge.

The current Orin is JetPack 6.2.2 / L4T R36.5 with Python 3.10. The final
real-robot runtime image is:

```text
huawei-nav-real:orin
```

It contains the ROS overlay and Jetson-compatible runtime packages. Large model
weights stay outside the image and are mounted by `docker_en.sh`.

```text
Required Python packages for full detector/mapping launch:
  torch
  torchvision
  ultralytics
  supervision
  open3d
  opencv-python==4.11.0.86
  scikit-learn
  shapely
  hydra-core / omegaconf / iopath
  rerun-sdk==0.18.2

Required deployment assets:
  SYSNAV_DETECTOR_MODEL_PATH
  SYSNAV_SAM2_CHECKPOINT
  SYSNAV_CLIP_VIT_B32_PATH (only needed by YOLOE .pt fallback models)
  SYSNAV_MOBILECLIP_BLT_TS_PATH (only needed by YOLOE .pt fallback models)
```

On the current Orin, the following smoke-test assets have been downloaded under
the HuaWeiNav checkout:

```text
SYSNAV_SAM2_CHECKPOINT=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/sam2/checkpoints/sam2.1_hiera_base_plus.pt
SYSNAV_DETECTOR_MODEL_PATH=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/yoloe-11s-seg.pt
SYSNAV_CLIP_VIT_B32_PATH=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/ViT-B-32.pt
SYSNAV_MOBILECLIP_BLT_TS_PATH=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/mobileclip_blt.ts
SYSNAV_DETECTOR_MODEL_TYPE=yoloe
```

`yoloe-11s-seg.pt` is a lightweight public fallback for startup validation. The
preferred offline real-robot asset remains a TensorRT engine exported with the
deployment vocabulary, such as `yoloe-26x-seg.engine`, because YOLOE `.pt`
models may need extra text-encoder dependencies during `set_classes()`. If
`SYSNAV_MOBILECLIP_BLT_PATH=/.../mobileclip_blt.pt` is set and
`/.../mobileclip_blt.ts` exists next to it, the run scripts auto-mount the `.ts`
asset into `/workspace/STRIVE/mobileclip_blt.ts`.

The strict smoke pass used on the Orin was:

```bash
SYSNAV_DETECTOR_MODEL_TYPE=yoloe \
SYSNAV_DETECTOR_MODEL_PATH=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/yoloe-11s-seg.pt \
SYSNAV_SAM2_CHECKPOINT=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
SYSNAV_CLIP_VIT_B32_PATH=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/ViT-B-32.pt \
SYSNAV_MOBILECLIP_BLT_TS_PATH=/home/orin26/code/HuaWeiNav/real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/external/mobileclip_blt.ts \
IMAGE_TAG=huawei-nav-real:orin \
REQUIRE_ASSETS=1 REQUIRE_LIO=1 REQUIRE_ML=1 CHECK_CAMERA=1 CHECK_DETECTOR_INIT=0 \
HZ_TIMEOUT=3 ECHO_TIMEOUT=5 \
bash scripts/smoke_real_robot_orin.sh
```

Use the single real-robot Docker entrypoint for deployment:

```bash
cd /home/orin26/code/HuaWeiNav
SUDO_STDIN_PASSWORD=1 ./docker_en.sh start
SUDO_STDIN_PASSWORD=1 ./docker_en.sh enter
SUDO_STDIN_PASSWORD=1 ./docker_en.sh status
```

The real-robot Docker runner defaults `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`.
On this Orin, ROS graph discovery worked from the container without it, but
host-published LIO data did not cross the Docker boundary until FastDDS shared
memory transport was disabled for the container.

## Motion Interface

STRIVE should stay above the low-level controller boundary:

```text
STRIVE NavigationIntent / MotionGoal
  -> RosWaypointController
  -> /way_point
  -> existing local planner / path follower / PD controller
  -> /cmd_vel or chassis bridge
```

The reference PD controller observed on the robot side consumes ego-frame
waypoint arrays on `/waypoint` and publishes `geometry_msgs/Twist` on `/cmd_vel`.
That repository is only a reference for topic contracts; do not modify it from
this overlay. If the robot does not run a `/way_point` consumer, add a bridge in
HuaWeiNav or the SysNav/local-planner layer that converts `/way_point` goals into
the controller's expected local path/waypoint format.
