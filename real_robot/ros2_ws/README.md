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
  Launch and high-level runtime package. It starts detection_node,
  semantic_mapping_node, and the optional STRIVE instruction runtime node
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

`run_sysnav_detection_mapping.sh` does not start the STRIVE instruction runtime
unless explicitly requested:

```bash
START_STRIVE_RUNTIME=1 \
STRIVE_INSTRUCTION="find a book" \
STRIVE_DATASET_TARGET=book \
STRIVE_POLICY_MODE=semantic_snapshot \
STRIVE_INSTRUCTION_PLAN_BACKEND=rules \
STRIVE_DRY_RUN=true \
bash scripts/run_sysnav_detection_mapping.sh \
  platform:=mecanum \
  cloud_topic:=/cloud_registered \
  odom_topic:=/aft_mapped_to_init \
  camera_topic:=/camera/image
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

### Bag Replay Runtime

Use bag replay when the recorded bag already contains STRIVE-facing topics such
as `/object_nodes_list`, `/room_nodes_list`, `/aft_mapped_to_init`, and
`/camera/image`. This path does not start detector/mapping.

```bash
bash scripts/run_real_robot_bag_replay.sh /path/to/recorded_bag \
  instruction:="find a book" \
  dataset_target:=book \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  dry_run:=true \
  run_directory:=/tmp/strive_real_robot_bag_replay
```

If the bag used different topic names, set the corresponding env vars before
launch:

```text
BAG_OBJECT_TOPIC=/object_nodes_list
BAG_ROOM_TOPIC=/room_nodes_list
BAG_ODOM_TOPIC=/aft_mapped_to_init
BAG_IMAGE_TOPIC=/camera/image
BAG_DETECTION_TOPIC=/detection_result
BAG_PATH_TOPIC=/path
```

### High-Level Runtime Test Commands

The high-level runtime node subscribes `/object_nodes_list`,
`/room_nodes_list`, `/aft_mapped_to_init`, `/camera/image`, and
`/detection_result`. The wrapper script sources ROS and the overlay, then adds
the repository root to `PYTHONPATH` for the shared STRIVE `real_robot` package.

Run tests in this order. Do not skip directly to `dry_run:=false`.

#### 1. Safe WAIT Smoke

This verifies launch, subscriptions, readiness gates, and JSONL logging. It
does not compile an instruction plan and never publishes `/way_point`.

```bash
cd /home/ubuntu/WorkSpace/project/Huawei\ Nav/Code/STRIVE
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  policy_mode:=wait \
  dry_run:=true \
  run_directory:=/tmp/strive_real_robot_runtime_wait
```

Inspect the decisions:

```bash
tail -n 20 /tmp/strive_real_robot_runtime_wait/runtime_decisions.jsonl
```

Expected result:

```text
intent.mode == "wait"
motion_goal == null
no /way_point publication
```

#### 2. Semantic Snapshot Dry-Run

This compiles the instruction into an `InstructionPlan`, adapts
`SemanticMapSnapshot` through `SemanticMapSnapshotPolicyContext`, and emits a
`NavigationIntent`. It still does not publish `/way_point`.

Use `instruction_plan_backend:=rules` for offline wiring tests. Use
`instruction_plan_backend:=llm` only when the LLM/VLM runtime is configured.

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  dataset_target:=book \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  dry_run:=true \
  enable_final_verifier:=false \
  run_directory:=/tmp/strive_real_robot_runtime_semantic_dry
```

Inspect:

```bash
tail -n 20 /tmp/strive_real_robot_runtime_semantic_dry/runtime_decisions.jsonl
```

Expected result after object snapshots arrive:

```text
intent.mode in {"go_to_object", "go_to_anchor", "wait"}
motion_goal is present only for go_to_* intents
navigation_status.metadata.dry_run == true for dispatched dry-run goals
no /way_point publication
```

#### 3. Semantic Dry-Run With Evidence Files

This is useful before enabling final verifier. It persists observation images
for replay/debug. It still does not publish `/way_point`, and it does not run
the final verifier unless a reached status is reported.

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  dataset_target:=book \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  dry_run:=true \
  enable_final_verifier:=false \
  persist_observation_images:=true \
  observation_image_directory:=/tmp/strive_real_robot_runtime_semantic_evidence/observations \
  run_directory:=/tmp/strive_real_robot_runtime_semantic_evidence
```

Inspect:

```bash
tail -n 20 /tmp/strive_real_robot_runtime_semantic_evidence/runtime_decisions.jsonl
find /tmp/strive_real_robot_runtime_semantic_evidence -maxdepth 3 -type f | sort | head -50
```

#### 4. Final Verifier Dry-Run

Enable this only after semantic dry-run produces reasonable target intents and
the evidence cache is available. Verifier `accept` is the only semantic STOP
authority. `dry_run_status:=reached` simulates the lower planner reporting
`NavigationStatus.REACHED`, so `ViewpointEvidenceLoop.verify_reached(...)` runs
without publishing `/way_point`.

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  dataset_target:=book \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=llm \
  dry_run:=true \
  dry_run_status:=reached \
  enable_final_verifier:=true \
  evidence_mode:=auto \
  persist_observation_images:=true \
  observation_image_directory:=/tmp/strive_real_robot_runtime_verifier/observations \
  run_directory:=/tmp/strive_real_robot_runtime_verifier
```

Expected result:

```text
NavigationStatus.REACHED -> ViewpointEvidenceLoop.verify_reached(...)
verifier_decision appears in runtime_decisions.jsonl
verifier accept -> intent.mode == "stop"
dry_run still prevents /way_point publication
```

#### 5. Publish Waypoints To Lower Planner

Run this only after `/path`, `/aft_mapped_to_init`, and the local planner are
healthy, and only when the robot safety boundary is already handled outside
STRIVE. This publishes `/way_point`; STRIVE still never publishes `/cmd_vel`.
The launch will reject this mode unless `lower_controller_enabled:=true` is
set, or unless `waypoint_topic` is changed to the configured test topic.

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  dataset_target:=book \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  dry_run:=false \
  lower_controller_enabled:=true \
  enable_final_verifier:=false \
  waypoint_topic:=/way_point \
  hold_topic:=/platform/safe_hold \
  cancel_topic:=/local_planner/cancel \
  path_topic:=/path \
  odom_topic:=/aft_mapped_to_init \
  run_directory:=/tmp/strive_real_robot_runtime_waypoint
```

Inspect from another terminal:

```bash
ros2 topic echo /way_point --once
tail -n 20 /tmp/strive_real_robot_runtime_waypoint/runtime_decisions.jsonl
```

Only after the waypoint path is verified should `enable_final_verifier:=true`
be used with `dry_run:=false`.

#### Useful Runtime Parameters

```text
policy_mode:=wait | first_object_smoke | semantic_snapshot
instruction:=...
dataset_target:=...
instruction_plan_backend:=rules | llm
vlm:=cognav
dry_run:=true | false
dry_run_status:=idle | queued | running | reached | blocked | timeout | preempted | failed
enable_final_verifier:=false | true
evidence_mode:=auto | full_image | bbox_crop
run_directory:=/tmp/strive_real_robot_runtime
lower_controller_enabled:=false | true
waypoint_topic:=/way_point
test_waypoint_topic:=/strive/test_way_point
hold_topic:=
cancel_topic:=
emergency_stop_topic:=
allow_emergency_stop_publish:=false | true
```

Safety defaults:

```text
dry_run:=true never publishes /way_point.
dry_run:=false requires lower_controller_enabled:=true unless waypoint_topic is the test topic.
emergency_stop_topic is never published unless allow_emergency_stop_publish:=true.
Any /cmd_vel or */cmd_vel publish topic is rejected before publishers are created.
```

The node also keeps a lightweight observation cache for evidence acquisition.
By default image refs stay as ROS URI strings and no camera bytes are written:

```text
detection_topic:=/detection_result
depth_topic:=
pointcloud_topic:=
persist_observation_images:=false
observation_image_directory:=
```

Set `persist_observation_images:=true` only when replay/debug evidence files are
needed. The contract still records only `image_ref` and sidecar metadata paths.

When `dry_run:=false`, the node injects `RosNavigationStatusProvider` into the
waypoint controller. It reads `/aft_mapped_to_init` and `/path`, plus an
optional string `planner_status_topic`, then writes distance, elapsed time,
path length, and progress samples into `NavigationStatus.metadata`.

Useful launch parameters:

```text
path_topic:=/path
planner_status_topic:=
xy_goal_tolerance_m:=0.35
z_goal_tolerance_m:=1.0
navigation_timeout_s:=60.0
no_progress_timeout_s:=12.0
min_progress_delta_m:=0.05
path_stale_timeout_s:=5.0
```

The same stale timeout is used for `planner_status_topic`, so an old
`blocked`/`timeout` message does not permanently affect later goals.

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

Docker runtime envs used by `docker_en.sh` include:

```text
START_STRIVE_RUNTIME
STRIVE_INSTRUCTION
STRIVE_DATASET_TARGET
STRIVE_POLICY_MODE
STRIVE_INSTRUCTION_PLAN_BACKEND
STRIVE_VLM
STRIVE_PRIOR_MAP_PATH
STRIVE_OBJECT_TOPIC / STRIVE_ROOM_TOPIC / STRIVE_ODOM_TOPIC / STRIVE_IMAGE_TOPIC
STRIVE_WAYPOINT_TOPIC / STRIVE_HOLD_TOPIC / STRIVE_CANCEL_TOPIC
LLM_PROVIDER / LLM_MODEL / LLM_API_BASE_URL / ARK_API_KEY / GEMINI_API_KEY
SYSNAV_DETECTOR_MODEL_PATH / SYSNAV_SAM2_CHECKPOINT / SYSNAV_CLIP_VIT_B32_PATH
```

For code-only transfer, use:

```bash
bash scripts/export_code_only.sh
```

The export script excludes `.git`, local env files, model weights, rosbag
files, runtime output, caches, and `real_robot/ros2_ws/{build,install,log}`.

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
