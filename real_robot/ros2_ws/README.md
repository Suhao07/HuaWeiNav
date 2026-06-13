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
