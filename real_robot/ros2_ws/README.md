# VLN 实物模式 ROS2 Workspace

本 workspace 保存当前 VLN 实物模式所需的 SysNav ROS2 组件。它负责真实机器人上的
检测、语义建图、局部规划、运动 action 和安全速度链路；VLN 高层 Python contract
与运行时位于上一级 `real_robot/` 目录。

完整的数据流、控制流和平台适配边界见：

- [`docs/real_robot_framework.md`](../../docs/real_robot_framework.md)
- [`docs/lvlm_server_deployment.md`](../../docs/lvlm_server_deployment.md)

实物模式采用单一 ROS2 运行环境和远程 LVLM 服务：

```text
机器人 ROS2 workspace
  SysNav detector / semantic mapping
  VLN instruction runtime
  local planner / path follower / SafetyVelocityMux

独立 LVLM 服务器或商业 API
  <- OpenAI-compatible HTTPS ->
  机器人 VLN runtime
```

## 1. Workspace 结构

```text
src/tare_planner/
  SysNav 兼容消息包。
  提供 DetectionResult、ObjectNode、ObjectNodeList、RoomNode、RoomNodeList、
  TargetObjectInstruction 等消息类型。

src/semantic_mapping/
  迁移的 SysNav detector_node 和 semantic_mapping_node。
  detector_node 订阅 RGB，发布 /detection_result；
  semantic_mapping_node 融合检测、点云和位姿，发布 /object_nodes_list。

src/strive_sysnav_bringup/
  VLN 实物模式 launch 和高层 runtime 节点。
  负责启动感知建图、instruction runtime，以及可选的下层控制链。

src/terrain_analysis/
src/local_planner/
  迁移的 SysNav 局部规划组件。
  localPlanner 消费点云、位姿和 /way_point，输出 /path；
  pathFollower 跟踪路径并发布 /cmd_vel/autonomy。

src/strive_motion_msgs/
src/strive_sysnav_motion/
  ExecuteWaypoint action、运动生命周期反馈和安全速度 mux。
```

`build/`、`install/` 和 `log/` 是 colcon 生成的本地产物，不是源码依赖，也不应作为
迁移代码提交。迁移时只需要 `src/` 和对应的部署配置。

## 2. 编译

从仓库根目录执行：

```bash
cd "/home/ubuntu/WorkSpace/project/Huawei Nav/Code/STRIVE"
bash scripts/build_real_robot_ros_ws.sh
```

脚本会编译：

```text
tare_planner
terrain_analysis
local_planner
semantic_mapping
strive_motion_msgs
strive_sysnav_motion
strive_sysnav_bringup
```

编译完成后，在同一个 shell 中加载 workspace：

```bash
source real_robot/ros2_ws/install/setup.bash
```

目标设备需要安装与 ROS2 发行版、CUDA、PyTorch 和 Jetson/amd64 架构匹配的依赖。
编译成功不代表检测器权重、传感器、DDS 网络或底盘已经验收。

## 3. 运行时模型与检测器资产

检测器和 SAM2 权重属于部署资产，不提交到 Git。启动前显式配置：

```bash
export SYSNAV_DETECTOR_MODEL_TYPE=yoloe
export SYSNAV_DETECTOR_MODEL_PATH=/path/to/yoloe-model.engine
export SYSNAV_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_base_plus.pt
```

某些 YOLOE `.pt` 回退模型还需要文本编码器资产：

```bash
export SYSNAV_CLIP_VIT_B32_PATH=/path/to/ViT-B-32.pt
export SYSNAV_MOBILECLIP_BLT_TS_PATH=/path/to/mobileclip_blt.ts
```

当前 detector 词表由 SysNav 的 `objects.yaml` 提供。`DetectorVocabularyAdapter` 只
记录 detector 的 label、prompt 和 provenance，不在 ROS 层硬编码自然语言别名。

## 4. 感知与语义建图链路

第一版 SysNav 复用链路如下：

```text
/camera/image
  -> detection_node
  -> /detection_result
  -> semantic_mapping_node
  -> /object_nodes_list
  -> RosObjectNodeAdapter
  -> SemanticMapSnapshot
```

semantic mapping 的输入由 launch 参数映射：

```text
相机：      /camera/image
注册点云：  /cloud_registered 或 /registered_scan
位姿：      /aft_mapped_to_init 或 /state_estimation
视角：      /viewpoint_rep_header，可选
```

VLN runtime 主要消费：

```text
/object_nodes_list       tare_planner/ObjectNodeList
/room_nodes_list         tare_planner/RoomNodeList，可选
/aft_mapped_to_init      nav_msgs/Odometry
/camera/image            sensor_msgs/Image
/detection_result        tare_planner/DetectionResult
/path                    nav_msgs/Path，可选
/local_planner/status    std_msgs/String，可选
```

当前迁移的 `semantic_mapping_node` 主要发布 `/object_nodes_list`；`/room_nodes_list`
是 runtime adapter 支持的可选输入，不能假定每个 SysNav 版本都会发布。

## 5. 启动感知建图

### 5.1 使用已存在的相机和定位节点

```bash
cd "/home/ubuntu/WorkSpace/project/Huawei Nav/Code/STRIVE"

bash scripts/run_sysnav_detection_mapping.sh \
  platform:=mecanum \
  use_sim_time:=false \
  camera_topic:=/camera/image \
  cloud_topic:=/cloud_registered \
  odom_topic:=/aft_mapped_to_init
```

脚本只启动检测和语义建图，默认不启动 VLN instruction runtime，也不接通真实底盘。

### 5.2 同时启动 USB 相机

如果机器人没有其他节点发布 `/camera/image`，可以由 bringup 启动 `usb_cam`：

```bash
bash docker/run_real_robot_sysnav_stack.sh \
  platform:=mecanum \
  start_usb_cam:=true \
  usb_video_device:=/dev/video0 \
  camera_topic:=/camera/image \
  cloud_topic:=/cloud_registered \
  odom_topic:=/aft_mapped_to_init
```

Theta 全景相机和 RealSense 都通过 `/camera/image` 接入。RealSense 的对齐深度可通过
`depth_topic` 额外传给 observation cache；相机型号、内参和外参由平台 profile 提供。

### 5.3 预期 topic

```text
感知输入：
  /camera/image
  /registered_scan 或 /cloud_registered
  /state_estimation 或 /aft_mapped_to_init
  /viewpoint_rep_header，可选

感知输出：
  /detection_result
  /object_nodes_list
  /annotated_image_detection
  /annotated_image
  /cloud_image

运动与安全：
  /way_point
  /path
  /cmd_vel/autonomy
  /cmd_vel
  /platform/safety_state
  /platform/safe_hold
  /local_planner/cancel
```

topic 名称必须以机器人 profile 和 launch 参数为准。启动前应使用 `ros2 topic list`、
`ros2 topic info` 和消息频率检查确认真实输入，而不是只检查 topic 是否存在。

## 6. 启动 VLN 高层 runtime

高层节点是 `strive_instruction_runtime`。它订阅 SysNav object/room snapshot、RGB、
位姿和运动反馈，调用现有 instruction/concept/verifier 模块，输出
`NavigationIntent -> MotionGoal`。

### 6.1 安全 WAIT smoke

该模式只验证启动、订阅、readiness gate 和 JSONL 记录，不编译指令计划，也不发布
`/way_point`：

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  policy_mode:=wait \
  dry_run:=true \
  run_directory:=/tmp/vln_real_robot_wait
```

检查运行记录：

```bash
tail -n 20 /tmp/vln_real_robot_wait/runtime_decisions.jsonl
```

预期结果：

```text
intent.mode == "wait"
motion_goal == null
没有 /way_point 发布
```

### 6.2 语义 snapshot dry-run

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  dataset_target:=book \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  dry_run:=true \
  enable_final_verifier:=false \
  run_directory:=/tmp/vln_real_robot_semantic_dry
```

该模式将输入编译为 `InstructionPlan`，通过
`SemanticMapSnapshotIntentAdapter` 输出高层意图，但 dry-run 不会把目标发送到真实
`/way_point`。

需要远程 LVLM 时，将 `instruction_plan_backend` 改为 `llm`，并按
[`docs/lvlm_server_deployment.md`](../../docs/lvlm_server_deployment.md) 配置商业 API
或公网 HTTPS 自部署服务。

### 6.3 证据与 final verifier dry-run

确认 snapshot、目标候选和 evidence cache 正常后，再开启 final verifier：

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book on a shelf" \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=llm \
  dry_run:=true \
  dry_run_status:=reached \
  enable_final_verifier:=true \
  persist_observation_images:=true \
  observation_image_directory:=/tmp/vln_real_robot_verifier/observations \
  run_directory:=/tmp/vln_real_robot_verifier
```

这里 `dry_run_status:=reached` 只是模拟下层返回 `NavigationStatus.REACHED`，不代表
真实底盘到达。预期控制流是：

```text
REACHED
  -> ViewpointEvidenceLoop.verify_reached()
  -> ViewEvidence
  -> final verifier
  -> accept 才能产生 stop intent
```

### 6.4 测试 waypoint

如果需要测试 ROS waypoint 交接，但尚未批准真实底盘，可以将输出指向独立测试 topic：

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  dry_run:=false \
  lower_controller_enabled:=false \
  waypoint_topic:=/strive/test_way_point \
  run_directory:=/tmp/vln_real_robot_test_waypoint
```

测试 topic 不得连接真实速度控制器。

## 7. LVLM 接入

实物机器人只运行 VLN 客户端，不加载 LVLM 权重。LVLM 可以是商业 API，也可以是独立
GPU 服务器上的自部署模型：

```text
机器人 VLN runtime
  -- HTTPS /v1/chat/completions -->
公网 DNS + TLS + 认证反向代理
  --> GPU 模型服务器
```

机器人端配置示例：

```bash
export LLM_PROVIDER=self_hosted
export STRIVE_LLM_CLIENT=self_hosted
export STRIVE_VLM=self_hosted
export VLN_LVLM_BASE_URL=https://<public-domain>/v1
export VLN_LVLM_API_KEY=<same-token-as-server>
export VLN_LVLM_MODEL=<served-model-name>
export VLN_LVLM_TIMEOUT_S=45
export VLN_LVLM_TRANSPORT_RETRIES=2
export VLN_LVLM_PARSE_RETRIES=1
```

模型服务器不需要加入机器人 ROS 网络，也不访问 `/cmd_vel`、`/way_point` 或传感器
topic。网络、解析或 schema 失败时，runtime 必须产生保守的 wait/replan 结果，不能
产生 final STOP。

## 8. 运动控制边界

VLN 不直接发布离散 Habitat action，也不直接发布 `/cmd_vel`。运动链路为：

```text
NavigationIntent
  -> MotionGoal
  -> RosWaypointController 或 RosActionMotionController
  -> /way_point / ExecuteWaypoint
  -> SysNav localPlanner
  -> /path
  -> pathFollower
  -> /cmd_vel/autonomy
  -> SafetyVelocityMux
  -> /cmd_vel
```

### 8.1 Waypoint backend

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  motion_backend:=waypoint \
  dry_run:=false \
  lower_controller_enabled:=true \
  waypoint_topic:=/way_point \
  hold_topic:=/platform/safe_hold \
  cancel_topic:=/local_planner/cancel \
  odom_topic:=/aft_mapped_to_init \
  path_topic:=/path \
  controller_contract_file:=/workspace/STRIVE/real_robot/control/<robot>_controller_contract.yaml
```

该模式由 `RosWaypointController` 发布 `geometry_msgs/PointStamped`，状态由
`RosNavigationStatusProvider` 根据 odom、path、超时和进度推断。

### 8.2 Action backend

Action backend 用于需要 goal id、feedback、cancel 和 result 的完整生命周期：

```text
RosActionMotionController
  -> /strive/execute_waypoint [ExecuteWaypoint]
  -> SysNavMotionServer
  -> /way_point
  -> localPlanner / pathFollower
```

启动 action server 的示例：

```bash
ros2 launch strive_sysnav_motion sysnav_motion_server.launch.py \
  waypoint_topic:=/way_point \
  odom_topic:=/aft_mapped_to_init \
  path_topic:=/path \
  hold_topic:=/platform/safe_hold \
  cancel_topic:=/local_planner/cancel \
  controller_contract_file:=/workspace/STRIVE/real_robot/control/<robot>_controller_contract.yaml \
  require_controller_contract:=true
```

同一运行图中只能有一个 `/way_point` owner。Action server 返回的 `REACHED`、
`BLOCKED`、`TIMEOUT`、`PREEMPTED`、`SAFETY_STOP`、人工接管和定位丢失，只描述运动
尝试结果，不表示自然语言任务成功。

## 9. 安全开关

真实运动交接必须满足：

```text
dry_run=false
lower_controller_enabled=true
controller_contract_file 已配置
controller_contract.approval_status == approved
waypoint frame、反馈、速度限制、watchdog、急停和人工接管字段通过校验
```

默认行为：

```text
dry_run=true
  只写 runtime_decisions.jsonl，不发布 /way_point。

dry_run=false 且 lower_controller_enabled=false
  只允许显式配置的 test waypoint，不连接真实下层控制器。

任何模式
  VLN 都不得创建 /cmd_vel 或 */cmd_vel publisher。
```

安全优先级为：

```text
急停 / 人工接管
  > 定位丢失 / 传感器过期
  > 控制器故障 / 命令 watchdog
  > blocked / timeout / no feasible path
  > VLN 语义重规划
```

## 10. Bag replay 与离线验收

如果 rosbag 已经包含 `/object_nodes_list`、`/room_nodes_list`、`/aft_mapped_to_init`
和 `/camera/image`，可以只回放这些 VLN-facing topic，不启动 detector/mapping：

```bash
bash scripts/run_real_robot_bag_replay.sh /path/to/recorded_bag \
  instruction:="find a book" \
  dataset_target:=book \
  policy_mode:=semantic_snapshot \
  instruction_plan_backend:=rules \
  dry_run:=true \
  run_directory:=/tmp/vln_real_robot_bag_replay
```

离线验收入口：

```bash
bash scripts/check_real_robot_acceptance.sh
```

离线测试可以覆盖：

```text
fake ROS object/room message -> SemanticMapSnapshot
semantic dry-run -> NavigationIntent，且没有 waypoint publisher
fake motion controller -> RUNNING / REACHED
REACHED -> ViewEvidence + verifier_payload
verifier accept -> stop intent
```

这些测试不等于真实相机、LiDAR、局部规划器、底盘或急停验收。

## 11. Orin / Jetson 运行入口

目标设备使用仓库提供的单一实物 Docker 入口：

```bash
SUDO_STDIN_PASSWORD=1 ./docker_en.sh start
SUDO_STDIN_PASSWORD=1 ./docker_en.sh enter
SUDO_STDIN_PASSWORD=1 ./docker_en.sh status
```

启动前应检查：

```text
ROS_DOMAIN_ID、RMW/DDS transport
相机、点云和 odom topic 是否有实际消息
frame_id 和时间戳是否正确
检测器、SAM2 和文本编码器权重是否存在
LVLM endpoint、模型名和 API key 是否可用
controller contract 是否批准
```

容器可以承载 SysNav detector/mapping 和 VLN runtime；大模型权重不放入机器人镜像，
LVLM 通过公网 HTTPS 访问。代码迁移可使用：

```bash
bash scripts/export_code_only.sh
```

该脚本排除 `.git`、私有环境文件、模型权重、rosbag、runtime output、缓存以及
`real_robot/ros2_ws/{build,install,log}`。

## 12. 当前不由本 workspace 保证的事项

- 真实设备的相机内参、LiDAR-camera 外参和时间同步；
- Point-LIO 或其他 SLAM 的定位稳定性；
- 真实 `/way_point` 到底盘的接收、路径跟踪和到达判定；
- 速度、角速度、加速度限制的实测；
- 急停、人工接管、通信中断和底盘故障；
- 特定机器人上的 ObjectNav 成功率。

建议按以下顺序推进：

```text
1. ROS graph 和传感器消息 smoke
2. detector / semantic mapping 输出检查
3. VLN WAIT 和 semantic dry-run
4. test waypoint 与 fake/HIL status
5. controller contract 审批
6. 低速、短距离、人工监控下的真实 waypoint
7. final verifier 和完整任务闭环
```
