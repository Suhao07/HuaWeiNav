# STRIVE 实物部署待做 Checklist

本文档跟踪实物兼容部署分支的剩余工作。原则是先打通可测试闭环，再接入真实硬件；
所有新增代码都应保持 `real_robot/contracts.py` 的平台无关边界。

## 1. 已有基础

- [x] 定义 `RealObservation`、`DetectionFrame`、`SemanticMapSnapshot`、`NavigationIntent`、`MotionGoal`、`ViewEvidence` 等实物 contract。
- [x] 实现 SysNav detector vocabulary provenance：`real_robot/detector_vocabulary.py`。
- [x] 实现 `/detection_result`、`/object_nodes_list`、`/room_nodes_list` 到 STRIVE snapshot 的 adapter。
- [x] 实现 `RosWaypointController`，将 `MotionGoal` 发布到 `/way_point`。
- [x] 实现 `SysNavSemanticMapBridge`、`SysNavInstructionRuntime` 和 `ViewpointEvidenceLoop` 骨架。
- [x] Vendor 第一版 SysNav ROS2 overlay：`tare_planner` messages、`semantic_mapping`、`strive_sysnav_bringup`。
- [x] 覆盖 contract、adapter、runtime skeleton 的轻量单元测试。
- [x] 增加 Orin 单容器入口 `docker_en.sh`，支持 start / enter / stop / restart / logs / smoke / start-lio。
- [x] 增加容器内 guarded entrypoint：`scripts/start_real_robot_framework.sh`。
- [x] 增加宿主侧 Livox + Point-LIO 启动 helper：`scripts/start_orin_lio_for_strive.sh`。
- [x] 增加 Orin bounded smoke：`scripts/smoke_real_robot_orin.sh`。
- [x] 默认阻塞底层控制器：`BLOCK_LOWER_CONTROLLER=1`、`ENABLE_LOWER_CONTROLLER=0`，不主动发布 `/cmd_vel`。

## 2. Live ROS Runtime

- [x] 新增实物高层 ROS2 node，负责订阅 object/room/pose/image topic，调用 `SysNavInstructionRuntime.step()`。
- [x] 为高层 node 增加参数：instruction、topic names、world frame、policy mode、prior map path、run directory。
- [x] 让高层 node 输出 `RuntimeDecision` JSONL，包含 snapshot size、intent、motion goal、status 和 reason。
- [x] 增加 WAIT 行为：未收到 object snapshot、pose 或 image 时不发布 waypoint。
- [x] 增加 dry-run 模式：只打印 `NavigationIntent`，不发布 `/way_point`。

当前入口：

```bash
bash scripts/run_real_robot_instruction_runtime.sh \
  instruction:="find a book" \
  dry_run:=true \
  policy_mode:=wait \
  run_directory:=/tmp/strive_real_robot_runtime
```

`policy_mode=wait` 是默认安全模式；`policy_mode=first_object_smoke` 只用于验证
`SemanticMapSnapshot -> NavigationIntent` 链路，不做自然语言语义判断。

## 3. NavigationStatus Provider

- [x] 新增 `RosNavigationStatusProvider`。
- [x] 订阅 odometry topic，计算当前 pose 与 active `MotionGoal.goal_pose` 的距离。
- [x] 订阅 path/local planner 状态，记录是否存在可执行路径。
- [x] 实现 timeout 判断。
- [x] 实现 no-progress 判断：连续一段时间距离无下降时返回 `BLOCKED` 或 `TIMEOUT`。
- [x] 明确 `REACHED` 阈值：xy tolerance、z tolerance、heading 是否参与判断。
- [x] 将 `NavigationStatus.metadata` 写入 distance、elapsed、path length、progress samples。
- [x] 增加 fake odom/path 单元测试，覆盖 `RUNNING/REACHED/BLOCKED/TIMEOUT/PREEMPTED`。

当前默认阈值：

```text
xy_goal_tolerance_m=0.35
z_goal_tolerance_m=1.0
navigation_timeout_s=60.0
no_progress_timeout_s=12.0
min_progress_delta_m=0.05
path_stale_timeout_s=5.0
heading_tolerance_rad=None
```

heading 暂不参与 `REACHED` 判断，因为当前 SysNav `/way_point` 接口只消费
`geometry_msgs/PointStamped`，不携带目标朝向。z 阈值默认较宽，用于兼容
SysNav waypoint z offset 与真实 odometry z 近似为 0 的情况。

## 4. Observation Cache And Evidence

- [x] 新增 `RosObservationCache`，缓存最新 RGB、pose、detection frame 和可选 point cloud/depth 引用。
- [x] 实现图像落盘或内存引用策略，contract 中只保存 `image_ref`。
- [x] 新增 `ObjectCropEvidenceProvider`，根据 object uid、track id、bbox 或 image path 构造 `ViewEvidence`。
- [x] 支持 full image evidence 和 bbox crop evidence 两种模式。
- [x] 为 evidence 增加 view quality facts：bbox area、center score、border margin、source timestamp。
- [x] 确保 `ViewpointEvidenceLoop` 只有在 `NavigationStatus.REACHED` 后调用 provider。
- [x] 增加测试：blocked/timeout 不调用 final verifier，不生成伪 evidence。

当前实现：

```text
RosObservationCache
  pose + RGB image + DetectionFrame + optional depth/pointcloud refs
  -> RealObservation(camera.image_ref, depth_ref, pointcloud_ref)

ObjectCropEvidenceProvider
  target_object_uid / anchor_object_uid
  -> object bbox2d 或 detection track bbox
  -> ViewEvidence(full_image 或 bbox_crop)
```

默认 `persist_observation_images=false`，图像引用使用 `ros://...` URI，避免真机高频
相机数据持续写盘。需要复盘 evidence 时可设置：

```text
persist_observation_images=true
observation_image_directory:=/tmp/strive_real_robot_runtime/observations
```

## 5. SemanticMapSnapshotPolicyContext

- [x] 废弃实物侧重复语义策略设计。
- [x] 新增 `planning/semantic_snapshot_context.py`，只做 `SemanticMapSnapshot -> mapper-like context` 适配。
- [x] 将 `ObjectNodeSnapshot` 包装成现有 `select_target_candidate()` 可消费的 object payload。
- [x] 复用现有 `InstructionObjectSearchPolicy`、`RuntimeConceptMatcher`、`VerificationLedger` 和 anchor-first 逻辑。
- [x] 保留 hard rejected instance 只屏蔽单个 uid 的语义，不屏蔽同类其它实例。
- [x] 增加 fake snapshot 测试：terminal target、anchor-first、instance-scoped hard reject。
- [x] 新增 `RealInstructionRuntimeState`，避免 live runtime 在 lower planner 仍执行同一 goal 时重复 dispatch。
- [x] 将真实 `InstructionPlan` provider 接入 live ROS node。
- [x] 将 context selection result 转成现有上层策略的 `NavigationIntent`，不要在 `real_robot` 重写导航状态机。
- [x] 在 `NavigationStatus.REACHED` 后接入 `ViewpointEvidenceLoop` / final verifier。

当前边界：

```text
SemanticMapSnapshot + InstructionPlan
  -> SemanticMapSnapshotPolicyContext
  -> planning.select_target_candidate(...)
  -> SemanticMapSnapshotIntentAdapter
  -> NavigationIntent
```

当前实现：

```text
policy_mode:=semantic_snapshot
  -> compile_instruction_plan(instruction, dataset_target, backend, vlm)
  -> StaticInstructionPlanProvider
  -> SemanticMapSnapshotIntentAdapter.decide(snapshot)
  -> GO_TO_OBJECT / GO_TO_ANCHOR / WAIT
  -> SysNavInstructionRuntime
  -> /way_point
  -> NavigationStatus.REACHED
  -> ObjectCropEvidenceProvider
  -> ViewpointEvidenceLoop.verify_reached(...)
  -> optional FinalInstructionVerifierAdapter
  -> verifier accept => STOP
```

安全默认：

```text
policy_mode:=wait
dry_run:=true
enable_final_verifier:=false
```

`enable_final_verifier:=false` 时仍会保留 reached evidence loop 接线，但不会产生
`STOP`；真机启用 final verifier 前应先用 `dry_run:=true` 检查
`runtime_decisions.jsonl`。

关键原则：

```text
real_robot owns ROS/SysNav adaptation, observation cache, status provider, waypoint bridge.
planning owns target/anchor/relation candidate selection.
instruction_adapter owns concept matching, constraints, ledgers, final verification state.
final verifier owns STOP authority.
```

也就是说，实物模式替换的是输入输出边界，不应该在 `real_robot` 目录再实现一份
terminal / anchor / relation / verifier 状态机。

## 6. Safety Boundary

- [x] 为 `RosWaypointController.hold()` 接入平台安全 hold/stop topic。
- [x] 为 `cancel()` 接入 lower planner cancel 或 stop 机制。
- [x] 增加 emergency stop 参数，默认不自动覆盖底层安全系统。
- [x] 确认 STRIVE 永远不直接发布 `/cmd_vel`。
- [x] 记录 lower controller 是否启用；未启用时只能 dry-run 或发布到测试 topic。

当前实现：

```text
RosWaypointController
  MotionGoal -> /way_point
  hold() -> optional std_msgs/Empty on hold_topic
  cancel() -> optional std_msgs/Empty on cancel_topic
           -> fallback std_msgs/Empty on hold_topic
  emergency_stop_topic 只有 allow_emergency_stop_publish=true 时才发布
```

安全边界：

```text
dry_run=true
  不创建 /way_point publisher，不发布 motion goal。

dry_run=false + lower_controller_enabled=false
  只允许 waypoint_topic == test_waypoint_topic。

dry_run=false + lower_controller_enabled=true
  允许发布 waypoint_topic，但仍禁止任何 /cmd_vel 或 */cmd_vel topic。
```

每条 `RuntimeDecision` JSONL 都记录：

```text
runtime_safety.lower_controller_enabled
runtime_safety.waypoint_topic
runtime_safety.hold_topic
runtime_safety.cancel_topic
runtime_safety.emergency_stop_topic
runtime_safety.allow_emergency_stop_publish
```

## 7. Deployment And Scripts

- [x] 提供 `docker_en.sh` 作为 Orin 宿主侧一键入口，默认镜像 `huawei-nav-real:orin`。
- [x] 提供 `scripts/start_real_robot_framework.sh`，启动前检查 `/cloud_registered` 和 `/aft_mapped_to_init`。
- [x] 提供 `scripts/start_orin_lio_for_strive.sh`，覆盖 Point-LIO `publish.scan_publish_en:=true`。
- [x] 提供 `scripts/smoke_real_robot_orin.sh`，做 LIO、相机、容器 DDS、CUDA/ML、detector 初始化检查。
- [x] 更新 `scripts/run_sysnav_detection_mapping.sh`，明确高层 node 是否一起启动。
- [x] 新增 `scripts/run_real_robot_instruction_runtime.sh`，只启动 STRIVE 高层 runtime。
- [x] 增加 bag replay 入口，读取录制的 object/room/odom/image topic。
- [x] 更新 Docker run 环境变量：instruction、topic remap、model paths、LLM provider、prior map path。
- [x] 确保权重、bag、缓存和 build/install/log 产物不进入代码导出。

当前实现：

```text
scripts/run_sysnav_detection_mapping.sh
  默认只启动 detector + semantic_mapping。
  START_STRIVE_RUNTIME=1 时才并行启动 strive_instruction_runtime。

scripts/run_real_robot_instruction_runtime.sh
  只启动 STRIVE 高层 runtime。

scripts/run_real_robot_bag_replay.sh BAG_PATH
  ros2 bag play --clock
  -> strive_instruction_runtime(use_sim_time=true, dry_run=true)
  -> 读取 object/room/odom/image/detection/path topic。

scripts/export_code_only.sh [DEST_DIR]
  导出代码，不包含 .git、权重、bag、缓存、ROS build/install/log、runtime output。
```

Docker/env 边界：

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

## 8. Smoke And Acceptance

- [x] Offline fake message unit tests 全部通过。
- [x] Orin bounded smoke 脚本已具备，不发布 `/way_point` 或 `/cmd_vel`。
- [x] Orin LIO topic gate 已记录：`/cloud_registered`、`/aft_mapped_to_init`。
- [x] Bag replay：能够从 `/object_nodes_list` 生成 `SemanticMapSnapshot`。
- [x] Dry-run：能够生成 `NavigationIntent`，不发布 waypoint。
- [x] Waypoint smoke：发布一个低风险 `/way_point`，能收到 `RUNNING/REACHED` 状态。
- [x] Evidence smoke：到达后能生成 `ViewEvidence`，并保存 verifier payload。
- [x] End-to-end smoke：单目标指令从 snapshot 到 waypoint 到 final verifier 闭环完成。

当前本地验收入口：

```bash
bash scripts/check_real_robot_acceptance.sh
```

当前本地结果：

```text
54 passed
```

覆盖范围：

```text
fake /object_nodes_list + /room_nodes_list -> SemanticMapSnapshot
SemanticMapSnapshot + InstructionPlan -> NavigationIntent
DryRunMotionController -> 不创建 /way_point publisher
RosWaypointController + fake odom -> RUNNING / REACHED
REACHED -> ObjectCropEvidenceProvider -> ViewEvidence + verifier_payload
snapshot -> waypoint -> reached -> final verifier accept -> STOP
```

注意：以上是离线 fake message acceptance，不代表 2026-06-25 已重新连接 Orin
执行真实 `/way_point` 或底盘闭环。真机 smoke 仍需在 Orin 上按部署文档复跑。

## 9. 文档同步

- [x] 每次新增 runtime 接口后更新 `docs/real_robot_deployment.md`。
- [x] 若实物模式引入先验地图，更新 `docs/prior_map_mode.md` 的实物接入章节。
- [x] 将真实 topic remap、硬件 smoke 结果和模型路径要求写入部署文档。
