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

## 5. SemanticSnapshotInstructionPolicy

- [ ] 新增 `SemanticSnapshotInstructionPolicy`，消费 `SemanticMapSnapshot` 和 `InstructionPlan`。
- [ ] 将 `ObjectNodeSnapshot` 转成 concept matcher 可消费的 lightweight object payload。
- [ ] 支持 terminal target 选择，输出 `GO_TO_OBJECT` 或 `VERIFY_TARGET` intent。
- [ ] 支持 anchor-first relation search，anchor 只能输出 `GO_TO_ANCHOR`，不能触发成功。
- [ ] 接入 `ConstraintEvaluator` 的关系/属性/房间约束证据。
- [ ] 接入 final verifier：只有 verifier `accept` 且物理合同满足时输出 `STOP`。
- [ ] 接入 view-control：`need_better_view` 时输出 `IMPROVE_VIEW` 或 `VERIFY_RELATION`。
- [ ] 增加 fake snapshot 测试：anchor 不可 stop、hard rejected instance 不屏蔽同类其它实例。

## 6. Safety Boundary

- [ ] 为 `RosWaypointController.hold()` 接入平台安全 hold/stop topic。
- [ ] 为 `cancel()` 接入 lower planner cancel 或 stop 机制。
- [ ] 增加 emergency stop 参数，默认不自动覆盖底层安全系统。
- [ ] 确认 STRIVE 永远不直接发布 `/cmd_vel`。
- [ ] 记录 lower controller 是否启用；未启用时只能 dry-run 或发布到测试 topic。

## 7. Deployment And Scripts

- [x] 提供 `docker_en.sh` 作为 Orin 宿主侧一键入口，默认镜像 `huawei-nav-real:orin`。
- [x] 提供 `scripts/start_real_robot_framework.sh`，启动前检查 `/cloud_registered` 和 `/aft_mapped_to_init`。
- [x] 提供 `scripts/start_orin_lio_for_strive.sh`，覆盖 Point-LIO `publish.scan_publish_en:=true`。
- [x] 提供 `scripts/smoke_real_robot_orin.sh`，做 LIO、相机、容器 DDS、CUDA/ML、detector 初始化检查。
- [ ] 更新 `scripts/run_sysnav_detection_mapping.sh`，明确高层 node 是否一起启动。
- [x] 新增 `scripts/run_real_robot_instruction_runtime.sh`，只启动 STRIVE 高层 runtime。
- [ ] 增加 bag replay 入口，读取录制的 object/room/odom/image topic。
- [ ] 更新 Docker run 环境变量：instruction、topic remap、model paths、LLM provider、prior map path。
- [ ] 确保权重、bag、缓存和 build/install/log 产物不进入代码导出。

## 8. Smoke And Acceptance

- [ ] Offline fake message unit tests 全部通过。
- [x] Orin bounded smoke 脚本已具备，不发布 `/way_point` 或 `/cmd_vel`。
- [x] Orin LIO topic gate 已记录：`/cloud_registered`、`/aft_mapped_to_init`。
- [ ] Bag replay：能够从 `/object_nodes_list` 生成 `SemanticMapSnapshot`。
- [ ] Dry-run：能够生成 `NavigationIntent`，不发布 waypoint。
- [ ] Waypoint smoke：发布一个低风险 `/way_point`，能收到 `RUNNING/REACHED` 状态。
- [ ] Evidence smoke：到达后能生成 `ViewEvidence`，并保存 verifier payload。
- [ ] End-to-end smoke：单目标指令从 snapshot 到 waypoint 到 final verifier 闭环完成。

## 9. 文档同步

- [ ] 每次新增 runtime 接口后更新 `docs/real_robot_deployment.md`。
- [ ] 若实物模式引入先验地图，更新 `docs/prior_map_mode.md` 的实物接入章节。
- [ ] 将真实 topic remap、硬件 smoke 结果和模型路径要求写入部署文档。
