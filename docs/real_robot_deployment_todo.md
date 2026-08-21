# VLN 实物部署状态与 TODO

> 更新日期：2026-08-21
> 代码基线：`main` 与 `realworld` 已在 `24f9bb6` 汇合
> 目标平台：Orin-26、Intel RealSense D435i、Livox MID-360
> 当前安全状态：感知与影子控制验证；真实运动未批准

本文是实物部署的单一进度清单。设计接口见
[`real_robot_framework.md`](real_robot_framework.md)，实测证据保留在对应日期的
`real_robot_*_evidence_*.md` 文档中。

勾选规则：只有代码、命令输出或真机记录能够直接证明的事项才标记为完成。单元测试、
bag replay、HIL 和影子 topic 不能替代真实底盘验收。

## 1. 当前结论

当前已打通两条无运动链路：

```text
D435i RGB + MID-360/Point-LIO
  -> SysNav detector
  -> semantic mapping
  -> /huawei_vln/d435i_object_nodes_list

VLN /way_point (PointStamped, world frame)
  -> WaypointFormatAdapter
  -> /strive/test_waypoint_array (Float32MultiArray, ego frame)
```

第一条链路已经产生真实对象节点，第二条链路已经在影子 topic 上完成坐标与消息转换。
这些结果证明接口和数据流可运行，但不能证明 RGB--LiDAR 标定达到任务精度，也不能证明
底盘能够安全执行 waypoint。

当前完整闭环仍缺三项硬条件：

1. RGB--LiDAR 时间偏移与 held-out 投影误差通过验收；
2. 机器人下层控制器 contract 获得所有者确认并完成急停、反馈和限速测试；
3. 在真实对象节点和真实 RGB 上完成 `InstructionPlan -> MotionGoal -> REACHED -> final verifier` 的受控闭环。

## 2. 已完成与证据边界

| 模块 | 当前状态 | 可证明内容 | 不能据此声称 |
|---|---|---|---|
| 平台无关 contract | 已完成 | `SemanticMapSnapshot`、`NavigationIntent`、`MotionGoal`、`NavigationStatus`、`ViewEvidence` 已实现并有单元测试 | 真车已运动 |
| SysNav detector | 真机数据链已验证 | D435i 图像进入 YOLOE，检测 topic 有真实消息 | 所有目标类别均稳定检测 |
| SysNav semantic mapping | 数据层影子验证 | 2026-08-19 实测对象节点约 0.5 Hz，坐标为米级，含 `chair`、`desk`、`cabinet` | 标定已批准、对象定位精度达标 |
| LIO 输入 | 真机数据链已验证 | `/cloud_registered_body` 约 9.4--9.6 Hz，`/aft_mapped_to_init` 约 100 Hz | 长时定位无漂移 |
| D435i--MID-360 标定 | 部分完成 | 内参、外参和历史 bag 已导入；已有只读评估脚本 | `calibration_status=calibrated` |
| Instruction runtime | 软件闭环完成 | snapshot、active goal、REACHED 后取证和 verifier 生命周期有离线测试 | 已在 Orin 上运行真实自然语言任务 |
| 远程 LVLM | HTTP/schema 软件接口完成 | 商业 API、自部署 OpenAI-compatible 服务和 schema smoke 已实现 | 机器人网络上的 p95 延迟、超时恢复已验收 |
| Waypoint adapter | 真机影子验证 | `PointStamped -> Float32MultiArray`、world-to-ego、过期丢弃在影子 topic 验证 | 外部 `/waypoint` 已接收或底盘已执行 |
| SysNav 原生下层 | 代码迁移、离线/HIL 覆盖 | `localPlanner`、`pathFollower`、motion action、安全 mux 的软件链存在 | 它已成为 Orin 底盘的生产控制器 |
| 外部机器人控制器 | 只读接口审计 | 已观察 `/waypoint` 和 `/topoplan/reached_goal` 的历史接口 | blocked/timeout/cancel、急停和所有权已确认 |
| 真实先验地图 | 接口存在，实物未验收 | `PriorMapData`、alignment 和 runtime context 可接入 | 已有与真机坐标对齐的语义先验地图 |
| 房间节点 | adapter 已实现，producer 未确认 | `/room_nodes_list` 可选消费接口存在 | 当前 semantic mapping 会发布真实房间图 |

## 3. 与 SysNav 的边界

SysNav 参考实现的原生运动链为：

```text
/state_estimation + /registered_scan + /terrain_map + /way_point
  -> localPlanner
  -> /path
  -> pathFollower
  -> /cmd_vel
```

当前 Orin 上观察到的是另一条机器人自有链路：

```text
/waypoint (Float32MultiArray, 历史代码按 ego [x, y] 使用)
  -> 外部局部规划/PD controller
  -> AgileX bridge / mux
  -> 底盘
```

两条链路不能同时启用。迁移的 SysNav `pathFollower` 与外部控制器若同时发布速度，会造成
控制权不唯一。当前实施方案固定为：

- VLN 保持 `MotionGoal` 和 `/way_point` 为平台无关高层边界；
- Orin 第一阶段复用现有 `WaypointFormatAdapter` 接入机器人自有 `/waypoint`；
- 速度、局部避障、急停和人工接管继续由机器人自有控制链负责；
- 迁移的 SysNav 原生下层只保留为可选后端，在独立完成物理验收前不得与外部控制器并行启动。

这是当前证据下修改最少的路径。现阶段不新增通用 controller registry，也不再抽象一层
“万能底盘接口”；等控制器所有者给出真实消息和反馈合同后，只实现一个明确的 Orin
status adapter。

## 4. 执行 TODO

### P0：统一代码与部署基线

- [x] `main` 与 `realworld` 汇合到共同提交 `24f9bb6`。
- [x] 实物离线 acceptance 覆盖 contract、runtime、motion safety 和 SysNav adapter。
- [x] waypoint adapter 的 Python、ROS node、launch 和单元测试进入仓库。
- [ ] 在 Orin 工作区拉取共同基线，记录完整 commit，而不是只记录短哈希。
- [ ] 重建 `huawei-vln-realworld:orin-r36.5`，记录 image ID、基础镜像、资产 SHA256 和构建日志。
- [ ] 重新执行 D435i profile `check`、detector smoke 和 runtime WAIT smoke，产物写入新的 run 目录。
- [ ] 运行至少 30 分钟资源监控，记录温度、CPU/GPU、内存、检测频率、点云频率和对象节点频率。

验收出口：代码、镜像、模型和运行日志能对应到同一个 deployment manifest，且不会启动
waypoint adapter 输出或任何真实运动节点。

### P1：完成 D435i--MID-360 标定

- [ ] 采集新的 RGB、aligned depth、CameraInfo、Livox、IMU 和 odom 同步 bag；数据覆盖至少 3 个距离、3 个方位和 2 个姿态。
- [ ] 用独立片段估计 RGB--LiDAR 时间偏移，并在 held-out 片段复核唯一性和稳定性。
- [ ] 定义并计算 LiDAR 到图像的投影误差，至少记录有效点数、median、p90、RMSE 和 inlier ratio。
- [ ] 由项目负责人确定验收阈值；阈值必须先写入验收记录，再判断 pass/fail。
- [ ] 通过后更新 `projection_orin26_d435i_mid360.yaml` 的 calibration id、日期、样本数、误差和时间偏移。
- [ ] 只有全部指标通过后，将 `calibration_status` 从 `extrinsics_only` 改为 `calibrated`。

验收出口：profile 的正常启动门能够接受该标定文件；使用 held-out bag 复现同一结论。
当前 `mapping_orin26_d435i_mid360.yaml` 中 `require_calibration: false` 仅用于数据链诊断，
不能作为正式融合批准。

### P2：感知、指令与 LVLM 的无运动闭环

- [ ] 使用已批准标定从 profile 正常启动 semantic mapping，而不是绕过 profile 直接 launch。
- [ ] 记录对象节点的位置误差、对象 UID 连续性、误检和重复簇；至少覆盖桌椅、柜体和小物体。
- [ ] 在真实 `/object_nodes_list`、RGB 和 odom 上运行 `policy_mode=semantic_snapshot`、`dry_run=true`。
- [ ] 验证远程 LVLM 的 instruction parse、concept grounding 和 final verifier，记录请求类型、p50/p95 延迟、超时和原始结构化响应。
- [ ] 用模拟 `REACHED` 只验证证据闭环：目标 crop、原始指令、对象 UID 和 verifier 决策必须写入同一 run 目录。
- [ ] 验证 LVLM 不可用、响应非法或超时时显式失败并保持 WAIT/HOLD；不得吞掉异常后继续发布目标。

验收出口：在完全不发布真实 waypoint 的条件下，一条真实自然语言指令能够产生可追踪的
`InstructionPlan -> NavigationIntent -> ViewEvidence -> verifier decision`。

### P3：确定唯一的真实运动后端

- [ ] 由底盘控制器所有者书面确认 `/waypoint` 的消息类型、数组语义、单位、轴向、frame 和目标有效期。
- [ ] 确认 `/topoplan/reached_goal` 是否为生产反馈；补齐 blocked、timeout、cancel/preempt、heartbeat 和故障状态接口。
- [ ] 确认最终 `/cmd_vel` owner、速度/加速度限制、watchdog、急停断言/复位和人工接管流程。
- [ ] 完成机器人专属 controller contract，保持 `cmd_vel_direct_publish=false`。
- [ ] 根据确认后的实际反馈，只实现一个 Orin status adapter；不为尚不存在的反馈类型预写兼容分支。
- [ ] 明确生产后端是“外部 `/waypoint` 控制器”还是“SysNav 原生下层”，并在 launch 中保证互斥。

验收出口：`approval_status=approved` 只能由真实合同和现场验收记录支持，不能由单元测试或
历史源码推断产生。

### P4：分阶段运动验收

- [ ] Shadow：保持 adapter `output_enabled=false`，验证 frame、新鲜度、坐标变换和无 `/cmd_vel` 所有权冲突。
- [ ] Bench：轮子离地或电机硬件禁用，连接真实 `/waypoint`，验证正常、过期、取消和错误 frame。
- [ ] Low-speed：第二人持急停，在空场执行不超过 0.5 m 的前进、后退、转向、HOLD、blocked 和 timeout。
- [ ] Safety：实测急停、人工接管、通信中断、odom 过期和 controller heartbeat 丢失。
- [ ] Rollback：验证关闭 handoff、停止本部署容器后，底盘保持安全状态且不影响外部 LIO。

验收出口：每个状态都有底层反馈和时间戳证据；VLN 侧 `NavigationStatus` 与底盘实际状态一致。

### P5：完整实物 ObjectNav 闭环

- [ ] 先运行单目标显式指令，不启用 room prior 和复杂关系。
- [ ] 验证 `MotionGoal` 到达后才采集 `ViewEvidence`；`REACHED` 不得直接等同于任务成功。
- [ ] 验证 final verifier 的 `accept / need_better_view / reject` 会产生对应的 STOP、视角调整或重新规划。
- [ ] 再增加属性目标、anchor relation 和小目标搜索；每类至少保留成功和失败样例。
- [ ] 最后接入实物语义先验地图，完成地图 frame alignment 后再启用 room-level guidance。
- [ ] 记录成功率、路径长度、耗时、LVLM 调用次数、人工接管次数和安全中断原因。

验收出口：至少完成一组可重复的端到端场景，运行目录同时包含传感器摘要、对象图、指令计划、
运动状态、视觉证据和 final verifier 结果。

## 5. 工程实施约束

后续实现遵循以下边界：

- 不根据 topic 名称猜测消息语义；不确定项保持未完成并写明所缺证据。
- 复用已有 `MotionControllerProtocol`、`WaypointFormatAdapter` 和 `RosNavigationStatusProvider`，不再新增平行基类。
- 一个阶段只解决一个已确认接口缺口；标定、语义策略和底盘控制不混在同一修改中。
- 传感器和外部控制器输入非法时显式报错或保持 WAIT/HOLD，不把异常转换成成功。
- 不为理论上可能但现场未出现的 controller 版本编写兼容分支。
- 每次真机运行使用新的 run id；历史证据只读，不覆盖旧日志。
- LVLM 负责语义满足性，几何与下层控制器负责可达性、距离、碰撞和运动安全。

任务成功的控制语义保持为：

```text
motion_reached
AND instruction_satisfied
AND safety_state_allows_stop
```

其中 `motion_reached` 只说明机器人到达一个运动目标，`instruction_satisfied` 必须由绑定
原始指令和当前视觉证据的 verifier 给出；两者任何一个缺失都不能上报任务成功。

## 6. 下一次现场工作的最短路径

下一次 Orin 工作只做 P0 和 P1：同步共同代码基线、重建镜像、采集标定 bag、运行 held-out
评估。不要同时打开真实 waypoint。P1 通过后，再单独进行 P2 的真实对象图和远程 LVLM
无运动闭环。这样每次失败都能定位在传感器、标定、语义或控制中的单一层级。
