# VLN 实物部署状态与 TODO

> 更新日期：2026-09-08
> 代码基线：本地 `realworld` 最新为 `ea37c38`（包含 `ad08c1a` 的低速 OCC 融合接受结论和 `3978b5c` 的 semantic mapping 合并索引修复）；`origin/realworld` 仍落后 6 个提交，未在本轮推送。新机器人工作区已用同一修复提交为 `0acbf3e` 并用于镜像重建，旧 `/home/orin26/HuaweiVLN` 保留为只读回滚工作区，未被覆盖。
> 目标平台：Orin-26、Intel RealSense D435i、Livox MID-360
> 当前安全状态：感知与影子控制验证；真实运动未批准

本文是实物部署的单一进度清单。设计接口见
[`real_robot_framework.md`](real_robot_framework.md)，实测证据保留在对应日期的
`real_robot_*_evidence_*.md` 文档中。

### 2026-09-08 清洁工作区部署记录

- 新工作区：`/home/orin26/HuaweiVLN_deployments/realworld_2c0b783`；旧
  `/home/orin26/HuaweiVLN` 未修改。
- 旧工作区备份证据：`/home/orin26/HuaweiVLN_backups/HuaweiVLN_pre_clean_20260907T151320Z/`
  （staged/unstaged patch、Git 状态和未跟踪清单）。
- 迁移内容限定为共同 Git 基线、已确认的传感器/坐标/融合/控制 adapter 原理代码、候选
  标定文件和五个现有模型资产；未复制旧 logs、outputs、build/install 或缓存。
- 新镜像 `huawei-vln-realworld:orin-r36.5` 已重建，ROS overlay 的 7 个包编译成功。
- 外部 Livox/Point-LIO 仍由 `/home/orin26/code` 所有；通过项目包装器以 ROS 参数开启
  `publish.scan_publish_en` 和 `publish.scan_bodyframe_pub_en`，未编辑该工作区。
- LIO-only 诊断报告：`logs/diagnostics/lio_dds_20260907T154932Z.md`；四个 topic
  均收到真实消息。独立 60 秒测量约为 `/cloud_registered_body` 9.43 Hz、odom 99.5 Hz。
- semantic mapping 已在新容器中启动，输入明确为 `/cloud_registered_body`；对象节点
  `/huawei_vln/d435i_object_nodes_list` 约 0.4--0.5 Hz，样本 frame 为 `map`、坐标为米级，
  融合日志持续出现有效投影点；对象图像已正常写入 `output/object_images/`。
- shadow adapter 仅使用 `output_enabled=false`；合成 PointStamped 已完成 ego 转换日志，
  未创建 `/waypoint` publisher，也未发布 `/cmd_vel`。
- 本轮仍未批准真实底盘控制；当前实际配置的 controller contract 仍为
  `approval_status: unapproved`。

### 2026-09-08 实时验收续测

- Point-LIO/Livox：`/livox/lidar` 约 100 Hz、`/livox/imu` 约 200 Hz，
  `/cloud_registered_body` 约 9.4 Hz，`/cloud_registered` 约 100 Hz，
  `/aft_mapped_to_init` 约 100 Hz；60 秒 `tf2_echo camera_init -> base` 收到 57 个
  连续样本，除启动前两次 frame 尚未出现外没有 `future extrapolation`。
- D435i：RGB、aligned depth、CameraInfo 均有实时消息；RGB/Depth 约 15 Hz、
  `1280x720`，编码分别为 `bgr8`/`16UC1`，CameraInfo frame 为
  `d435i_camera_color_optical_frame`。该 frame 与 projection YAML 的历史别名不同，
  当前仍通过显式 OCC 投影配置处理，未广播伪 TF。
- Semantic mapping：发现对象合并索引缺陷并在 `3978b5c` 修复；容器内重建后连续运行
  超过 2 分钟无 `IndexError`、`Traceback` 或 `future extrapolation`。真实对象节点
  `/huawei_vln/d435i_object_nodes_list` 约 0.55 Hz，样本包含 `printer` 和合并后的
  SysNav object ID，坐标为 `map` 米制。
- `semantic_snapshot`：以自然语言 `Find the printer`（显式 `dataset_target=printer`）
  运行 `dry_run=true`，已记录 `InstructionPlan`、目标 UID `sysnav_object:3` 和地图坐标；
  因没有输入 `/viewpoint_rep_header`，`sysnav_viewpoint_bridge` 没有可发布的
  `ViewpointPose`，runtime 持续 `WAIT`，未生成 waypoint 或速度命令。
- Viewpoint bridge：节点已启动并订阅 `/viewpoint_rep_header`、`/huawei_vln/d435i_object_nodes_list`
  和 `/aft_mapped_to_init`；当前 `/viewpoint_rep_header` publisher 为 0，
  `/strive/sysnav/viewpoint_pose` 没有实际消息，因此真实 SysNav viewpoint 仍待外部
  SysNav 规划器启动后验收。
- 30 分钟资源/温度监控已完成；最终报告保留在
  `logs/diagnostics/resources/20260908T034523Z/`。

### 2026-09-08 监控收尾

- 30 分钟监控已完成：`/cloud_registered_body` 平均 9.441 Hz（窗口范围
  9.202--9.677 Hz），`/aft_mapped_to_init` 平均 99.661 Hz（最低窗口
  80.992 Hz）；Point-LIO 进程 CPU 54.7--62.2%，RAM 4.98--7.29 GB，
  `tj` 温度 61.5--72.7°C。监控目录内没有 `future extrapolation`、崩溃或 traceback。
  该监控的 `process.log` 以 Point-LIO 为跟踪对象，不能替代“修复后 detector + semantic
  mapping + Point-LIO 连续 30 分钟”的专门整栈 soak；整栈长测仍需单独安排。
- 语义合并索引修复已固化为镜像
  `huawei-vln-realworld:orin-r36.5-fix`（ID `27d904a21207`，7 个 ROS 包构建成功）；
  原 `huawei-vln-realworld:orin-r36.5` tag 保留未覆盖。临时检测/建图容器已停止，
  Livox、Point-LIO 和 D435i 外部进程保持运行。

勾选规则：只有代码、命令输出或真机记录能够直接证明的事项才标记为完成。单元测试、
bag replay、HIL 和影子 topic 不能替代真实底盘验收。

## 1. 当前结论

### 2026-09-07 状态刷新

用户已确认 OCC 项目的 D435i--MID-360 外参与当前安装状态一致，并批准当前 candidate
用于低速 VLN 试运行。当前 profile 已写入 `calibration_status: calibrated`，外参文件为
`real_robot/calibration/orin26_d435i_mid360_candidate_20260826.json`。因此，**低速场景的
空间融合不再是部署阻塞项**。动态时间偏移和严格像素投影验收属于高动态/高精度增强项，
不再阻塞当前低速 VLN 验收。

最新 candidate 的证据摘要：

- 静态有效阶段：12/18（近距 6 个阶段因棋盘超出画面不可用）；
- 平面 P90：train 0.0105 m，held-out 0.0112 m；
- 深度 P90：train 0.0684 m，held-out 0.0692 m；
- held-out 深度内点率：79.11%；
- 像素代理 P90：train 11.46 px，held-out 11.27 px；
- 动态时间偏移：train -15 ms，held-out +20 ms，差异 35 ms，仍未验证。

因此当前允许启动 semantic mapping 做静态/低速数据链试运行；
`time_offset_status: unvalidated` 继续保留为事实记录，但不否定低速融合可用性。
真实 waypoint、`/cmd_vel` 和底盘运动仍受独立控制安全门禁约束。
Point-LIO 动态 TF 仍需连续窗口测试；出现 `future extrapolation` 时不能称为稳定部署完成。

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
这些结果证明低速数据链和接口可运行；它们仍不能证明底盘能够安全执行 waypoint，
也不能替代高动态时间偏移/像素精度增强验收。

当前低速 VLN 仍缺三类硬条件：

1. 在修复后的 detector + semantic mapping + Point-LIO 组合上完成专门的整栈长时间
   soak，并把动态 TF、资源、掉帧和温度结果写入同一运行证据；
2. 接入并验收真实 SysNav viewpoint provider，完成不发布 waypoint 的任务级语义闭环；
3. 由底盘所有者批准 controller contract，完成 bench、低速和安全测试后，才能打开真实
   waypoint/底盘输出。

RGB--LiDAR 动态时间偏移和严格像素投影误差不再列为低速 VLN 的硬条件；若目标扩展到
高速或高精度融合，再单独开启增强验收。当前 `calibration_status: calibrated` 是经用户
确认的 OCC 低速运行决定，不应在后续清单中反复作为 P0 阻塞项。

本轮已经完成的**软件闭环**如下，但这些内容仍不等同于实物验收：

```text
SemanticMapSnapshot
  -> NavigationIntent
  -> SysNavGoalResolver
  -> MotionGoal / WAIT
  -> active goal tracking
  -> REACHED 后的新鲜 RGB / detection
  -> ViewEvidence / verifier 生命周期
```

其中 `SysNavGoalResolver` 只消费下层提供的 viewpoint pose，不再把对象中心或房间质心
伪装成执行位姿；当前迁移的 ROS 消息还没有携带完整 viewpoint pose，因此正式语义路径
在 bridge 尚未收到时间对齐的 viewpoint pose 时显式保持 `WAIT`。
`scripts/check_real_robot_acceptance.sh` 当前离线结果为 `113 passed`，该结果只证明软件
合同和状态生命周期，不代表真实控制器已接管底盘。

## 2. 已完成与证据边界

| 模块 | 当前状态 | 可证明内容 | 不能据此声称 |
|---|---|---|---|
| 平台无关 contract | 已完成 | `SemanticMapSnapshot`、`NavigationIntent`、`MotionGoal`、`NavigationStatus`、`ViewEvidence` 已实现并有单元测试 | 真车已运动 |
| SysNav detector | 真机数据链已验证 | D435i 图像进入 YOLOE，检测 topic 有真实消息 | 所有目标类别均稳定检测 |
| SysNav semantic mapping | 低速真机数据链已验证 | 2026-09-08 新工作区实测对象节点约 0.4--0.5 Hz，坐标为米级，含有效投影点 | 对象精度和长时间稳定性仍需任务级验收 |
| LIO 输入 | 60 秒真机数据链已验证 | `/cloud_registered_body` 约 9.4 Hz，`/aft_mapped_to_init` 约 100 Hz | 长时 TF、漂移和资源稳定性 |
| D435i--MID-360 标定 | 低速试运行已接受 | OCC 外参/current candidate 已导入，profile 为 `calibration_status=calibrated` | 高动态时间偏移和严格像素精度增强验收 |
| Instruction runtime | 软件闭环完成（离线） | snapshot、SysNav viewpoint resolver、active goal、REACHED 后新鲜取证和 verifier 生命周期有离线测试 | 已在 Orin 上运行真实自然语言任务 |
| SysNav viewpoint bridge | 软件 bridge 已实现，真机数据待验收 | `/viewpoint_rep_header` + 同时刻 odom -> `/strive/sysnav/viewpoint_pose`；object-viewpoint 观测历史、共视交集和无效时间/frame 的 WAIT 已有代码和测试 | Orin 上已验证 viewpoint 频率、时间偏移和 frame |
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

## 4. 当前待做清单

### P0：传感器、坐标与整栈运行稳定性

- [x] 新的机器人工作区和 `huawei-vln-realworld:orin-r36.5-fix` 镜像已建立；旧
  `/home/orin26/HuaweiVLN` 未覆盖，外部 `/home/orin26/code` 未修改。
- [x] Livox、Point-LIO、D435i 实时输入已恢复；当前 `/livox/lidar` 约 100 Hz、
  `/livox/imu` 约 200 Hz、`/cloud_registered_body` 约 9.4 Hz、
  `/aft_mapped_to_init` 约 100 Hz，且 Point-LIO 使用 `prop_at_freq_of_imu=false`。
- [x] 完成独立 60 秒 LIO/TF 检查；`camera_init -> base` 有连续样本，检查窗口未出现
  `future extrapolation` 或 `past extrapolation`。
- [x] D435i RGB、aligned depth、CameraInfo 的实时分辨率、编码和内参已记录；detector
  使用 BGR→RGB、CUDA、`imgsz=320`，与当前运行 profile 一致。
- [x] OCC 外参已按当前安装状态作为低速融合标定值使用；不再把严格动态时间偏移或像素
  误差作为低速 VLN 阻塞项。
- [ ] 在修复后的 detector + semantic mapping + Point-LIO **同时运行**时完成一段专门的
  30 分钟以上整栈 soak，记录各节点频率、掉帧、CPU/GPU、内存和温度。已有 30 分钟记录
  主要跟踪 Point-LIO，不能替代这项。
- [ ] 明确记录 `camera_init -> aft_mapped -> base` 的长时 TF 时间戳，以及下游若需要时
  `camera_init -> d435i_camera_color_optical_frame` 的实际连接方式；当前映射使用显式 OCC
  投影配置，禁止广播未经确认的伪 TF。
- [ ] 将本地 `realworld` 的最新完整 SHA（当前 `ea37c38`）推送到 `origin/realworld`，
  并让机器人部署 manifest 固定到同一完整 SHA；这属于可追溯性，不是传感器启动阻塞。

验收出口：整栈运行证据能对应到同一代码 SHA、镜像 ID 和 profile，且全程未发布真实
`/waypoint`、`/way_point` 或 `/cmd_vel`。

### 历史记录：2026-08-21 无运动实物执行记录

> 以下条目保留作历史证据，不覆盖 2026-09-08 的当前状态；其中当时的 LIO 无消息和旧镜像
> 结果已由后续清洁工作区部署和实时检查更新。

- Profile check：`orin26_livox_mid360_generic_rgb` 与 `orin26_livox_mid360_d435i` 均通过；两者均保持 `START_SEMANTIC_MAPPING=false`、`dry_run=true`、真实运动关闭。
- 镜像与软件：`huawei-vln-realworld:orin-r36.5`=`25048b859e0d`；ROS workspace 7 包编译成功；离线 acceptance 为 `113 passed`。
- GPU/模型 smoke：`torch cuda=True`，YOLOE detector init 成功；USB 相机只读 smoke 收到 `/camera/image` 和 `/camera_info`，分辨率 `1920x1080`、编码 `rgb8`、frame `default_cam`。
- LVLM/schema/verifier 影子测试：在 `--network none` 的最终镜像中运行结构化输出、非法 JSON 重试、保守 fallback、final verifier 和 deployment receipt 测试，`16 passed`；未进行真实远程 LVLM 延迟验收。
- runtime shadow：D435i profile runtime smoke 和 live `semantic_snapshot` dry-run 均保持 `WAIT`，原因是实时 `object_nodes`、pose、image 尚未同时到达；未发布 `/way_point`、`/waypoint` 或 `/cmd_vel`。
- waypoint adapter shadow：节点日志为 `output_enabled=False`；`/waypoint`、`/strive/test_waypoint_array`、`/way_point`、`/cmd_vel` 均未发现 publisher。
- LIO 只读诊断：报告为 `logs/diagnostics/lio_dds_20260821T133422Z.md`；`/livox/lidar`、`/livox/imu`、`/cloud_registered`、`/aft_mapped_to_init` 均未收到实际 header。Point-LIO 参数显示 `publish.scan_publish_en=False`，因此 semantic mapping 继续关闭；未重启或修改外部 LIO。
- 现场 bag 清单中暂未发现同时包含 `/viewpoint_rep_header` 与 `/object_nodes_list` 的 bag，因此 viewpoint replay 留待取得对应 bag 后执行。

### P1：真实对象图、viewpoint 与无运动任务闭环

- [x] 使用已批准 OCC candidate 从 profile 启动 semantic mapping，输入固定为
  `/cloud_registered_body`；对象节点已有真实样本，坐标为 `map` 米制。
- [x] semantic mapping 的对象合并索引缺陷已在 `3978b5c` 修复，并已重建到机器人镜像；
  修复后运行未再出现该 `IndexError`。
- [x] 完成 `SemanticMapSnapshot -> NavigationIntent -> SysNavGoalResolver ->
  MotionGoal/WAIT` 软件链路和 `REACHED -> fresh evidence -> verifier` 生命周期的离线测试。
- [x] `semantic_snapshot` 已在 Orin 上以 `dry_run=true` 读取真实对象节点；无可执行 viewpoint
  时保持 `WAIT`，没有发布任何 waypoint 或速度命令。
- [ ] 记录真实对象节点的位置稳定性、UID 连续性、误检和重复簇；至少覆盖桌椅、柜体和
  小物体，并保留对应 RGB/点云证据。
- [ ] 获取真实 SysNav viewpoint provider，使 `/viewpoint_rep_header` 有实际 publisher；
  当前 bridge 节点虽已启动，但输入 publisher 为 0。
- [ ] 在真实 viewpoint、object nodes、RGB 和 odom 同时存在时，完成不发布 waypoint 的
  `policy_mode=semantic_snapshot` 任务验收，记录 viewpoint 的 frame、频率、时间偏移和
  object-viewpoint 关联。
- [ ] 对远程 LVLM 做网络侧 instruction parse、concept grounding 和 final verifier 测试，
  记录 p50/p95 延迟、超时、服务不可用和非法 JSON 时的 `WAIT/HOLD` 行为；密钥不得写入
  仓库或日志。

验收出口：一条真实自然语言指令能在无运动条件下产生可追踪的
`InstructionPlan -> NavigationIntent -> ViewpointPose/WAIT -> ViewEvidence -> verifier`
证据；没有 viewpoint 时必须明确保持 `WAIT`。

### P1-增强：高动态/高精度融合（低速 VLN 非阻塞）

- [x] OCC D435i--MID-360 外参与当前安装状态一致，`calibration_status: calibrated` 已按
  低速 VLN 决定写入 profile。
- [ ] 若以后扩展到高速或高精度场景，再用独立连续片段复核 RGB--LiDAR 时间偏移的 held-out
  稳定性。
- [ ] 若以后需要像素级精度，再计算严格 LiDAR→图像边缘误差和更严格深度误差。

以上两项不阻塞当前低速 semantic mapping 或 VLN 试运行。`require_calibration: false` 仍是
诊断开关，不能代替 profile 中已记录的 candidate。

### P2：平台无关软件回归（已完成，持续维护）

- [x] `SemanticMapSnapshot -> NavigationIntent -> SysNavGoalResolver -> MotionGoal/WAIT`、
  `REACHED -> fresh evidence -> verifier`、`ViewpointPose` bridge 和 waypoint adapter 均有
  离线测试；最近一次 acceptance 为 `113 passed`。
- [x] waypoint adapter 当前保持 shadow：`output_enabled=false`，不创建生产 waypoint
  publisher，不发布 `/cmd_vel`。
- [ ] 每次修改 detector、mapping、viewpoint 或 runtime 后重新运行 acceptance，并把结果
  和部署 SHA 写入新的 run 目录。

### P3：确定并批准唯一的真实运动后端（当前最大阻塞）

- [ ] 由底盘控制器所有者书面确认 `/waypoint` 的消息类型、数组语义、单位、轴向、frame 和目标有效期。
- [ ] 确认 `/topoplan/reached_goal` 是否为生产反馈；补齐 blocked、timeout、cancel/preempt、heartbeat 和故障状态接口。
- [ ] 确认最终 `/cmd_vel` owner、速度/加速度限制、watchdog、急停断言/复位和人工接管流程。
- [ ] 完成机器人专属 controller contract，保持 `cmd_vel_direct_publish=false`，并由底盘
  所有者签字/现场确认后才允许改变。
- [ ] 根据确认后的实际反馈，只实现一个 Orin status adapter；不为尚不存在的反馈类型预写兼容分支。
- [ ] 明确生产后端是“外部 `/waypoint` 控制器”还是“SysNav 原生下层”，并在 launch 中保证互斥。

验收出口：`approval_status=approved` 只能由真实合同和现场验收记录支持，不能由单元测试或
历史源码推断产生。

### P4：分阶段安全运动验收（P3 批准后）

- [x] Shadow：保持 adapter `output_enabled=false`，已验证 frame、新鲜度、坐标变换和无
  `/cmd_vel` 所有权冲突；不得把 shadow 结果当作真实控制验收。
- [ ] Bench：轮子离地或电机硬件禁用，连接真实 `/waypoint`，验证正常、过期、取消和错误 frame。
- [ ] Low-speed：第二人持急停，在空场执行不超过 0.5 m 的前进、后退、转向、HOLD、blocked 和 timeout。
- [ ] Safety：实测急停、人工接管、通信中断、odom 过期和 controller heartbeat 丢失。
- [ ] Rollback：验证关闭 handoff、停止本部署容器后，底盘保持安全状态且不影响外部 LIO。

验收出口：每个状态都有底层反馈和时间戳证据；VLN 侧 `NavigationStatus` 与底盘实际状态一致。

### P5：完整实物 ObjectNav 闭环

以下三项已经具备离线软件实现，但仍需在真实传感器、SysNav viewpoint provider 和底盘
反馈上重新验收；因此不把它们作为实物完成项：

- [x] 软件层实现 `MotionGoal` 到达后再进入 `ViewEvidence`，`REACHED` 不直接等同于任务成功。
- [x] 软件层实现到达时间戳之后的新鲜 RGB、pose 和 detection 检查；不满足时保持同一 goal 的 `VERIFYING`，不重发 waypoint。
- [x] 软件层保留 `accept / need_better_view / reject_candidate` 的 verifier 决策边界，具体视角重规划仍依赖实际 SysNav viewpoint provider。

实物验收仍从以下条目开始：

- [ ] 先运行单目标显式指令，不启用 room prior 和复杂关系。
- [ ] 在真实底盘上验证 `MotionGoal` 到达后才采集 `ViewEvidence`；`REACHED` 不得直接等同于任务成功。
- [ ] 在真实传感器上验证 `REACHED` 后的 RGB、pose 和 detection 时间戳晚于到达事件；未满足时保持同一 goal 的 VERIFYING，不重发 waypoint。
- [ ] 在真实 SysNav viewpoint provider 和底盘反馈上验证 final verifier 的 `accept / need_better_view / reject` 会产生对应的 STOP、视角调整或重新规划。
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

按以下顺序推进，期间保持真实运动关闭：

1. 用 `huawei-vln-realworld:orin-r36.5-fix` 重启一套干净进程，完成 detector + semantic
   mapping + Point-LIO 的整栈 30 分钟 soak，保存频率、掉帧、资源、温度和 TF 证据。
2. 启动真实 SysNav viewpoint provider，确认 `/viewpoint_rep_header` 有消息；再运行一条
   `semantic_snapshot` 自然语言指令，完成对象 UID、viewpoint、RGB 和 verifier 的无运动
   证据闭环。没有 viewpoint 时必须继续 `WAIT`。
3. 由底盘所有者确认并批准 controller contract，补齐状态 adapter；在批准前绝不发布真实
   `/waypoint` 或 `/cmd_vel`。
4. 依次执行轮子离地 bench、有人持急停的低速短距离和通信/急停故障测试，完成后才进入
   完整 ObjectNav。

高动态时间偏移、严格像素误差和深度精度保留为非阻塞增强项；只有目标场景升级时才重新
开启该验收线。
