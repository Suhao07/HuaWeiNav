# 正式实物模式一键入口

**实物模式：已完成（2026-11-05，bridge-off 规划与速度输出验收）。** 已验证真实输入下为前方 chair 的接近目标生成局部轨迹和限幅速度；未实际行驶或到达。默认 bridge 关闭。详细参数、迁移方法及停止阈值与到达容差的配合见 [实物参数调节说明](real_robot_parameter_tuning.md)。

## 后续停止阈值修复（本地离线验证，待更新部署镜像）

按用户要求，正式默认 **停止距离 0.5 m、Action 到达容差 1.0 m**；前视距离 1.0 m、减速距离 1.0 m。新增 `NAV_FOLLOWER_STOP_DISTANCE_M`、`NAV_FOLLOWER_LOOKAHEAD_M`、`NAV_FOLLOWER_SLOWDOWN_DISTANCE_M`。入口拒绝非有限/非正数、停止距离不小于容差或前视距离的组合。`--check` 可查看有效值，不会开启 bridge。

旧 0.20/0.10 m 组合的提前停止条件已离线复现；69 项相关测试通过。见 [诊断及离线证据](real_robot_stop_distance_diagnosis.md)。下文 0.10 m 容差和镜像 ID 的验收数据属于此前正式实测，未被本次离线修复覆盖。当前车端旧镜像不包含新参数接线；需将本次源码和 `real_robot/navigation_parameters.py` 纳入构建、重建 ROS 安装目录，并使用新镜像后才会生效。本轮未构建车端镜像、未启动组件或运动。

1.0 m 是到接近点的容差，standoff 仍为 0.8 m；容差内候选不派发 Action，不强行绑定或宣称语义成功。

## 默认启动：规划与速度运行，底盘 bridge 关闭

在 Orin 的部署目录运行：

```bash
cd /home/orin26/HuaweiVLN_deployments/yoloworld_candidate_71f340c5
bash scripts/run_real_robot_navigation.sh
```

默认 `STRIVE_EXECUTION_MODE=preview`。新入口以此模式开关管理 bridge，不需要修改旧的 `START_AGILEX_PLATFORM`/`AGILEX_REMOTE_CONTROL_ENABLED`；仅设置旧开关不会让 preview 接到底盘。宿主相机和 LIO 定位需已正常启动；此入口不会重新初始化保存地图或改动车辆定位。已有 STRIVE 容器运行时会拒绝重复启动，先导出旧 runtime 数据再停止旧容器。

一键入口启动容器和正式链路：

`感知/融合 → InstructionRuntime → StandoffGoalResolver → ExecuteWaypoint Action → MotionServer → localPlanner → pathFollower → SafetyVelocityMux`

雷达 odometry 通过固定外参转换为底盘 odometry，保持同一世界原点；高层和下层共用这个底盘 pose。与旧 shadow 验收不同，正式链路不读取 JSONL 日志驱动目标，不由测试探针发布 waypoint，不伪造运动反馈。

默认目标 `Find the chair`，显式 standoff 策略要求目标在底盘前方；可用 `STRIVE_INSTRUCTION` 和 `STRIVE_DATASET_TARGET` 修改目标。该策略只处理当前可见对象的接近，不替代未知空间探索规划，也不自动解决历史实例绑定。`goal_resolution=viewpoint` 仍保留原视点策略，现入口显式选择 `standoff`。

默认输出（全部与底盘隔离）：

| 话题/Action | 内容 |
| --- | --- |
| `/strive/preview/base_odom` | 同一世界系中的底盘位姿与子坐标系 twist |
| `/strive/preview/execute_waypoint` | 正式 Action 请求/反馈/结果 |
| `/strive/preview/way_point` | 仅由 MotionServer 发布的目标 |
| `/strive/preview/path` | 原生规划轨迹，局部 vehicle 坐标系 |
| `/strive/preview/raw_cmd_vel` | 跟踪器原始候选速度 |
| `/strive/preview/cmd_vel` | 安全复用器限速/限加速度后的最终 TwistStamped |
| `/strive/preview/safety` | 安全状态 |

preview 内部不使用 dry_run 来伪造状态：Action/规划/速度真实运行，bridge 没有启动。车不移动时应触发无进展超时/hold，不能报到达或任务 STOP。参数 `NAV_NO_PROGRESS_TIMEOUT_S` 默认 12 秒，不应为了维持非零速度而延长为无限。

停止：前台 Ctrl-C 会清理所有子进程；或先导出数据再 `docker stop -t 10 huawei-vln-navigation`。日志在容器 `/workspace/STRIVE/output/runtime/navigation/`，包括各组件 `.log` 与 `runtime/runtime_decisions.jsonl`。正式入口已自动把日志挂载到宿主 `logs/navigation-<时间>/`，退出后仍保留；可通过 `NAV_HOST_RUN_DIRECTORY` 指定目录。也可手动导出副本：

```bash
mkdir -p logs/navigation-export
docker cp huawei-vln-navigation:/workspace/STRIVE/output/runtime/navigation/. logs/navigation-export/
```

## 迁移参数

这些环境变量在启动前设置，随后执行同一个一键命令：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `STRIVE_ROBOT_PROFILE` | Orin26 YOLOWorld profile | 相机/雷达话题、投影标定、模型路径、世界系配置 |
| `NAV_IMAGE_TAG` | `huawei-vln-realworld:orin-r36.5-navigation` | 已部署镜像 |
| `NAV_BASE_X_M / NAV_BASE_Y_M / NAV_BASE_YAW_RAD` | `-0.2 / 0 / -1.5708` | 底盘在雷达坐标系中的平移与 yaw；必须按新机器人标定 |
| `NAV_CLOUD_TOPIC` | `/cloud_registered` | 与 odometry 世界系相同的注册点云 |
| `NAV_VEHICLE_LENGTH_M / NAV_VEHICLE_WIDTH_M` | `0.6 / 0.6` | 规划车体外廓，按实测含突出物与裕量修改 |
| `NAV_STANDOFF_M` | `0.8` | 距目标中心的接近退让距离，不是物体表面净空 |
| `NAV_MAX_SPEED_MPS` | `0.15` | 规划/跟踪/安全 mux 线速度上限 |
| `NAV_MAX_ANGULAR_RPS` | `0.35` | 安全 mux 与 bridge 角速度上限，rad/s |
| `NAV_FOLLOWER_YAW_DEG_S` | `20.0` | 原生跟踪器角速度上限，**deg/s** |
| `NAV_MAX_ACCEL_MPS2 / NAV_MAX_ANGULAR_ACCEL_RPS2` | `0.15 / 0.3` | 安全 mux 加速度限制 |
| `NAV_GOAL_TOLERANCE_M` | `1.0` | MotionServer/高层目标 XY 容差 |
| `NAV_FOLLOWER_STOP_DISTANCE_M` | `0.5` | 跟踪点停止阈值，必须小于到达容差与前视距离 |
| `NAV_FOLLOWER_LOOKAHEAD_M` | `1.0` | 跟踪前视距离 |
| `NAV_FOLLOWER_SLOWDOWN_DISTANCE_M` | `1.0` | 根据局部路径末端距离减速 |
| `NAV_COMMAND_TIMEOUT_S` | `0.25` | 安全 mux 与 bridge 指令超时，需不超过合同上限 |
| `NAV_NO_PROGRESS_TIMEOUT_S` | `12.0` | 无进展停止时限 |
| `NAV_HOST_RUN_DIRECTORY` | 部署目录 `logs/navigation-<时间>` | 宿主持久化日志目录 |
| `NAV_RUN_DIRECTORY` | `/workspace/STRIVE/output/runtime/navigation` | 容器内日志目录，建议每次实验单独指定 |

雷达到底盘适配只支持固定 XY/yaw 外参，转换标准 Odometry 子坐标系 twist；不是任意 SE(3) 标定或定位滤波器。相机/深度/雷达投影标定、时钟同步、world_frame、雷达高度/障碍高度阈值也须按新机器人核对。不能只改车长车宽就认为已迁移完成。

## 后续开启 bridge：需要明确修改的参数

**本次没有执行以下 live 命令，也没有填写或伪造控制合同审批。**

1. `STRIVE_EXECUTION_MODE=live`：控制命名空间切换到 `/strive/navigation`，启用底盘 bridge，并要求 chassis-ready 与合同校验。默认仍以 HOLD 启动，由 bridge 根据实际状态/选源发布 autonomy enable。
2. `ALLOW_REAL_MOTION=true`：单独、显式允许真实运动；缺失则启动前拒绝。
3. `CONTROL_CONTRACT_FILE=/workspace/STRIVE/real_robot/control/ROBOT_controller_contract.yaml`：必须是已核定的合同，不能直接使用模板。宿主机文件放在部署 `real_robot/control/` 或指定 `CONTROL_CONTRACT_HOST_DIR`；容器只读挂载该目录。
4. `AGILEX_URI`：底盘 rosbridge 地址，本台默认 `ws://192.168.1.102:9090`。更换非 AgileX 底盘需实现对应 bridge，不能仅更换 URL。
5. live 启动后仍保持 HOLD。现场再次明确授权、确认通道可用后，使用 `/platform/set_autonomy` 的 `std_srvs/srv/SetBool` 服务请求 `data: true`；桥接器核验遥测与选源后才发布自动模式授权，高层等待这份新鲜授权才下发 Action。返回手动使用 `data: false`。本次未调用该服务。
6. 合同还必须填写 `agilex_rosbridge.hardware_state` 的 `allowed_control_modes`、`ready_base_states`、`ready_fault_codes`、`telemetry_timeout_s`；不要把未核定的故障码（包括此前的 128）加入 ready 列表。
7. 同步合同与 `NAV_MAX_*` 的速度/加速度/超时限制，以及下表话题。合同或 bridge 自检失败时不得放宽故障码绕过。

合同中的关键接口须与 live 入口一致：

| 项目 | live 值 |
| --- | --- |
| waypoint / action / frame | `/strive/navigation/way_point` / `/strive/navigation/execute_waypoint` / `camera_init`（或 profile 世界系） |
| autonomy / manual / final velocity | `/strive/navigation/raw_cmd_vel` / `/strive/navigation/manual_cmd_vel` / `/strive/navigation/cmd_vel` |
| planner status | `/strive/navigation/status` |
| manual takeover / estop | `/strive/navigation/manual_takeover` / `/strive/navigation/estop` |
| 最终速度 owner | `safety_velocity_mux` |

示例（仅供后续现场授权后使用）：

```bash
STRIVE_EXECUTION_MODE=live ALLOW_REAL_MOTION=true \
CONTROL_CONTRACT_FILE=/workspace/STRIVE/real_robot/control/ROBOT_controller_contract.yaml \
bash scripts/run_real_robot_navigation.sh
```

`--check` 仅检查模式/必需路径及默认输出，不启动 ROS/Docker，不代表合同内容或底盘状态已验收。开启 bridge 前仍需核定底盘模式与故障码、指令选源、轴向/单位、硬件 watchdog、急停和人工接管；这些不能由 preview 轨迹替代。原生跟踪器静止时的转向切换虽已受 mux 限加速度约束，实际闭环稳定性仍须低速现场测试。

## 本次接入修复

- 将已验证 standoff 几何提取为共享模块，正式 runtime 可显式选择；旧 viewpoint 模式默认行为保留。
- 新增底盘 odometry 适配，正式高层与下层共用同一底盘坐标。
- standoff 的 XY 到达容差显式进入 Action 请求，避免客户端旧的 0.35 m 默认覆盖入口的 0.10 m 配置；已经处于配置容差内时不制造运动目标。
- MotionServer 终止结果（包括无进展/超时/到达）先取消原生目标并 hold，再返回原终止原因，防止 Action 已结束而旧轨迹继续产生速度。
- live 高层等待 bridge 的新鲜自动模式授权；preview 无 bridge、独立控制命名空间。
- 宿主日志持久化；一键入口拒绝并行启动另一套已知 STRIVE 容器。

## 验证证据与外部服务边界

本地 56 项针对性测试通过，涵盖目标解析、Action 终止撤销、控制合同、限速/watchdog 与 live 授权等待。最终镜像经过断网的 11 个源码/配置文件 SHA256/编译/导入检查。

一次车端复测遇到外部 LIO 进程存活但停止发布的故障；preview 因没有定位与对象输入保持 WAIT/零速度，没有把该窗口记为规划成功。用户确认车位未变后，单独恢复同一保存地图的 Point-LIO 并使用之前验证过的初值。这个恢复步骤不加入自动一键入口：未知车位时不能自动复用旧初值。

DDS 传输：正式入口设置 `FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/STRIVE/real_robot/config/fastdds_udp.xml`，明确使用 UDPv4，关闭共享内存传输。宿主 Point-LIO 也使用同一配置内容；本次通过 tmux `livox_odom` 会话环境设置。若重新创建该传感器会话/改为 systemd 服务，需要在定位进程启动前设置宿主配置路径，不能只在导航容器内设置。相机/雷达自身的启动和地图初值仍由传感器部署维护。

## 最终部署回执（2026-11-05）

- 镜像：`huawei-vln-realworld:orin-r36.5-navigation`，ID `896fbd15b040f6d8ea1a5a4f5ba1023fa537f92b5935a3679d75ee9a6e23cffc`。
- 容器：`huawei-vln-navigation`，当前 `preview`；bridge 未启动，物理控制话题发布者为 0。
- 55 秒真实输入窗口：5353 条底盘 odometry，1 个正式 Action 目标，1151 条有效轨迹，最多 35 点/约 0.34 m。最终限幅速度峰值 0.0525 m/s、0.0494 rad/s。
- Action 结果为 `blocked / no_progress`，没有 REACHED；到达容差 0.10 m。终止后观测到连续 11.51 秒零速度。
- 重启导航容器后，同一个 LIO PID 46253 仍持续供数：5293 条 odometry、1162 条有效轨迹，末尾 100 条速度全零。UDP-only 配置后的这次重启回归通过；这不是长期稳定性统计，也未单独证明此前故障全部由共享内存导致。
- 本地证据：`logs/diagnostics/formal-navigation-validation/delivery_receipt.json`。宿主最终构建上下文：部署根目录 `formal-navigation-build/`；所有源码/配置哈希记录在 `manifest.json`。

本次仍未进行实际行驶、未知空间探索或到达后的视觉 STOP 验收。
