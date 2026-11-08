# 实物模式参数调节与迁移说明

## 状态与适用范围

**实物模式：已实现。** 2026-11-05 已按约定完成 bridge-off 规划与速度输出验证；下述现场验收边界不改变软件已实现状态。
真实相机、点云和定位输入已接入正式 runtime → Action → MotionServer → 原生规划/跟踪 → 安全 mux；默认不启动底盘 bridge。启动方式见 [正式入口](real_robot_navigation_entry.md)。

“规划到目标”在此指为前方 chair 的接近点生成局部轨迹和速度，不是车辆已到达，也不保证完整轨迹一次覆盖到目标。正式记录含 1 个 waypoint、1,151 条有效路径，最长约 0.34 m、最多 35 点；最终速度峰值 0.0525 m/s、0.0494 rad/s。车辆静止，最终为 `blocked/no_progress`，没有 `REACHED`，之后持续零速 11.51 s。证据：[validation.json](../logs/diagnostics/formal-navigation-validation/validation.json)。这些是已保存的测试结果，不是当前在线状态。

真实行驶、移动中重规划、最终停车与视觉 STOP、未知空间探索属于下一阶段现场验证。它们不再阻塞本次软件部署完成标记；也不能由本次静止验收推断已通过。本次后续修复已更新本地启动配置并离线验证；车端旧镜像未更新，未开启 bridge 或运动。

## 参数从哪里改

| 层级 | 改法 | 生效边界 |
| --- | --- | --- |
| 已暴露的 `NAV_*` | 启动前设置环境变量 | 主机脚本经 Docker 白名单传入容器；重新启动正式入口生效 |
| lower launch 参数 | 修改 `sysnav_lower_stack.launch.py` 默认值，或在容器入口增加显式 launch 参数 | 部分值被正式入口覆盖，不能只改默认值；更新部署镜像并核对有效值 |
| 原生 localPlanner/pathFollower 参数 | 在 lower launch 对应 Node 的 `parameters` 中显式传入 | 当前未提供任意参数透传；不能凭空设置同名 `NAV_*`，也不能假定 `ros2 param set` 会实时更新 C++ 缓存 |
| 感知/标定/外部 LIO | 修改所选机器人 profile、投影/融合配置或传感器服务配置 | 核对配置文件挂载、启动覆盖值和镜像版本；导航入口不负责重置或重启 LIO |

下表的“正式值”来自正式入口及其未覆盖的源码默认值，不代表重新读取了在线参数。迁移时必须保存实际配置；只做 `--check` 不等于 ROS 或硬件验收。

## 1. 坐标系、外参和输入：迁移首先检查

| 参数/配置 | 当前正式值 | 调整依据与验证 |
| --- | --- | --- |
| `NAV_BASE_X_M / NAV_BASE_Y_M / NAV_BASE_YAW_RAD` | `-0.2 / 0 / -1.5708` | 底盘在雷达坐标系中的位姿，米/弧度。测量新平台安装关系，验证车头 +X 与目标前方判定、轨迹方向一致；方向反了先查外参，不能用速度负号补救 |
| `SYSNAV_WORLD_FRAME`、`ODOM_TOPIC` | `camera_init`、`/aft_mapped_to_init` | 保持地图、目标、注册点云和 odom 同一世界原点。重启或搬动车辆后先重定位；改 frame 字符串不能完成坐标变换 |
| `NAV_CLOUD_TOPIC` | `/cloud_registered` | 规划使用世界系注册点云；不要替换成 `/cloud_registered_body` 后仅改标签。后者是当前感知链路的输入之一，两者用途不同 |
| 相机内参、雷达到相机投影外参 | 由机器人 profile 指定 | 换相机、分辨率或支架后重新核对。用近/远处物体、画面左右两侧检查投影和 3D 中心，不能只对齐一张椅子 |
| `FASTRTPS_DEFAULT_PROFILES_FILE` | 容器 `/workspace/STRIVE/real_robot/config/fastdds_udp.xml` | 当前使用 UDP-only DDS；外部 LIO 服务启动前也需对应配置。容器重启回归通过一次，不代表长期稳定性已证明 |

底盘适配器目前仅显式配置固定 XY/yaw，不是任意六自由度外参接口。新平台若有明显高度、横滚或俯仰安装差异，应扩展变换并验证，不要挤进 XY 参数。原生 `sensorOffsetX/Y` 当前为 0；已有底盘 odom 转换时不要再次补同一外参。

迁移验收还须区分输入量级与几何正确性：有限且量级合理的 odometry 不能证明对象投影或外参正确；异常位姿/速度样本应由质量门拒绝，阈值以实际机器人配置为准，不直接沿用历史诊断数值。viewpoint 的点云/里程计必须在同一世界系并满足实际时间配对门限，相机高度需现场标定；若使用外部 viewpoint 生产者，不应再并行启动内置生产者。

## 2. 车体尺寸、障碍与接近距离

| 参数 | 当前值/入口 | 调节含义与风险 |
| --- | --- | --- |
| `NAV_VEHICLE_LENGTH_M / NAV_VEHICLE_WIDTH_M` | `0.6 / 0.6` m | 按实际外廓含突出部件测量；用于原生车体/旋转障碍检查。缩小会低估碰撞，扩大可能使狭窄处无解 |
| `NAV_STANDOFF_M` | `0.8` m | 到对象中心的接近距离；增大则停得更远，减小则更近。不是离椅子表面的净空，也不是完整车体碰撞保证 |
| `use_terrain_analysis` | 正式入口固定 `false` | 当前用原始注册点云。切为 true 必须同时验证 terrain 输出与高度语义，不能只翻开关 |
| `minRelZ / maxRelZ` | 原生 `-0.5 / 0.25` m | 相对车辆位置变换后的高度裁剪。需检查地面是否被当障碍、椅腿/低矮障碍是否被裁掉；迁移时结合 odom 原点高度核对 |
| `laserVoxelSize` | 原生 `0.05` m | 增大减少点数但可能丢细椅腿；减小保留细节但增加计算量和噪声影响 |
| `pointPerPathThre` | 原生 `2` | 碰撞路径计数阈值。提高可能漏掉稀疏障碍，降低可能对孤立噪声过敏；以保留的原始点云验证，不能只看有没有路径 |
| `checkObstacle / checkRotObstacle` | 正式 launch 均为 true | 保持障碍与转动扫掠检查。不要关闭它们来让测试通过 |
| `goalClearRange` | 原生 `0.5` m | 配合 `pathCropByGoal` 决定纳入检查的目标距离范围；代码按 `relativeGoalDis + goalClearRange` 裁剪，并非清空目标周围障碍的半径 |

**换更大的机器人不能只改长宽。** 原生规划还读取预生成 `paths.ply`、`correspondences.txt` 等碰撞对应关系；长宽参数没有重新生成这些资源。应核对路径库生成假设、缩放与实际车体扫掠范围，必要时重新生成路径资源。`pathScale=1.0`、`minPathScale=0.75` 等不是通用安全膨胀参数。

接近距离应结合对象沿接近方向的尺寸、底盘前缘、定位/感知误差和停车余量设置。中心距离 0.8 m 对不同椅子不具有相同净空。当前候选还要求对象在底盘前方，机器人距对象中心大于 `standoff + 0.15 m`，候选位移大于到达容差；因此增大 standoff 可能直接导致候选被拒绝，而不是总能产生更远目标。

## 3. 到达容差与跟踪：实际运动前重点核对

| 参数 | 当前值 | 调整方向与联动 |
| --- | --- | --- |
| `NAV_GOAL_TOLERANCE_M` | `1.0` m（按用户要求） | 同时传给 runtime 的 MotionGoal 和 MotionServer。过小会被定位抖动或跟踪停止距离阻挡，过大可能过早报告到达 |
| `NAV_FOLLOWER_STOP_DISTANCE_M` → `stopDisThre` | 正式 launch `0.5` m；原生独立运行默认仍为 `0.20` m | 作用于当前跟踪路径点距离 `dis`，进入阈值后线速度趋零；不是直接比较 Action 目标距离 |
| `NAV_FOLLOWER_LOOKAHEAD_M` → `lookAheadDis` | 正式 launch `1.0` m | 增大通常使跟踪更平顺，但弯道偏离和切角需检查；减小可能提高局部响应并放大抖动 |
| `NAV_FOLLOWER_SLOWDOWN_DISTANCE_M` → `slowDwnDisThre` | 正式 launch `1.0` m | 按局部路径末端距离 `endDis` 减速；当前短路径可能显著低于速度上限，不应把低速直接判断为限速故障 |
| `yawRateGain / stopYawRateGain` | 原生均 `7.5` | 角度误差控制增益。过大可能左右摆动/触及角速度上限，过小转向迟缓；先排除外参和定位抖动 |
| `twoWayDrive` | planner/follower 原生均 true | 允许前后方向选择；仅支持前进的新平台需两端一致修改，并验证换向行为与底盘接口符号 |
| `velocity_tolerance_mps / stable_reach_time_s` | lower launch `0.08 / 0.2` | 到达判定的速度阈值与稳定时长；结合低速工况和实际反馈定义，不宜只扩大距离阈值来宣布成功 |

**已离线确认旧参数存在提前停止条件，并完成入口修复。** 原生跟踪器的 0.20 m 停止阈值与旧 0.10 m Action 容差存在冲突；短路径跟踪点落在两者之间时，线速度趋零或只剩很小振荡，Action 仍未到达。固定直线案例中旧值最终 blocked；保持其他条件不变，仅把停止阈值改为 0.05 m 可进入旧到达容差。该 0.05 m 是诊断对照，不是最终配置。

**最终按用户指定：停止距离 0.5 m，到达容差 1.0 m。** 前视距离同步设为 1.0 m，确保大于停止距离；减速距离保留 1.0 m。主机入口和容器入口均在启动组件前检查四项距离为有限正数，要求 `stop < goal_tolerance` 且 `stop < lookahead`；不合法配置直接拒绝，不静默钳制。`--check` 显示有效值。直接绕过脚本运行 ROS launch 不执行这项入口检查。

三项 follower 参数已经过环境变量 → Docker 白名单 → 容器 launch 参数 → 原生 ROS 参数完整接线。修改后需要重建/更新部署镜像，重新启动才能生效；旧镜像 `896fbd15b040...` 不包含本次修复，不能把新源码的 `--check` 当作旧镜像已更新。

**距离语义：** 此处 1.0 m 到达容差针对 standoff 接近点，不是到椅子表面或中心的距离；standoff 仍为 0.8 m。理想共线情况下容差边界可能距对象中心约 1.8 m，不能由“数值对齐”推断与仿真对象成功距离完全等价。源码可核对仿真 `success_distance=1.0`，但仿真停止还使用其他比例/几何条件，本次 0.5 m 明确按用户要求设置。

接近点已在 1.0 m 容差内时，现有 resolver 不再派发短程目标，也不伪造语义成功。先前约 0.34 m 的接近目标因此可能不再产生 Action，这是新容差下的预期过滤。验证规划应选择接近点在容差外的目标，不能为输出轨迹强行派发。

### 停止距离离线诊断

以下合并自 2026-11-05 诊断记录，不是本次新跑结果。短路径的跟踪点可能成为路径末端；follower 的 `dis > stopDisThre` 与 Action 的目标距离判定独立，因此存在“跟踪器已减速、Action 尚未到达”的区间，也可能出现微小速度振荡。此前实车 bridge-off 的 `no_progress` 本身由未驱动车辆产生，不能反过来归因于此参数冲突。

对照测试直接抽取生产 C++ 的速度选择、转向限幅、减速与加速度步进代码编译，合成运动反馈交给实际 `RosNavigationStatusProvider`。假设 100 Hz 理想积分、1 cm 末端离散、0.75 m 局部路径上限、零航向误差、无障碍；不包含完整规划器、ROS/DDS、mux 或底盘动力学。进展阈值 0.05 m、无进展时限 12 s、到达速度阈值 0.08 m/s、稳定时间 0.2 s 保持不变。

| 初始目标距离 | 旧停止 0.20 / 到达 0.10 m | 诊断停止 0.05 / 到达 0.10 m |
|---|---|---|
| 0.15 m | 12.00 s blocked，余距 0.150 m | 3.00 s reached，余距 0.0971 m |
| 0.34 m | 26.14 s blocked，余距 0.1804 m | 8.55 s reached，余距 0.0970 m |
| 0.45 m | 17.70 s blocked，余距 0.1881 m | 10.47 s reached，余距 0.0970 m |

0.05 m 仅用于辨别冲突，不是交付配置。最终停止 0.5 / 到达 1.0 m、前视/减速 1.0 m 的同类测试结果如下：

| 初始目标距离 | 结果 | 到达判定时速度 |
|---|---|---|
| 1.2 m | 6.82 s reached，余距 0.5242 m | 0.0765 m/s |
| 1.6 m | 10.40 s reached，余距 0.5241 m | 0.0765 m/s |
| 2.0 m | 13.98 s reached，余距 0.5241 m | 0.0765 m/s |

`reached` 满足既有速度/稳定时间合同，不代表速度严格为零；Action 终止仍撤销目标并 hold，真实制动距离未测。位置固定、目标在容差外的负对照仍为 blocked；未修改控制算法、放宽碰撞检查或伪造进展。

历史记录为 69 项回归通过，涵盖上述对照、非法配置拒绝、主机/容器检查、Docker/launch 接线、容差内候选过滤及终止隔离。具备项目测试依赖的 Python 环境可在仓库根目录复现：

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q \
  -p no:cacheprovider --capture=sys \
  tests/test_navigation_stop_distance.py tests/test_real_navigation.py \
  tests/test_sysnav_ros_adapters.py tests/test_shadow_approach.py
```

原始输出保留于 `logs/diagnostics/navigation-stop-distance/pytest-requested-defaults.txt`，配置与源码哈希见同目录 `receipt.json`；本次仅合并文档，不覆盖这些证据，也不更新车端旧镜像。

## 4. 速度、加速度与时间保护

| 参数 | 正式值 | 调节依据 |
| --- | --- | --- |
| `NAV_MAX_SPEED_MPS` | `0.15` m/s | 同时约束原生速度、autonomy speed、mux 和 live bridge；这是上限，不是承诺实际输出速度 |
| `NAV_MAX_ANGULAR_RPS` | `0.35` rad/s | mux/bridge 角速度上限 |
| `NAV_FOLLOWER_YAW_DEG_S` | `20.0` deg/s | follower 上限，约 0.349 rad/s；与上一项取最严格限制，注意单位不同 |
| `NAV_MAX_ACCEL_MPS2` | `0.15` m/s² | follower/mux 的线加速度限制；减小更平缓但跟踪/制动过程变长 |
| `NAV_MAX_ANGULAR_ACCEL_RPS2` | `0.3` rad/s² | 角速度变化限制；结合底盘响应、地面摩擦调节 |
| `NAV_COMMAND_TIMEOUT_S` | `0.25` s | mux 与 live bridge 的指令超时；过小可能间歇停车，过大延长失联后的旧指令窗口 |
| `NAV_NO_PROGRESS_TIMEOUT_S` | `12` s | 无有效进展时终止 Action 并撤销原生目标。静止 preview 触发是预期结果，不能关闭它假装完成 |
| `min_progress_delta_m` | lower launch `0.05` m | 进展判定阈值；应高于定位噪声且适配最低真实速度。降低可能把抖动当进展 |
| `navigation_timeout_s / path_stale_timeout_s` | lower launch `60 / 5` s | 总执行时间和路径新鲜度；不是替代传感器/速度 watchdog 的保护 |
| `sensor_watchdog_timeout_s` | lower launch `0.5` s | 传感器新鲜度门限；先解决消息停更/时间戳问题，再决定是否调整 |

可用 `v × 延迟 + v²/(2a)` 粗估反应/减速距离以辅助选余量，但它不包含全部控制与硬件行为，也不是急停实测值。用 0.15 m/s、0.25 s、0.15 m/s² 代入约为 0.113 m，仅是简化算例；硬件制动、加速度限制、网络与传感器延迟都需实际测量。保护门控可能直接置零，不能把该算例当作所有停车路径的实现。

迁移时应先校准速度单位、正负方向和实际反馈，再决定是否提高速度；不能通过放宽 watchdog 掩盖 DDS/LIO 停更。当前只验证过一次容器重启后输入继续发布。

## 5. 感知与融合：先查几何和同步，再调阈值

当前候选 profile 使用 YOLOWorld、`SYSNAV_DETECTOR_IMGSZ=640`。修改模型、类别词或分辨率时同时复核耗时、椅子召回和误检，不应只用单帧能否出框判断可用目标。

融合节点源码同步默认值为 `cloud_sync_before_s=0.5`、`cloud_sync_after_s=0.1`、`cloud_sync_max_gap_s=1.25`、`odom_max_age_s=0.5`，`allow_latest_odom=false`；运行时仍需核对 launch/config 覆盖。扩大时间窗口可能提高配对数量，但运动时错位也会增加；缩小窗口可能减少错误融合但丢更多观测。用图像、检测、点云和 odom 时间戳核对，不能为了产生目标而使用过期位姿。

出现“椅子清楚但无目标”，依次检查检测框/掩码、深度或投影支持点、3D 中心/尺寸、融合 UID、前方/standoff/已尝试候选过滤。历史混合实例和碎片仍保留为局限；不要降低关联阈值强行合并或强行绑定。当前正式入口 `LLM_OFFLINE=1`、最终视觉 verifier 关闭；调感知参数不会自动启用豆包或验证视觉 STOP。

## 6. 后续接底盘要改什么

完整步骤以 [正式入口的 live 配置](real_robot_navigation_entry.md) 为准。关键配置为：

- `STRIVE_EXECUTION_MODE=live`，`ALLOW_REAL_MOTION=true`；默认保持 preview。
- `CONTROL_CONTRACT_FILE` 指向容器内真实控制合同，主机目录通过 `CONTROL_CONTRACT_HOST_DIR` 等既有挂载配置提供。
- `AGILEX_URI` 当前接口地址为 `ws://192.168.1.102:9090`，迁移时核对转接服务、协议和话题。
- 合同内 `allowed_control_modes`、`ready_base_states`、`ready_fault_codes`、`telemetry_timeout_s` 必须按新底盘实际协议确认；不要把未解释的 fault 128 加入 ready 白名单。
- 合同速度上限、世界系、Action/速度话题必须匹配正式 live 命名空间 `/strive/navigation`；最终速度由 safety mux 唯一拥有。
- live 初始 HOLD；未来获得现场运动授权且 ready 后才调用 `/platform/set_autonomy`。本轮没有执行该服务或启动 bridge。

新平台若不是现有 AgileX 转接协议，还需适配通信和状态语义；仅修改 IP 不能保证适配完成。

## 7. 调试顺序和最小记录

1. 静止核对时间戳、世界系、外参和车头方向；验证有效定位，避免以旧原点套用新车位。
2. 用真实点云检查车体外廓、细椅腿、地面裁剪与转动扫掠；迁移大车核对路径库。
3. preview 下检查 chair 中心、standoff 接近点、局部路径与限幅速度，确认终止后零速；不需要重复历史全图重放或模型调用。
4. 后续授权运动前解决停止阈值/到达容差配合，再按独立现场验收逐步检查低速跟踪、停车、接管与失联行为。不要一次同时改变几何、控制增益和速度。

只检查启动配置的示例（不启动容器或 bridge）：

```bash
STRIVE_EXECUTION_MODE=preview \
NAV_STANDOFF_M=0.8 NAV_GOAL_TOLERANCE_M=1.0 \
NAV_FOLLOWER_STOP_DISTANCE_M=0.5 NAV_FOLLOWER_LOOKAHEAD_M=1.0 \
NAV_FOLLOWER_SLOWDOWN_DISTANCE_M=1.0 \
NAV_MAX_SPEED_MPS=0.15 NAV_MAX_ANGULAR_RPS=0.35 \
NAV_FOLLOWER_YAW_DEG_S=20.0 \
NAV_MAX_ACCEL_MPS2=0.15 NAV_MAX_ANGULAR_ACCEL_RPS2=0.3 \
bash scripts/run_real_robot_navigation.sh --check
```

| 现象 | 优先检查 |
| --- | --- |
| 轨迹方向反了/目标被判在后方 | 底盘 yaw 外参、frame 与车头方向，不先改控制增益 |
| 有椅子但无候选 | 3D 支持、前方条件、standoff + 0.15 门限、容差和已尝试 UID |
| 有目标但无路径 | 点云坐标/高度裁剪、真实障碍、路径库 footprint，不先关碰撞检查 |
| 有路径但速度小/为零 | 路径长度与末端减速、角度误差、停止阈值、Action 终态、mux HOLD/新鲜度 |
| 靠近后不报到达 | 同时看 Action 目标距离和 follower 跟踪点距离，以及速度/稳定时间 |
| 频繁 no_progress | 是否实际允许运动、定位噪声、真实速度与进展阈值、目标是否受阻 |

每次保留：镜像 ID/代码版本、profile 和参数差异、地图/定位初值、目标 UID 与接近点、输入频率/时间戳、路径和最终速度、终止状态及零速持续时间。运动试验额外记录真实位置误差、净空、停车距离和接管结果。配置修改后的新结果应单独归档，不覆盖本次通过记录。

## 源码入口

- [主机启动入口](../scripts/run_real_robot_navigation.sh)、[容器入口及正式覆盖值](../scripts/start_real_robot_navigation.sh)、[Docker 环境变量白名单](../docker/run_real_robot_sysnav_stack.sh)
- [lower launch](../real_robot/ros2_ws/src/strive_sysnav_motion/launch/sysnav_lower_stack.launch.py)
- [接近点与底盘坐标转换](../real_robot/approach.py)
- [localPlanner](../real_robot/ros2_ws/src/local_planner/src/localPlanner.cpp)、[pathFollower](../real_robot/ros2_ws/src/local_planner/src/pathFollower.cpp)
- [融合同步参数](../real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/semantic_mapping_node.py)
