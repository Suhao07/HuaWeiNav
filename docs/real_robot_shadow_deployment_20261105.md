# 实物模式部署：bridge 关闭的规划与速度验收

> 已升级到 [正式实物模式一键入口](real_robot_navigation_entry.md)，实物模式已按 bridge-off 规划与速度输出口径完成。调参见 [实物参数调节说明](real_robot_parameter_tuning.md)。本文保留作此前 shadow/静止验收记录。

2026-11-05 用户将本轮完成口径调整为：真实机器人传感器输入，指定前方 chair，选择目标、产生合理规划轨迹、由跟踪器输出速度，默认不打开底盘 bridge。不再把历史精确实例绑定、豆包复测、实际行驶或最终视觉 STOP 作为本轮完成前置条件。

## 入口与输出

在运行真实感知/定位的容器中执行 `scripts/run_real_robot_shadow_acceptance.sh`。脚本读取生产 runtime 的最新目标 UID，在实时对象消息中查找同一椅子，检查底盘前方半平面，用可配置的距离产生接近候选，再交给原生 SysNav localPlanner、pathFollower 和 SafetyVelocityMux。目标中心本身不作为可到达位置。

这是一个有时限的 shadow 验收入口。候选不写入历史地图，不伪装为已验证视点，不强制绑定。原高层的历史视点解析路径仍保留；本入口不证明未知空间自主探索、全局可达性或最终到达。机器人不移动时不会模拟 odometry 前进，也不会宣称 REACHED。

输出：

| 话题 | 含义 |
| --- | --- |
| `/strive/shadow/base_odom` | 雷达定位与底盘外参组合的底盘 pose；仅供本次平面规划，twist/covariance 未估计 |
| `/strive/shadow/way_point` | 世界坐标系下的接近候选 |
| `/strive/shadow/path` | 原生规划器轨迹，`vehicle` 局部坐标系 |
| `/strive/shadow/raw_cmd_vel` | 原生跟踪器未经安全复用器处理的候选速度 |
| `/strive/shadow/cmd_vel` | 安全复用器限速/限加速度后的 TwistStamped，底盘局部坐标系 |
| `/strive/shadow/status` | 规划状态 |

脚本不启动 AgileX bridge 或 MotionServer。安全复用器仅订阅和发布隔离的 shadow 控制话题；传感器超时保护保持开启。检测到 AgileX bridge 或 shadow 速度话题有未知订阅者时拒绝验收；最终记录 `/cmd_vel`、`/cmd_vel/autonomy`、`/navflow_cmd_vel`、`/way_point` 发布者计数。不要把 shadow 话题直接接到硬件。

本台车命令（容器内，运行目录必须与当前会话一致）：

```bash
export SHADOW_DECISIONS=/workspace/STRIVE/output/runtime/shadow-static-20261105/runtime_decisions.jsonl
export SHADOW_RUN_DIRECTORY=/tmp/strive-shadow-acceptance
export SHADOW_BASE_X_M=-0.2 SHADOW_BASE_Y_M=0.0 SHADOW_BASE_YAW_RAD=-1.5708
bash /workspace/STRIVE/scripts/run_real_robot_shadow_acceptance.sh
```

输出目录保存 `result.json`、`planner.log`、`follower.log`。默认 35 秒后退出并清理子进程，感知/定位继续运行。重复运行使用新输出目录，避免覆盖证据。

## 迁移机器人需要调整的参数

| 参数组 | 本台设置/默认 | 迁移规则 |
| --- | --- | --- |
| 传感器话题 | `SHADOW_ODOM_TOPIC=/aft_mapped_to_init`、`SHADOW_CLOUD_TOPIC=/cloud_registered`；对象话题可用 `--objects` 指定 | 里程计和注册点云必须同一世界坐标系；不要把 body 点云冒充 world 点云 |
| 世界坐标系 | `SYSNAV_WORLD_FRAME=camera_init`；mapping 的 `world_frame` 参数 | 与里程计 header 一致；本轮修复了融合输出写死 map 的错误，错帧直接拒绝 |
| 雷达到底盘外参 | `SHADOW_BASE_X_M=-0.2`、`SHADOW_BASE_Y_M=0`、`SHADOW_BASE_YAW_RAD=-1.5708` | 含义是底盘在雷达坐标系中的 XY 与 yaw，单位 m/rad；新机器人必须按安装标定修改，已是底盘 odom 则用零偏置 |
| RGB/深度/雷达投影 | `PROJECTION_CONFIG`、相机内参、外参、时间同步参数 | 这是检测框到三维物体的基础，不可跨机器人照抄；使用既有 profile 与标定流程 |
| 车体尺寸 | `SHADOW_VEHICLE_LENGTH_M=0.6`、`SHADOW_VEHICLE_WIDTH_M=0.6` | 当前为规划器配置值，未作为实测整车外廓验收；包含突出物/裕量；换车必须实测确认，不能为出路径缩小 |
| 接近距离 | `SHADOW_STANDOFF_M=0.8` | 从融合物体中心沿视线退让，非物体表面净空；根据车体半径、目标尺寸与任务距离调节，候选仍需规划避障 |
| 线速度/加速度 | `SHADOW_SPEED_MPS=0.15`、`SHADOW_MAX_ACCEL_MPS2=0.15` | 单位 m/s、m/s²；shadow 验证配置，不等于底盘已经验收这些动态性能 |
| 角速度 | `SHADOW_MAX_YAW_DEG_S=20.0`，mux 上限 0.35 rad/s，`SHADOW_MAX_ANGULAR_ACCEL_RPS2=0.3` rad/s² | **此原生跟踪器参数用 deg/s**；底盘 API/安全 mux 通常用 rad/s，不可直接照抄数值 |
| 轨迹行为 | localPlanner `checkObstacle=true`、`checkRotObstacle=true`；pathFollower `lookAheadDis`、`stopDisThre`、`twoWayDrive` | 根据轮式运动学与制动距离调节；当前不是对任意 Ackermann/差速/全向底盘的自动适配 |
| 障碍高度/点云尺度 | localPlanner `minRelZ`、`maxRelZ`、`laserVoxelSize`、`pointPerPathThre` | 随雷达高度和环境配置。本入口沿用原生默认高度过滤，不可把生成一条路径当作完整环境安全认证 |
| 检测模型 | `SYSNAV_DETECTOR_MODEL_TYPE=yolo_world`、`SYSNAV_DETECTOR_IMGSZ=640`、模型路径 | 算力/内存决定尺寸；阈值按检出与误检调节；本轮目标来自在线 chair 检测与融合 |
| 时限/证据 | `SHADOW_DURATION_S=35`、`SHADOW_DECISIONS`、`SHADOW_RUN_DIRECTORY` | 决策必须新鲜、会话目录必须匹配，超时或不在前方不制造目标 |
| 底盘桥（默认关） | `START_AGILEX_PLATFORM=false`、`AGILEX_REMOTE_CONTROL_ENABLED=false`、`ALLOW_REAL_MOTION=false` | 真运动须沿用 Action→MotionServer→安全 mux→底盘 bridge，并填写 `CONTROL_CONTRACT_FILE`；不能仅把 shadow 速度重映射到硬件 |

## 后续真运动与本轮完成范围

实际接底盘时另行验证模式/故障码、控制选源、速度轴向与单位、限速/限加速度、指令超时、急停、取消与人工接管。未观测空间的探索策略、运动中的定位、真实到达后的新鲜视觉确认/STOP，继续保留为后续现场验收，不属于本轮 bridge-off 交付结论。

历史实例融合混合/碎片与豆包掩码解释修复保留为已知局限，不阻断本轮可见 chair 的 shadow 规划，也不标记为已经解决。

## 本次验收实测

35.01 秒，2833 条 odom、2828 条注册点云、16 条对象更新；目标 `sysnav_object:2` chair。
目标距离 0.427 m，轨迹最大 44 点/0.430 m；以目标发布时底盘 pose 转换的最佳路径端点距目标约 0.037 m（规划几何误差，不是实际到达误差）。2680 条有效轨迹，640 条安全复用器最终速度，其中 639 条非零；峰值线速度 0.066 m/s、角速度 0.113 rad/s。四个物理控制话题发布者全为 0。

原生 raw 转向速度有正负切换（峰值 0.349 rad/s），最终 mux 限速/限加速度后峰值降低。车辆未移动，不能据此评价真实闭环跟踪稳定性。未通过调小车体、关闭避障或伪造位姿推进来让测试通过。

本地证据在 `logs/diagnostics/shadow-acceptance/strive-shadow-final/`，原坐标系不一致与前方判断拒绝记录也保留在同级目录。原始只含跟踪器、未接安全 mux 的结果保留在 `strive-shadow-base-fixed/`，不作为最终输出验收。

本地测试：导航根因回归与最初 shadow 几何 26 项通过；增加外参组合后 shadow + safety 两组 13 项通过。修复了旧测试抽取融合函数时漏带已拆分依赖函数的测试夹具。未改融合算法。

## 固化版本与回滚

最终镜像 `huawei-vln-realworld:orin-r36.5-shadow-acceptance`，ID `38f67ba7e58a8579e3c9acb5b9654c9a4ce9f65fc532fe588cdb2eeead7bbf82`。当前静止容器 `huawei-vln-shadow-acceptance` 的基础镜像为 `ce6a4d560b8c`，两份 shadow 脚本已覆盖为最终验收版本；最终镜像固化相同脚本，详见 `logs/diagnostics/shadow-acceptance/delivery_receipt.json`。

Orin 持久化启动脚本：`/home/orin26/HuaweiVLN_deployments/yoloworld_candidate_71f340c5/shadow-acceptance-build/start_static.sh`。该脚本只启动静止感知/runtime，默认不开 bridge。shadow 验收按上文命令单独运行。

旧服务数据已保存到部署根目录 `logs/pre-shadow-acceptance/`。需要回滚时先导出当前 `/workspace/STRIVE/output/runtime/shadow-static-20261105`，停止新容器，再运行原 `correspondence-candidate-build/start_static.sh`。保持定位/相机服务不动；任何 runtime 重启都会更换会话身份，旧视觉包不迁移为新证据。
