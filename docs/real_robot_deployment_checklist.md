# Orin-26 实物部署实时 Checklist

> 最后更新：2026-08-11
> 工作区：`/home/orin26/HuaweiVLN`
> 代码分支：`realworld`
> 原则：不修改其他工作区、不重启现有 LIO、STRIVE 不直接发布 `/cmd_vel`。

本文档是实物部署的执行记录，而不是设计愿景。只有有明确命令输出、测试结果或
落盘工件的项才可勾选。任何失败都保持在 dry-run 或 detector-only 阶段。

## 1. 当前硬件与环境基线

- [x] Orin：Ubuntu 22.04.5、JetPack 6.2 / L4T R36.5、aarch64。
- [x] Docker 29 与 `nvidia-container-runtime` 可用。
- [x] 已创建独立工作区 `/home/orin26/HuaweiVLN`，并 checkout `realworld`。
- [x] 已确认远端 `origin/realworld` 为 `7fdf2c8e03f5e422771d4dd6148a950860940e14`。
- [x] 现有 `/home/orin26/code/HuaWeiNav` 含未提交改动；本部署不修改它。
- [x] ROS graph 可发现 Point-LIO 的 `/cloud_registered`、`/aft_mapped_to_init`、`/base_odom`、`/path` 发布端点。
- [x] ROS graph 可发现 Livox MID-360 的 `/livox/lidar`、`/livox/imu` 发布端点。
- [x] 已识别 Generic USB RGB 相机（稳定路径见 profile）及 Intel RealSense D435i。
- [x] 已确认当前没有活跃 `/way_point`、`/cmd_vel` 或经验证的下层控制器。
- [x] 2026-08-11 Point-LIO 的只读 tmux 日志确认正在处理首帧 LiDAR/IMU 并持续输出 mapping 时延；未修改该 session。
- [x] 2026-08-11 只读核对 `/dev/video0`：udev 身份为 Generic USB Camera（VID `0bda`、PID `3035`、序列号 `200901010001`），`/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0` 指向该设备。
- [x] 2026-08-11 只读核对当前 ROS graph：`/depth_camera_adapter`、`/laserMapping`、`/livox_lidar_publisher`、`/tf_aft_mapped_to_base` 存在；没有控制节点。`/depth_camera_adapter` 仅订阅 `/camera/camera/depth/image_rect_raw`，发布 `/depth_camera`，frame=`depth_camera`，输出 32×24、范围 0.05–2.5 m。
- [ ] 单独 ROS CLI 订阅 `/livox/lidar`、`/livox/imu`、`/cloud_registered`、`/aft_mapped_to_init` 在匹配的 Reliable/Best-Effort QoS 下仍会超时，需在启用融合前解决外部 DDS data-plane 接收并记录频率、header frame、timestamp。
- [x] 已提供 `lio-diagnostics` profile 子命令：只读采集 ROS/DDS 环境、LIO endpoint QoS 与实际 header 样本，并将报告仅写入本工作区 `logs/diagnostics/`。
- [x] 2026-08-11 已生成 `logs/diagnostics/lio_dds_20260811T060757Z.md`：host Fast DDS 默认 transport、domain 0 下 Point-LIO 参数服务可读，但四个实际 header 样本均在 8 秒内超时；保持 `START_SEMANTIC_MAPPING=false`，不重启外部 `livox_odom`。
- [x] 2026-08-11 最新只读诊断 `logs/diagnostics/lio_dds_20260811T095246Z.md`：Point-LIO/Livox 进程仍在，四个 topic 的 publisher/subscriber 端点可发现，但 `/livox/lidar`、`/livox/imu`、`/cloud_registered`、`/aft_mapped_to_init` 的实际 header 样本再次全部在 8 秒内超时；`publish.scan_publish_en=false`、`scan_bodyframe_pub_en=false`，继续保持 `START_SEMANTIC_MAPPING=false`。
- [ ] 已确认运行中的外部 Point-LIO 参数 `publish.scan_publish_en=false`，这会使 `/cloud_registered` 保持可发现却不输出实际点云；需由其所有者确认并在其工作流中启用，再重新运行本工作区的只读诊断。Livox 原始 IMU 与 LIO 里程计仍须同时通过实际 header 门槛；仅开启 scan output 不能替代该验收。
- [x] 容器专用 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` 已与 host-side LIO 诊断分离：diagnostic 默认恢复外部 Point-LIO 的 host Fast DDS transport，避免把容器 workaround 错用于参数/传感器验收。
- [ ] 为本次 deployment 保存机器/代码/镜像/资产 SHA256 清单。

## 2. 隔离边界

- [x] 代码、输出、日志、runtime 决策文件均限制在 `/home/orin26/HuaweiVLN`。
- [x] 新容器名为 `huawei-vln-realworld`，新镜像标签为 `huawei-vln-realworld:orin-r36.5`。
- [x] profile 默认 `START_LIO=0`、`MANAGE_HOST_LIO=false`、`RESTART_POLICY=no`。
- [x] 容器默认不挂载相机设备；只有 `START_USB_CAM=true` 才挂载声明的设备。
- [x] 不使用 `--privileged`，模型通过只读 volume 注入。
- [x] smoke 不再 stop/start 宿主共享 ROS daemon。
- [x] 2026-08-11 只读资源检查：根分区 233G/206G（94%），可用约 15G；Docker 镜像 20 个、约 25.36G；部署容器当前不存在，未执行 prune。
- [x] 2026-08-11 存储审计已落盘至 `docs/real_robot_storage_audit_20260811.md`，记录项目、bag、模型、缓存和 Docker 占用；未删除或移动数据。
- [x] 2026-08-11 17:47 只读复查：根分区 233G/108G（49%），可用约 113G；`Policy_part/bags` 从约 98G 变为约 20K，Docker 镜像仍为约 25.36G。
- [ ] 构建前确认至少有足够的 Docker 构建余量；当前余量偏紧，先不构建，不得对其他镜像执行 prune。
- [ ] 新容器启动后确认不会重启、停止或改变 `livox_odom` tmux session。

## 3. 资产与秘密

- [x] 已建立 `assets/` 约定：权重不纳入 Git，不从其他工作区以软链形式引用。
- [x] 已将 `yoloe-11s-seg.pt`、`sam2.1_hiera_base_plus.pt`、`mobileclip_blt.pt`、`mobileclip_blt.ts`、`ViT-B-32.pt` 复制到本工作区，不使用链接。
- [x] 已记录 SHA256：`yoloe=8e439445…ce3e3`、`sam2=a2345aed…004c5`、`mobileclip_pt=670844f7…27441`、`mobileclip_ts=a67804d1…95e54`、`clip=40d36571…950af`。
- [ ] 创建 `.env.realworld`（权限 600），仅存必要 API key；不提交、不打印。
- [x] 已通过 `scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb check` 验证资产与稳定相机路径。

## 4. 可插拔机器人 profile

- [x] 已添加 `orin26_livox_mid360_generic_rgb.env`，集中维护 ROS topic、模型、相机、容器和安全开关。
- [x] 已添加 `scripts/run_real_robot_profile.sh`，统一提供 `check/build/smoke/start/status/stop/logs`。
- [x] profile 能在 Bash 中直接 source 时定位自身工作区根目录；profile 启动器默认不会再级联加载通用 `.env.realworld`，避免其它机器人遗留的 topic、相机或容器名覆盖当前 profile；只有显式设置 `SYSNAV_ENV_FILE` 才会加载补充文件。
- [x] 内部输出隔离到 `/huawei_vln/detection_result`、`/huawei_vln/object_nodes_list`。
- [x] 外部 LIO 输入保持为只读 `/cloud_registered`、`/aft_mapped_to_init`、`/path`。
- [x] Generic UVC 能力已读取：1920×1080 MJPEG 30 FPS；1280×720 YUYV 10 FPS。profile 显式使用后者（`USB_FRAMERATE=10.0`）。
- [x] 2026-08-11 独立 `usb_cam` smoke 已打开 `/dev/video0`，使用 1280×720 YUYV 10 FPS；验证结束后无残留容器或节点。
- [x] 2026-08-11 容器内 `/camera/image` 已收到 header（`frame_id=default_cam`）；preflight 与 launch 的相机参数一致。
- [x] 2026-08-11 容器内 `/camera/image` 实测编码为 `yuv422_yuy2`，20 帧窗口稳定到 `9.946 Hz`（profile 输入为 1280×720 YUYV 10 FPS）。
- [x] 2026-08-11 相机→YOLOE detector 闭环已验证：`/huawei_vln/detection_result` 收到 `frame_id=map` 的真实时间戳和 track ID；空检测帧不会再使 `detection_node` 退出。
- [x] 2026-08-11 只读检查其他项目历史配置：旧 `tools/usb_camera_node.py` 使用 `/image_raw`、640×480 MJPEG 30 FPS；它与当前已实测的 `usb_cam` profile 不同，不作为本部署的启动配置。
- [x] `USB_CAMERA_INFO_URL` 已成为可插拔 profile 参数；`real_robot/calibration/` 由容器只读挂载。标定后只需放入该目录并填写 `file:///workspace/STRIVE/real_robot/calibration/<camera>.yaml`，profile check 会验证文件存在。
- [ ] 为 usb_cam 提供你标定后的 `camera_info` 文件；在该文件就绪前启动日志会提示缺少 `/root/.ros/camera_info/default_cam.yaml`，但不影响当前 detector-only 验收。
- [ ] RealSense D435i driver、RGB-D topic 和 device 权限按独立 profile 验证。

## 5. 标定与传感器融合

- [x] 已确认旧 `mecanum` 投影固定为 1920×640 全景相机，不能用于当前 Generic RGB/D435i。
- [x] 新增可验证的 `pinhole` / `equirectangular` 投影配置模块。
- [x] 未经批准的 `calibration_status` 会拒绝 semantic mapping。
- [x] Orin profile 默认 `START_SEMANTIC_MAPPING=false`，允许安全 detector-only bringup。
- [x] detector-only profile 不依赖 LIO 数据；启用 semantic mapping 时 helper 强制要求 LIO topic 与实际 header 样本双门槛。
- [ ] 采集 RGB 内参（分辨率、`fx/fy/cx/cy`、畸变）。
- [ ] 标定 MID-360 到 RGB optical frame 的外参和平移单位。
- [ ] 记录时间偏移与投影重投影误差。
- [ ] 将校准后的 profile 标记为 `calibrated` 并启用 semantic mapping。
- [ ] 使用离线 bag 验证点云-掩码融合结果，再接入实时传感器。

### 外参/内参填写入口（暂不启用融合）

你只需填写
`real_robot/ros2_ws/src/semantic_mapping/config/projection_orin26_generic_rgb_template.yaml`
中的以下参数：

| 字段 | 含义 | 单位 / 约定 |
| --- | --- | --- |
| `image.width`, `image.height` | 实际 RGB 图像尺寸 | pixel |
| `intrinsics.fx/fy/cx/cy` | 已校正 RGB pinhole 内参 | pixel |
| `distortion` | 原始相机畸变记录；运行时输入应为 rectified RGB | — |
| `lidar_to_camera.translation_m` | MID-360 原点在 RGB optical frame 中的位置 | metre |
| `lidar_to_camera.rotation_rpy_rad` | `Rz(yaw) @ Ry(pitch) @ Rx(roll)` | radian |
| `rgb_minus_lidar_time_offset_s` | RGB timestamp 减 LiDAR timestamp | second |
| `validation.*` | 标定方法、日期、RMSE、样本数 | report metadata |

变换约定固定为 `p_camera = R_lidar_to_camera @ p_lidar + t_lidar_to_camera`。
在你确认内外参、时间偏移和重投影误差前，保持
`calibration_status: uncalibrated` 和 profile 的
`START_SEMANTIC_MAPPING=false` 不变；因此填写参数本身不会启动融合或控制。

## 6. Docker 与 ROS 验收

- [x] 已构建 ARM64 ROS overlay 镜像 `huawei-vln-realworld:orin-r36.5`；当前 image ID 为 `sha256:0da648cc2028…`；使用独立代码副本与只读基础镜像，不挂载旧工作区。
- [x] 已验证容器内 ROS Humble、`tare_planner`、`semantic_mapping`、`strive_sysnav_bringup` import/overlay。
- [x] 已验证 CUDA/torch/Ultralytics/Open3D 等依赖导入，`torch.cuda.is_available()=True`。
- [ ] 验证容器经 host DDS 只读接收 LIO 点云与里程计。
- [x] 2026-08-10 detector-only 容器已启动验证：`detection_node` 仅发布 `/huawei_vln/detection_result`，模型为 YOLOE。
- [x] 已确认该容器启动时 `/cmd_vel`、`/way_point` 均不存在；日志显示 lower controller 被 blocked-control gate 阻断。
- [x] 验证后已仅停止/删除 `huawei-vln-realworld` 容器，保留构建与 smoke 日志。
- [x] 最终镜像内的投影配置测试通过（`3 passed`，使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 避开基础镜像预装 pytest/anyio 插件不兼容）。
- [x] 最终镜像内 `sysnav_detection_mapping.launch.py --show-args` 已暴露 `usb_camera_info_url` 参数；验证过程使用 `--network none` 临时容器。
- [ ] 校准完成后启动 semantic mapping，确认 `/huawei_vln/object_nodes_list`。
- [x] 2026-08-10 已启动 `START_STRIVE_RUNTIME=1`、`STRIVE_DRY_RUN=true`；`RuntimeDecision` JSONL 持久化在 `output/runtime/dryrun_20260810_c/`，因缺少 object/pose/image 正确输出 `WAIT`。
- [x] 修复 ROS Humble runtime node 对内建 `use_sim_time` 的重复声明；现在 profile 提供 `runtime-smoke` 以重复验证上述安全链路。

## 7. 控制闭环门槛

- [x] 代码层只允许 waypoint handoff；不直接发布 `/cmd_vel`。
- [x] profile 默认 `BLOCK_LOWER_CONTROLLER=1`、`ENABLE_LOWER_CONTROLLER=0`、`STRIVE_DRY_RUN=true`。
- [x] 真实运动要求同时设置 `ALLOW_REAL_MOTION=true`、`STRIVE_DRY_RUN=false`、`STRIVE_LOWER_CONTROLLER_ENABLED=true`、`ENABLE_LOWER_CONTROLLER=1`、`BLOCK_LOWER_CONTROLLER=0`。
- [x] 已新增只读挂载的 `real_robot/control/` 契约模板；即便有人打开上述开关，profile 与 runtime 仍要求经过批准的 controller contract，且明确禁止直接 `/cmd_vel` 发布并确认急停。
- [x] 2026-08-11 在 `--network none` 临时容器中强制打开 lower-controller 开关；因 `CONTROL_CONTRACT_FILE` 为空被拒绝（exit 4），控制命令没有运行。
- [x] 2026-08-11 只读核对 `/home/orin26/code/Urban-Nav-SR/Policy_part`：`/waypoint` 为 `std_msgs/Float32MultiArray`，`/topoplan/reached_goal` 为 `std_msgs/Bool`，PD controller 发布 `geometry_msgs/Twist` 到 `/cmd_vel`；配置上限为 `max_v=1.5 m/s`、`max_w=0.5 rad/s`，waypoint 数据按 ego-frame 二维点处理但消息本身没有 frame 字段。
- [x] 2026-08-11 只读核对 AgileX 桥接：输入 `/cmd_vel`，rosbridge 默认 `ws://192.168.1.102:9090`，输出 `/navflow_cmd_vel`（`geometry_msgs/Twist`），mux 相关为 `/mux_vel/add`、`/mux_vel/select`、`/mux_vel/selected`（`std_msgs/String`）和 `/base_cmd_vel`；历史命令曾使用 `--max-linear 1.5 --max-angular 0.45`，未执行。
- [x] 2026-08-11 只读核对控制运行态：无 PD/bridge/mux 进程、无 `/tmp/navflow_cmd_vel_bridge_enabled`、无本机 9090 监听；唯一 tmux 会话为外部 `livox_odom`，未修改。
- [x] 2026-08-11 只读发现状态辅助接口：`/odom`（`nav_msgs/Odometry`）、`/interface_management/BMS_status`（`tools_msgs/RobotBmsStatus`）、`/sensor_status`（`tools_msgs/SensorStatus`）；源码中未找到可批准的急停 topic/service。
- [x] 2026-08-11 已同步观测版 `real_robot/control/orin26_controller_contract.yaml`；该文件明确 `approval_status: unapproved`，仅记录外部接口事实，不满足真实运动门控，也不进入 Git。
- [x] 2026-08-11 隔离 `--network none` 容器中的 `RosWaypointController` 测试通过（8 passed）；验证 `geometry_msgs/PointStamped`、frame/坐标写入、STOP 不发布和禁止 `/cmd_vel` 直连。
- [ ] 外部 `/waypoint`（`Float32MultiArray`、无 header）与 STRIVE `/way_point`（`PointStamped`）存在接口不匹配；需要所有者确认或提供显式 adapter 后，才能进行真实下层 handoff。
- [ ] 确认底盘/局部规划器的启动所有权、订阅 topic、消息类型、frame、状态回执与急停接口。
- [ ] 先在 `/strive/test_way_point`（`geometry_msgs/PointStamped`）做无人订阅的消息与坐标系验证；该 topic 不得连接底盘控制器。
- [ ] 在人工监控、急停可达、限速和受控场地中验证真实 `/way_point` handoff。
- [ ] 验证 REACHED/BLOCKED/TIMEOUT/PREEMPTED、HOLD 与故障降级行为。

## 8. 回滚

- [x] 回滚仅允许停止/删除 `huawei-vln-realworld` 容器，保留日志与工件。
- [x] 禁止删除其他 Docker 镜像、容器、LIO session、bag 或工作区。
- [ ] 执行一次 dry-run 容器回滚演练并记录命令与结果。

## 9. 迁移到其他机器人时必须重新验收

不要复制本 profile 后直接启用 mapping 或运动。每一台机器人都必须有自己的
`real_robot/profiles/<robot>.env`、标定 YAML、日志目录和容器名，并重新完成以下
检查：

1. 用 `/dev/v4l/by-id/`、序列号或 udev 规则绑定相机；不要把易变的
   `/dev/videoN` 当作宿主机身份。
2. 记录 ROS_DOMAIN_ID、RMW/DDS transport、每个输入 topic 的消息类型、QoS、frame
   和时间戳来源；跨机 DDS 可见不等于传感器数据在流动。
3. 对每个 LiDAR--相机组合重新测量内参、外参、单位和时间偏移；只有重投影误差
   达标后才将 `calibration_status` 改为 `calibrated`。
4. 在启用 semantic mapping 前，验证点云和里程计都能收到实际 header 样本；仅发现
   publisher 时不得启动融合。
5. 控制接入前，书面确认底盘控制器所有权、目标 topic/消息类型/frame、速度/加速度
   上限、状态回执、超时、急停和人工接管流程。没有这些契约时始终保持
   `BLOCK_LOWER_CONTROLLER=1`。
6. 为新机器人生成独立资产 SHA256 清单、容器 image ID、代码 commit、启动日志和
   回滚命令；模型权重始终只读挂载且不提交 Git。

## 日常命令

```bash
cd /home/orin26/HuaweiVLN

# 检查 profile 和资产，不启动任何 ROS node。
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb check

# 构建隔离镜像；不访问底盘或宿主 LIO。
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb build

# 只读 smoke；profile 默认不访问相机，也不发布运动命令。
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb smoke

# 只读记录 LIO/DDS 证据；不会启动、停止或修改现有 Livox/Point-LIO。
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb lio-diagnostics

# 仅验证 Docker/ROS/GPU/模型依赖；不创建 LIO 的 ROS CLI 采样订阅，
# 不能替代上一条传感器验收。
bash -lc 'source real_robot/profiles/orin26_livox_mid360_generic_rgb.env; \
  SYSNAV_ENV_FILE=/dev/null CHECK_LIO_SAMPLES=0 REQUIRE_LIO=0 bash docker_en.sh smoke'

# 验证高层 runtime 的 dry-run 决策会保留在本工作区 output/；
# 不会启动传感器、LIO、语义融合或控制器。
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb runtime-smoke

# 运行可插拔投影配置单元测试。基础镜像的系统 pytest 与 anyio 插件版本
# 不匹配，因此显式关闭第三方 pytest 自动加载；这不影响运行时依赖。
docker run --rm --network none --entrypoint bash huawei-vln-realworld:orin-r36.5 \
  -lc 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_real_robot_projection_config.py'

# 完成各阶段门槛后才启动；初始 profile 仍是 detector-only + dry-run。
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb start

# 仅停止本次容器。
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb stop
```
