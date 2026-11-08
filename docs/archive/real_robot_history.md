# 实物部署历史摘要（2026-08—09）

> **历史证据摘要，不是当前部署指令。** 本文合并九篇早期记录，包括其中的 8 月 14 日、26 日补记；“通过”“未验收”和参数值仅描述当时版本及测试窗口，不代表现在的机器人状态。
> **实现完成不等于运动验收。** 数据闭环、离线测试、影子 adapter 和模拟里程计 HIL 均不能证明真实行驶安全，也不授予底盘控制权。
> 当前操作请查阅 [导航入口](../real_robot_navigation_entry.md)、[参数手册](../real_robot_parameter_tuning.md) 和 [部署状态与待验收事项](../real_robot_deployment_todo.md)。本文不保留过时启动命令长抄。

## 原文恢复与章节对照

本次合并实际 Git HEAD（原文恢复基线）：`7e7b4af1fa1c086c938286fa22475ceef7007e40`。
九篇完整版保存在该提交的 **`docs/` 原路径**，不是 `docs/archive/`；本目录只保留摘要，不声称保留全部原文。
只读恢复方式：`git show 7e7b4af1fa1c086c938286fa22475ceef7007e40:docs/<原文件名>`。

| 原文件名（均位于基线 `docs/`） | 摘要章节 |
| --- | --- |
| `real_robot_storage_audit_20260811.md` | [存储与环境](#storage) |
| `real_robot_runtime_evidence_20260812.md` | [感知与标定](#perception)、[定位与重连](#localization)、[控制安全](#safety) |
| `real_robot_calibration_acceptance_20260813.md` | [感知与标定](#perception)（含 8/14、8/26 复核） |
| `real_robot_reconnect_evidence_20260813.md` | [感知与标定](#perception)、[定位与重连](#localization)、[控制安全](#safety) |
| `real_robot_d435i_dataflow_evidence_20260813.md` | [感知与标定](#perception)、[定位与重连](#localization) |
| `real_robot_lio_resource_diagnostic_20260813.md` | [定位与重连](#localization)（含 8/14 根因与复测） |
| `real_robot_reconnect_evidence_20260819.md` | [感知与标定](#perception)、[控制安全](#safety) |
| `real_robot_20260821_no_motion_evidence.md` | [定位与重连](#localization)、[无运动与隔离验证](#validation) |
| `real_robot_status_evidence_20260923.md` | [存储与环境](#storage)、[控制安全](#safety)、[无运动与隔离验证](#validation) |

下述日志路径均是历史机器/工作区证据位置，不保证文件如今仍在线；运行日志不纳入 Git。
本次仅合并文档，不删除、搬移或压缩任何 bag、日志、模型、镜像或运行数据。

<a id="storage"></a>
## 1. 存储与环境

- **8/11 初次只读审计**：根文件系统总量 233G，已用 206G、可用 15G（94%）；`/home/orin26` 168G，其中 `code` 112G。
- `/home/orin26/code/Urban-Nav-SR` 约 110G，主要是 `Policy_part/bags` 98G（staging 48G、auto_import 51G）；ONNX 资产约 5.4G，可视化约 3.7G。
- `/home/orin26/VEOcc-Rywang` 约 37G，其中标定 `calibration/d435i-v009` 15G、`runs` 15G、runtime 4.6G、deployments 2.6G。
- `/home/orin26/HuaweiVLN` 约 1.9G、原 `/home/orin26/code/HuaWeiNav` 约 1.8G；模型副本和隔离部署不能直接视为可删冗余。
- 大文件证据包括 `Urban-Nav-SR/Policy_part/bags/auto_import/approved/nav_20260602_235343/*.mcap`（18.6G）与 `VEOcc-Rywang/runs/postmount-long-01/raw/raw_0.db3`（9.0G），均按 `/home/orin26` 下相应项目解释。
- Docker 报告 20 个可计费镜像，共享后约 25.36G；7 个可直接引用 VEOcc tag。完整构建链的 38 条记录含 31 条无标签中间层，不能当成独立镜像累加。
- 当时 `huawei-vln-realworld:orin-r36.5` 逻辑大小 7.18G、独有层约 1.68G；停止容器合计约 586M，build cache 为 0B，无该项目运行容器。
- **8/11 17:47 CST 只读复查**：根分区变为已用 108G、可用 113G（49%）；Urban-Nav-SR 约 12G，bags 约 20K，仅剩目录结构，约 98G 空间变化主要来自历史 bag。
- VEOcc、HuaweiVLN 与 Docker 占用未见相应减少；审计本身没有执行清理。其他项目保留集、备份和镜像回收需所有者批准，不能从本摘要推导删除许可。
- **9/23 构建后**根分区可用 64,044,326,912 字节（约 59.65 GiB），使用率 73%；构建前仅有 `df -h` 的 60G 粗粒度记录，无法精确归因磁盘增量。本轮未清理旧镜像、工作空间或数据。

<a id="perception"></a>
## 2. 感知与标定：数据通路和几何验收分开

### 相机、外参与数据闭环

- **8/12 Generic USB smoke**：稳定设备路径 `/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0`；1920×1080 MJPEG 配置 30 fps，实测约 24.9 Hz，输出 `/camera/image`、`rgb8`、frame `default_cam`。
- 该历史 profile 使用 `mjpeg2rgb`；字面 `mjpeg` 不是合法 ROS encoding，曾导致打开设备后退出。内参为 `fx=fy=749.2058, cx=1003.1, cy=526.5258`，模型 `radial_3`。
- D435i serial `233522079589`；8/13 host 驱动使用 RGB-D 同步和 depth 对齐 color、1280×720@30 Hz，HID IMU 禁用，Point-LIO 使用 Livox IMU。
- 实时 CameraInfo 为 `d435i_color_optical_frame`、`plumb_bob`、零畸变项；`fx=909.4949951, fy=909.1580200, cx=626.6063843, cy=370.9116516`。这些是该分辨率的运行观测，不能混用历史 640×480 内参。
- D435i 初期 topic 为 `/camera/d435i/color/image_raw`、对应 CameraInfo 和 aligned depth；8/14 后续记录修正为 `/camera/d435i/d435i_camera/...`。这是日期差异，不是固定当前 topic 约定。
- 8/13 detector-only 容器收到 chair、cabinet、desk、tv_monitor 等检测。未启用的 USB 参数曾使用只读 `camera_x001_intrinsics.yaml` 占位，D435i 实际内参仍取实时 CameraInfo。
- 外参资产为 `real_robot/calibration/orin26_d435i_mid360_targetless_v009_r009_extrinsics.json`，calibration id `d435i-mid360-targetless-v009-r009`；投影配置为 `real_robot/ros2_ws/src/semantic_mapping/config/projection_orin26_d435i_mid360.yaml`。
- 约定 `p_camera = R_lidar_to_camera @ p_lidar + t`，`T_camera_from_lidar` 的平移 `[0.10970761, 0.67284314, 0.27480904] m`，`rpy_zyx_rad=[-1.935569001, 0.007663065, -3.131491315]`。
- `det(R)≈0.999999997`、正交误差约 `7.9e-9` 只说明矩阵数值自洽。D435i 外参不得复制给 Generic USB 的 `default_cam`，不得自动广播未经确认的相机静态 TF。
- **8/13 隔离 mapping smoke**打通 RGB + CameraInfo + detection + `/cloud_registered_body` + `/aft_mapped_to_init` → object nodes；短时图像约 18–19 Hz、body cloud 1.4–1.9 Hz、检测 2.2 Hz。
- 融合样例：2223 个 cloud points、106 个投影入图，mask hits `[8,13,8,1,11]`；输出 cabinet/chair 的对象点云分别约 37/46 点。
- 但同期 odometry 约 `5e5–8e5`，对象 position/bbox3d 继承异常量级；**消息通路通过，几何失败，不可用于导航或 waypoint 规划**。
- 稀疏点曾因固定腐蚀 2 次出现 mask 命中但融合为零；测试采用 body 输入、`filter_depth_jumps=false`、`mask_erosion_iterations=0`，并不构成当前推荐调参。
- **8/19 重建复测**：body cloud 9.4–9.6 Hz、odom 约 100 Hz；12 秒收到 4 条对象消息（原记录约 0.5 Hz），frame `map`，chair/desk/cabinet 坐标回到米级，无邻近点云超时。
- 上述 smoke/复测曾临时开启 mapping，只证明数据链；`extrinsics_only` 未因此升级，不能解释为允许长期启用语义建图。

### 时间偏移、投影误差与负结果

- 原外参 JSON 不含 RGB–LiDAR 时间偏移；`rgb_minus_lidar_time_offset_s=0.0` 与 `time_offset_status=assumed_zero_unvalidated` 是未验证占位，非实测零延迟。
- 8/13 使用既有 `/home/orin26/VEOcc-Rywang/runs/debug-motion-02/raw`：约 29.5 秒，`/base_odom` 最大位移约 2.507 m，含 RGB、aligned depth、Livox LiDAR/IMU 与里程计。
- 历史 640×480 内参 `fx=606.330017, fy=606.105346, cx=311.070922, cy=247.274445`；5 组 7×6 棋盘角点 PnP RMSE 为 0.307、0.291、0.401、0.697、0.262 px。
- PnP 只验证相机观测/内参；无 LiDAR 点与棋盘角点一一对应，**不能替代 LiDAR–相机像素投影误差**。OCC 的平面/法向残差和 overlay 也不能替代它。
- 时间偏移定义 `Δt=t_RGB−t_LiDAR`；按候选取 LiDAR packet，用 odometry 补偿到 RGB 时刻后与 aligned depth 比较。bag 接收时间差不能识别曝光/驱动的物理时延。

| 复核日期与输入 | 主要测量 | 验收解释 |
| --- | --- | --- |
| 8/13 `debug-motion-02` | 候选 +0.040 s；60 对帧、350 有效对应；median 0.271 m、P90 1.212 m | `unidentifiable / not accepted`，禁止写入正式配置 |
| 8/14 同 bag 交替 held-out | train −4 ms；train/held-out median 0.230/0.283 m，RMSE 0.655/0.631 m，P90 1.248/1.095 m | held-out median 超过当时 0.25 m 工作阈值，`pass=false` |
| 8/14 对应数 | train projected/valid/inlier=193/192/176；held-out=197/195/175，inlier ratio 0.917/0.897 | 仍不是棋盘边缘像素误差 |
| 8/26 动态 bag | 训练候选 +0.090 s；2301 有效对应，median 0.045 m、RMSE 0.256 m、P90 0.464 m、inlier 0.996 | 阈值式 `pass=true` 不代表时延可辨识 |
| 8/26 train/held-out | 对应数 1148/1153，median 0.043/0.048 m，RMSE 0.246/0.266 m，P90 0.416/0.477 m，inlier 0.997/0.994 | held-out 自身最低点 −0.120 s，多偏移 median 近似；+0.090 s 不得写入 profile |
| 8/26 8 个静态场景 | 184546 有效对应；aggregate RMSE 0.432 m、median 0.063 m、P90 0.591 m；`abs(error)<0.20 m` 比例 0.746 | RGB-D 深度一致性，不是真正 LiDAR-to-image 像素 RMSE（仍未测量） |

- 8/26 输入：`/home/orin26/HuaweiVLN/logs/calibration/d435i_mid360_dynamic_20260823T062132Z/dynamic_bag`，167.58 秒、2507 帧 aligned depth、16723 个原始 LiDAR packet、16731 个 odometry 样本；使用 bag 内 640×480 CameraInfo。
- 该复核使用原始 `/livox/lidar`，不是 `/cloud_registered_body`；结果日志：
  - `/home/orin26/HuaweiVLN/logs/calibration_eval/dynamic_20260823T062132Z/time_offset_raw_lidar.json`
  - `/home/orin26/HuaweiVLN/logs/calibration_eval/static_v009_r009_imported/depth_metrics.json`
- 早期脚本为 `scripts/evaluate_d435i_lidar_calibration.py` 与 `scripts/evaluate_rgb_lidar_time_offset.py`，日志根 `/home/orin26/HuaweiVLN/logs/calibration_eval/`；只读评估不创建控制 publisher。
- OCC 可复用输入还包括 `/home/orin26/VEOcc-Rywang/calibration/d435i-v009/initial-solve-v009-r004/calibration-result.json`，复用不等于验收。
- 历史补采建议是底盘静止、人工移动棋盘，记录 sensor header timestamp；至少 3 个独立动态片段、每段 ≥1000 个有效跨传感器对应，分离估计和 held-out 序列。
- 仍需唯一稳定且跨片段复现的偏移极小值、真正投影 median/P90 像素误差和深度残差、标定 id/分辨率/畸变/日期/样本数；阈值由负责人确认。直到这些条件满足，历史结论仍是 `extrinsics_only`，不能批准正式 semantic mapping。
- 8/14 增加的位姿质量门拒绝非有限值、`max(abs(position))>200 m`、线速度 >5 m/s 或角速度 >10 rad/s；只丢弃异常样本，不改变底盘控制。
- 修正 LIO 后 30 秒 proxy：2998 样本全有限，最大绝对位置 0.0297 m、位置范数 P95 0.0314 m、最大线速度 0.0230 m/s；这不能替代对象点云和投影验收。

<a id="localization"></a>
## 3. 定位、重连与资源争用

- **外部所有权**：Livox/Point-LIO 属于机器人原工作流，D435i 驱动使用独立 host session；HuaweiVLN 隔离容器订阅输入。历史恢复/定点重启不授权改写外部工作空间或接管底盘。
- 8/12 实测 TF：`camera_init → aft_mapped` 为动态 odometry，`aft_mapped → base` 为静态边，平移 `[-0.2,0,0] m`、yaw `−1.5708 rad`。
- 未发现 `camera_init → default_cam` 或 `camera_init → veocc_d435i_color_optical_frame`；`camera_init` 是 LIO 定位 frame，不是相机光学 frame，不能假设 TF 已对齐。
- 8/13 原 helper 通过运行时开启 scan/body cloud 发布；未修改 `/home/orin26/code/point_lio_ws/install/point_lio/share/point_lio/config/mid360_orin.yaml`，也未修改 VEOcc 源项目。
- 实收 Livox LiDAR/IMU、registered/body cloud、odometry；frame 分别包括 `livox_frame`、`camera_init`、`body`、`camera_init → aft_mapped`。
- 只读诊断 `/home/orin26/HuaweiVLN/logs/diagnostics/lio_dds_current_20260813.md` 记录四个检查 topic 的 actual header PASS；发现 endpoint 本身不是数据到达证明。
- 8/13 LIO-only 短时曾达 world cloud 约 80 Hz、body cloud 约 9 Hz、处理 3.7–5.7 ms；加入 detector 后 world cloud 18–35 Hz 且下降、body 约 6 Hz、处理 90–165 ms，并反复丢弃旧帧。
- 后续出现处理 563–1259 ms 和 `[ros2run]: Segmentation fault`；停止 detector 后旧 LIO 仍可能积压，首次重启采样约 0.9–1.0 Hz，不能直接归因于 QoS 或“Orin 无法共存”。
- 起初 `ulimit -c=0`、core_pattern 为 apport pipe、无 coredumpctl，未得到可供 gdb 分析的 core；没有据此修改系统 sysctl 或安装软件。
- 8/14 干净单实例、CUDA detector `imgsz=320`、CPU 隔离与 mapping 节流后：35 秒 LIO-only body 9.47 Hz/odom 99.99 Hz，detector-only body 9.43 Hz，detector+mapping body 约 9.3 Hz/odom 63–68 Hz。
- 追加约 60 秒共存窗口 body cloud 8.4–8.6 Hz、最大间隔 0.354 秒，没有 1 Hz 退化或崩溃；这是短窗口保守基线，不是长期稳定性承诺。
- 历史资源策略包括 sensor-data BEST_EFFORT 浅队列、mapping executor 2 线程、detector 间隔逐步降到 2 秒、过期帧丢弃、点云转换间隔 1 秒/最多 10000 点、mapping timer 2 秒。
- 当时运行时 CPU 分配：Point-LIO 0、Livox 1–2、D435i 3–4、mapping 5、detector 6–7；未改外部项目配置。`lap` 固化于 `huawei-vln-realworld:orin-r36.5-qos-lap-20260814`，旧 tag 保留。
- **8/14 后续根因**：直接运行 Point-LIO 漏掉原 launch 的 `use_imu_as_input`、`prop_at_freq_of_imu`、`point_filter_num=6`、`space_down_sample=true`、surf/map filter=0.5、`ivox_nearby_type=6` 等，叠加未清干净实例导致积压和百万米位姿；不能把异常归咎于相机外参。
- `scripts/start_orin_lio_for_strive.sh` 恢复原 launch 契约，只覆盖输出开关；保留原 Livox 驱动和单一 LIO 实例，不修改机器人 YAML。

| 8/14 后续 60 秒窗口 | body cloud / odometry | CPU、内存与温度 |
| --- | --- | --- |
| LIO-only | 9.52 Hz / 100.0 Hz；位姿厘米级 | LIO 约 57% CPU、159 MB RSS；最高结温 60.4°C、GPU 56.6°C |
| detector 共存、mapping 关闭 | 9.43–9.50 Hz / 99.1–100.1 Hz | LIO 54.8–55.0% CPU、162.8 MB RSS；结温 60.8–61.2°C |

- 后一窗口无新增 >20 ms 高延迟；110 条既有 drop warning 均在干净启动/soak 前。证据：`/home/orin26/HuaweiVLN/logs/diagnostics/resources/20260814T091053Z/`。
- 崩溃捕获记录 `/home/orin26/HuaweiVLN/logs/diagnostics/lio-crash/configure-20260814T090856Z.txt`：只为指定 PID 调整 core limit 为 unlimited，记录环境并生成不自动执行的 GDB 命令；apport pipe 和无 coredumpctl 状态未变。全局 core 目录或解包 apport 需管理员另行批准。
- **8/21 并非此前成功的延续证明**：`logs/diagnostics/lio_dds_20260821T133422Z.md` 中 LiDAR、IMU、registered cloud、odom 均有 endpoint 但独立 header sample 超时；参数查询 scan/body 发布为 false、path 为 true。
- 同日 30 分钟资源窗口却测到 odom 约 100 Hz、body cloud 无实际样本；两个采样窗口/方法应分别记录，不抹平负结果。LIO 约 35% CPU，末段 60–66°C、内存约 5.5 GB。
- 8/21 资源证据 `logs/diagnostics/resources/20260821T133641Z/` 包含 tegrastats/process/cloud_hz/odom_hz 日志及 core_pattern、run.info；当轮没有重启或修改原 Livox/Point-LIO，也未开启 mapping。
- **9/23** 新镜像 host 网络、ROS_DOMAIN_ID=0、UDPv4 订阅 `/aft_mapped_to_init` 8 秒无消息，timeout=124；没有据此改名 frame 或假定 TF 对齐。

<a id="safety"></a>
## 4. Bridge-off、影子接口与控制安全

- 8 月上述数据测试未发布真实 `/waypoint`、`/way_point`、`/cmd_vel`，未接管底盘；真实 adapter 输出关闭。mapping smoke 的存在不改变控制门禁。
- 早期合同为 `approval_status=unapproved`、`owner_confirmation=false`、`allow_strive_waypoint_handoff=false`、`cmd_vel_direct_publish=false`、`emergency_stop_verified=false`。
- `/way_point`（PointStamped）→ `/waypoint`（Float32MultiArray）的格式/坐标 adapter 已实现，但默认不输出；实现状态不能代替 robot-specific contract 的书面批准。
- **8/19 影子闭环**仅使用 `/strive/test_way_point` → `/strive/test_waypoint_array`；真实 odometry、输入 frame `camera_init` 下有效输入转为 `[1.2046,1.8612]`，过期时间戳以 `dropping stale or pose-unavailable waypoint` 拒绝。
- 影子 topic 不是底盘接口；真实控制三 topic 与 `/topoplan/reached_goal` 均无发布者/未知。当时仍缺底盘所有权、坐标语义、到达/阻塞/超时反馈、限速、急停和人工接管确认。
- **9/23 软件控制权实现**用真实 `/mux_vel/selected` 反馈确认选源；服务响应仅说明调用结果。同连接已有同源反馈可用以适配事件发布语义，断连清除观测，每次授权切换清除旧速度缓存。
- 就绪门禁依赖 `/robot_status` 的 control_mode/base_state/fault_code 许可集合及本机单调时钟测量的遥测年龄。
- 合同 `controller_contract.agilex_rosbridge.hardware_state` 的 `allowed_control_modes`、`ready_base_states`、`ready_fault_codes` 必须是本车确认的非空枚举，`telemetry_timeout_s` 必须为正；**当时实车许可值尚未填入**。
- 硬件状态失效/过期即撤权、清除命令，并在连接可用时发送零速；恢复后必须显式重新授权，选源期间瞬时故障也会让本次自动授权失败。
- **桥接发零不能证明断网停车；底盘自身 watchdog 必须实测。** 输入超时、进程崩溃、断网、遥控器接管和急停后的停车时间不能由离线测试推断。
- 9/23 现场仅订阅遥测和查消息定义，未切 mux 或发底盘速度；8 秒收到 robot_status 398 条、ranger_status 397 条、selected 1 条。
- 末样本 `control_mode=3, base_state=2, fault_code=4`，线/角速度为零；故障原文“遥控器失联保护 / Remote control lost connection protection status”，selected=`/run_management/virtual_joy`，复查仍同故障。
- 类型分别为 `robot_status_core/RobotStatus`、`ranger_msgs/RangerStatus`、`std_msgs/String`；所查定义未给出控制模式枚举。单次样本不能解释全部模式/急停含义，尤其不能直接当作许可枚举。

<a id="validation"></a>
## 5. 无运动、隔离验证与未验收边界

- **8/21 基线** `00c376a75f05e29b8d8a0b36a97857d53f270163`：Generic RGB 与 D435i profile check 通过；mapping、真实 waypoint 和底盘控制全关闭，D435i 使用 body cloud。
- ROS workspace 7 包构建完成，镜像 `huawei-vln-realworld:orin-r36.5` ID `25048b859e0d`；CUDA 可用、YOLOE 初始化通过，USB 图像和 CameraInfo 实收。
- 禁网镜像 LVLM/schema/verifier 16 测试、离线验收 113 测试通过；未测真实远程服务延迟。两种 runtime-smoke 均 dry-run `WAIT`，缺 object_nodes、pose、image；15 秒 adapter `output_enabled=False`。
- **9/23 部署** `/home/orin26/HuaweiVLN_deployments/realworld_2c0b783`，镜像 `huawei-vln-realworld:orin-r36.5-agilex`，ID `sha256:b1e6b46af5187a8f35dbac337e1da3849eff07537b557be728fddacde65f7938`。
- 安装目录 `/workspace/STRIVE/real_robot/ros2_ws/install`；显式源码 overlay 构建 8 包，清除镜像继承旧 ROS 产物后构建约 4 分 10 秒。此前误指旧源码的“7 包快速构建”结果被排除。
- 原生 lower stack 经合同校验由聚合 launch 启动；外部控制器路径仍要求 `LOWER_CONTROLLER_CMD`。原生路径、外部命令缺失、合同失败三个分支检查通过，不能据此绕过合同。
- 原生 viewpoint 生产者复用 SysNav 栅格/点云/射线/代表点模型：覆盖变化或对象可见性触发候选，2 米去重，仅新增节点时发布；静止不持续产生 ID 是预期行为。
- 它只承担观测，未迁入原完整规划器运动输出或 x86 OR-Tools；要求点云/odom 同一 world_frame、默认时间差 ≤0.25 秒，相机高度历史默认 0.265 米仍需现场标定；有外部生产者时需避免重复启动。
- SysNav 原 HEAD `0fa15cc8bc18be7409272fb65fa514a9a5bca6b0`；原始文件哈希、版权及适配依据可从本节对应完整版与包内 SOURCE_SHA256SUMS、LICENSE、SOURCE_NOTICE.md 恢复。

| 9/23 验证 | 结果及适用范围 |
| --- | --- |
| AgileX 边界回归 / 离线验收 | 13 / 126 项通过，覆盖反馈顺序、失效、过期及恢复授权 |
| 原生规划 HIL | goal_reached、9 条真实规划路径，但 odometry 由测试模拟 |
| 原生安全仲裁 HIL | goal_reached、153 条路径、154 条最终命令（152 非零）；无实车运动 |
| 最终镜像隔离验证 | 禁网、无宿主源码挂载；安装路径、4 项规划资产、正式入口参数解析通过 |
| viewpoint→bridge | 合成输入生成 ID [0,1]，静止去重与对象 42 关联通过；不是真实传感器闭环 |

- 验证日志位于本地与 Orin 工作区的 `logs/current_source_20260923/`、`logs/image_validation_20260923/`；HIL 清理持有真实节点 PID，避免只结束 ros2 run 父进程而残留节点。
- **截至这些历史记录仍未完成**：本车状态枚举/急停依据、获批合同和限速/watchdog；真实传感器、TF、时间戳、相机参数和 viewpoint–对象关联；动态标定与像素投影验收。
- 仍需台架/低速验证选源、方向、人工接管、超时/进程故障/断网停车时间，完成真实 ObjectNav 目标任务，以及 detector、mapping、Point-LIO、viewpoint、导航运行时共同运行至少 30 分钟。
- 8/21 的 30 分钟无运动资源记录和 9/23 的隔离 HIL 均不满足上述全栈真实任务验收；保留镜像/已实现代码与继续限制真实运动并不矛盾。
