# Orin-26 实物 LIO 资源诊断记录（2026-08-13）

## 安全边界

- 本次只读取传感器、定位输出、进程和日志。
- 未发布 `/waypoint`、`/way_point`、`/cmd_vel`，未启动 waypoint adapter、底盘控制器或 STRIVE runtime。
- Livox 和 Point-LIO 属于机器人原有工作流；项目容器只通过 ROS 订阅输入。

## 测试顺序与结果

1. 停止项目容器 `huawei-vln-realworld-d435i`，保留机器人原有 Livox driver。
2. 仅在 `livox_odom` tmux 的定位 pane 重启 Point-LIO，并通过运行时参数打开：
   - `publish.scan_publish_en=true`
   - `publish.scan_bodyframe_pub_en=true`
3. 仅 Point-LIO 的短时采样曾得到约：
   - `/cloud_registered`: 约 80 Hz
   - `/cloud_registered_body`: 约 9 Hz
   - LIO 处理耗时约 3.7–5.7 ms
4. 单独启动 detector（semantic mapping 仍关闭）后，重复采样观察到：
   - `/cloud_registered` 约 18–35 Hz，并持续下降
   - `/cloud_registered_body` 约 6 Hz，抖动明显
   - LIO 处理耗时约 90–165 ms
   - 日志反复出现 `Drop old lidar frames to reduce lag!`
5. detector 停止后，Point-LIO 日志仍有积压；随后独立重启 Point-LIO。重启后的首次采样仍约 0.9–1.0 Hz，尚未恢复到验收频率，故未启动 detector 或 semantic mapping。

## 崩溃证据

Point-LIO pane 中出现以下退出路径：

```text
Drop old lidar frames to reduce lag!
LIO mapping process time: 563–1259 ms
[ros2run]: Segmentation fault
```

机器人当前 core dump 条件：

```text
ulimit -c = 0
/proc/sys/kernel/core_pattern = |/usr/share/apport/apport ...
coredumpctl: command not found
```

因此本次没有可供 `gdb` 分析的 core 文件。未修改系统 sysctl、未安装软件包。

## 配置修正

- detector 默认推理间隔调整为 1.0 s，并丢弃超过 0.75 s 的图像帧。
- D435i semantic mapping 的点云参数统一为 `/cloud_registered_body`；LIO preflight 仍使用 `/cloud_registered`。
- `SYSNAV_MAPPING_EXECUTOR_THREADS=2`，为主机 Point-LIO 保留调度余量。
- `docker_en.sh` 和 framework 启动脚本均使用 `STRIVE_POINTCLOUD_TOPIC`，避免重复 launch 参数将 body cloud 覆盖回 `/cloud_registered`。

## 2026-08-14 复测与优化结果

本次先停止项目容器，定点清理没有响应 SIGINT 的旧 Point-LIO wrapper/进程，再在同一个 `livox_odom:0.1` pane 启动单一实例。重启命令仍只使用机器人原有安装和参数文件，并通过运行时参数打开 body cloud：

```text
publish.scan_publish_en=true
publish.scan_bodyframe_pub_en=true
```

干净实例的 35 秒窗口结果：

| 运行组合 | `/cloud_registered_body` | `/aft_mapped_to_init` | 备注 |
| --- | ---: | ---: | --- |
| LIO-only（重启后） | 9.47 Hz | 99.99 Hz | LIO mapping process 约 2–4 ms |
| detector-only，detector 绑定 CPU 6–7 | 9.43 Hz | — | CUDA detector、`imgsz=320`，无 mapping |
| detector + semantic mapping，detector 绑定 6–7、mapping 绑定 5 | 约 9.3 Hz | 约 63–68 Hz | 未出现 1 Hz 级退化或 Point-LIO 崩溃 |

同一共存配置追加约 60 秒 body-cloud soak 后，`/cloud_registered_body` 稳定在约
8.4–8.6 Hz（最大观测间隔约 0.354 s），没有回落到 1 Hz，也没有出现 Point-LIO
segmentation fault。长窗口略低于 35 秒窗口，作为 Orin-26 的保守运行基线。

此前出现的 0.7–1.0 Hz 并非新的 QoS 结论，而是 Point-LIO 内部已经积压的旧实例；日志中的 mapping process 曾达到 0.5–1.4 s，且重复 `Drop old lidar frames to reduce lag!`。定点终止并重启后恢复到上述频率。

## 资源优化变更

- detector 明确使用 `cuda:0`，显式将 `.pt/.pth` 模型迁移到 CUDA，默认输入尺寸为 640；Orin-26 profile 进一步使用 `imgsz=320`。
- detector 默认按 profile 降频（当前 `image_processing_interval=2.0 s`），丢弃过期图像。
- semantic mapping 输入固定为 `/cloud_registered_body`，点云转换节流 `1.0 s`、最多保留 10000 点，mapping timer 为 `2.0 s`。
- 点云、里程计、相机订阅使用 `qos_profile_sensor_data`（BEST_EFFORT、浅队列），避免高带宽传感器输入在语义节点中形成可靠传输回压。
- 运行时 CPU 亲和性：Point-LIO 可独占核心 0；Livox 使用核心 1–2；D435i 使用核心 3–4；detector 使用 6–7；semantic mapping 使用核心 5。亲和性是启动后的运行时设置，不修改机器人原有项目配置。
- Ultralytics BoT-SORT 所需 `lap` 已固化在本项目专用镜像 `huawei-vln-realworld:orin-r36.5-qos-lap-20260814`，旧镜像 tag 仍保留。

## 数据流与控制流验收状态

- semantic mapping 已能收到 `/cloud_registered_body` 和 `/aft_mapped_to_init`，日志出现 `Fusion precheck`、投影命中和 retained points。
- `/huawei_vln/d435i_object_nodes_list` 已有 1 个 publisher，容器内 `ros2 topic echo --once` 能收到 `tare_planner/msg/ObjectNodeList`。
- 对象几何数值仍不能作为标定验收结论；RGB–LiDAR 时间偏移、重投影误差、分辨率和标定 id 尚未完成正式报告，因此应继续标记为“数据链路可运行、几何未验收”。
- `/waypoint`、`/way_point`、`/cmd_vel` 均为 `Unknown topic`，当前没有控制发布者；`START_STRIVE_RUNTIME=0`、`START_WAYPOINT_ADAPTER=0`、`ALLOW_REAL_MOTION=false` 保持不变。

## 当前结论与后续

当前 Orin-26 在低频 detector + 节流 semantic mapping + CPU 隔离配置下可以与 Point-LIO 同时运行，数据链路和对象消息已打通；此前的 1 Hz 现象应按“Point-LIO 旧实例积压/未干净重启”处理。仍未完成的验收包括：

1. 用不少于 60 秒的长窗口记录稳定性（含 CPU/GPU、温度、掉帧和 Point-LIO warning 计数）。
2. 生成正式 RGB–LiDAR 时间偏移和 held-out 重投影误差报告，并将 calibration status 从 `extrinsics_only` 改为 `calibrated` 前再决定是否长期启用 semantic mapping。
3. 仅在填写并批准 robot-specific controller contract 后，做 waypoint adapter 的无运动格式验证；真实底盘控制仍保持关闭。
4. 独立配置 Point-LIO core dump/backtrace 采集，不修改机器人其他工作空间。

## 2026-08-14 重新连接后的根因定位与复测

### Point-LIO 启动参数修正

本次发现此前的直接 `ros2 run point_lio pointlio_mapping` 漏掉了机器人原有
`mapping_mid360_orin.launch.py` 的关键参数：`use_imu_as_input`、
`prop_at_freq_of_imu`、`point_filter_num=6`、`space_down_sample=true`、
`filter_size_surf/map=0.5`、`ivox_nearby_type=6` 等。漏参时 Point-LIO 日志反复
出现 `Drop old lidar frames to reduce lag!`，并把 `/aft_mapped_to_init` 推到
百万米量级；这不是标定外参导致的对象尺度问题。

现已恢复机器人原 launch 契约，只通过运行时参数打开：

```text
publish.scan_publish_en=true
publish.scan_bodyframe_pub_en=true
```

项目脚本 `scripts/start_orin_lio_for_strive.sh` 已显式携带上述原 launch 参数，
不修改机器人 Point-LIO 工作空间中的 YAML。复测时 Point-LIO 进程为单实例，
并保留原有 Livox 驱动。

### 60 秒传感器与资源测量

LIO-only 60 秒记录：

- `/cloud_registered_body`：约 9.52 Hz；
- `/aft_mapped_to_init`：约 100.0 Hz；
- Point-LIO 进程约 57% CPU、约 159 MB RSS；
- Jetson `tegrastats` 最高结温约 60.4 °C，GPU 约 56.6 °C；
- 运行过程中位姿保持厘米级，未再出现百万米坐标。

恢复 detector（semantic mapping 仍关闭）后再次做 60 秒共存 soak：

- `/cloud_registered_body`：约 9.43–9.50 Hz；
- `/aft_mapped_to_init`：约 99.1–100.1 Hz；
- Point-LIO CPU 约 54.8–55.0%、RSS 约 162.8 MB；
- 结温约 60.8–61.2 °C；
- 本 60 秒窗口无新的高延迟（>20 ms）记录；日志中较早的 110 条 drop warning 均发生在本次干净启动/soak 之前。

证据目录：

```text
/home/orin26/HuaweiVLN/logs/diagnostics/resources/20260814T091053Z/
/home/orin26/HuaweiVLN/logs/diagnostics/lio-crash/configure-20260814T090856Z.txt
```

### 崩溃捕获配置

已添加 `scripts/configure_real_robot_lio_crash_capture.sh`。它只对指定 PID 执行
`prlimit --core=unlimited`，记录 `/proc/<pid>/limits`、`core_pattern`、内核和
进程命令行，并生成不自动执行的 GDB post-mortem 命令。当前机器人状态仍是：

```text
core limit (Point-LIO PID): unlimited
core_pattern: apport pipe
coredumpctl: absent
gdb: /usr/bin/gdb
```

没有修改系统 sysctl、没有安装软件包；如要把 apport crash 文件解包或启用全局
core 目录，需要机器人管理员另行批准。

### 题点与下一步

- D435i 实际 topic 已修正为 `/camera/d435i/d435i_camera/...`，profile 与投影配置已同步；
- semantic mapping 仍保持关闭，直到标定验收通过；
- `scripts/monitor_real_robot_resources.sh` 提供可重复的只读长时监控；
- 旧的 1 Hz 结论应归因于漏参启动和未清理的 Point-LIO 实例，不再归因于“Orin 必然无法共存”。
