# Orin-26 实物重新连接与数据层验收记录（2026-08-13）

## 安全边界

本次只恢复并读取传感器、定位和检测数据，没有发布 `/waypoint`、`/way_point`、`/cmd_vel`，没有启动底盘控制器，也没有执行机器人运动。Point-LIO 和 RealSense 驱动均通过独立的 host 进程/session 运行；HuaweiVLN 使用独立 Docker 容器。

## 已恢复的数据链路

### Livox / Point-LIO

- 使用项目已有 `/home/orin26/HuaweiVLN/scripts/start_orin_lio_for_strive.sh` 重启 `livox_odom` 会话。
- 仅通过运行时参数打开输出：

  ```text
  publish.scan_publish_en=true
  publish.scan_bodyframe_pub_en=true
  ```

- 未修改 `/home/orin26/code/point_lio_ws/install/point_lio/share/point_lio/config/mid360_orin.yaml`。
- 实际消息已收到：`/livox/lidar`、`/livox/imu`、`/cloud_registered`、`/cloud_registered_body`、`/aft_mapped_to_init`。
- 样本 frame：Livox 为 `livox_frame`，注册点云为 `camera_init`，机体点云为 `body`，里程计为 `camera_init -> aft_mapped`。
- 只读诊断报告：`/home/orin26/HuaweiVLN/logs/diagnostics/lio_dds_current_20260813.md`，四个 topic 均为 `RESULT: PASS (actual header received)`。

### Intel RealSense D435i

- USB 设备：Intel RealSense D435i，序列号 `233522079589`。
- 驱动 session：`huawei_vln_d435i`，使用 ROS Humble `realsense2_camera`，未修改 VEOcc-Rywang 源项目。
- 配置：RGB-D 同步、depth 对齐到 color、`1280x720@30Hz`；D435i HID IMU 保持禁用，Point-LIO 使用 Livox IMU。
- 已收到：
  - `/camera/d435i/color/image_raw`
  - `/camera/d435i/color/camera_info`
  - `/camera/d435i/aligned_depth_to_color/image_raw`
  - `/camera/d435i/aligned_depth_to_color/camera_info`
- 实时 `CameraInfo`：`fx=909.494995`、`fy=909.158020`、`cx=626.606384`、`cy=370.911652`，`plumb_bob`，当前畸变项为零。

### HuaweiVLN Docker

- 镜像：`huawei-vln-realworld:orin-r36.5`。
- 当前专用容器 profile：`orin26_livox_mid360_d435i`，容器名 `huawei-vln-realworld-d435i`。
- 本次运行的 detector-only 容器使用 `/camera/d435i/color/image_raw`，检测节点已订阅该 topic 并输出 `/huawei_vln/d435i_detection_result`。
- 容器内已收到自定义检测消息，样本包含 `chair`、`cabinet`、`desk`、`tv_monitor` 等目标。
- `/cloud_registered` 和 `/cloud_registered_body` 当前由 host Point-LIO 发布；在 detector-only 模式下 semantic mapping 不订阅点云，这是预期行为。
- 为兼容当前 ARM64 镜像的 launch 参数，D435i profile 暂时给未启用的 USB-camera 参数提供只读 `camera_x001_intrinsics.yaml` 占位；D435i 实际内参仍只取 `/camera/d435i/color/camera_info`。

## 控制安全验收

当前 ROS graph 中：

```text
/waypoint  : Unknown topic
/way_point : Unknown topic
/cmd_vel   : Unknown topic
```

因此本次恢复没有控制发布者。waypoint adapter 仍为 `START_WAYPOINT_ADAPTER=false`、`output_enabled=false`；controller contract 仍未批准。

## 后续门禁

1. D435i–MID-360 外参资产仍为 `calibration_status: extrinsics_only`；需要测量 RGB-LiDAR 时间偏移，并补充重投影误差、日期和样本数。
2. 完成标定验收后，才能将 D435i projection profile 标记为 `calibrated` 并启用 semantic mapping。
3. waypoint adapter 只能先做隔离 dry-run 格式验证；真实输出需要批准 robot-specific controller contract。
4. 真实运动仍需要确认底盘所有权、反馈/阻塞/超时、限速和急停流程。
