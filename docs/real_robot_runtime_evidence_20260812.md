# Orin-26 实物运行证据记录（2026-08-12）

本文把本次只读实物验证的关键输出集中记录，供部署验收和后续迁移复核使用。
记录不包含任何控制指令；没有发布 `/waypoint`、`/way_point` 或 `/cmd_vel`。

## 1. 1920×1080 MJPEG 相机 smoke

隔离容器使用宿主 Generic USB Camera 的稳定设备路径，参数为：

```text
device       /dev/video0 (host: /dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0)
width        1920
height       1080
V4L2 mode    MJPEG 30 fps
usb_cam      pixel_format=mjpeg2rgb
camera_info  camera_x001_intrinsics.yaml (read-only mount)
```

现场观察结果：

| 项目 | 结果 |
|---|---|
| image topic | `/camera/image` |
| image header frame | `default_cam` |
| image size | `1920×1080` |
| image encoding | `rgb8` |
| measured rate | 约 `24.9 Hz` |
| CameraInfo frame | `default_cam` |
| distortion model | `radial_3` |
| K | `[749.2058, 0, 1003.1, 0, 749.2058, 526.5258, 0, 0, 1]` |
| distortion | `[1.5847e-5, -1.3585e-7, 4.7386e-11]` |

`usb_cam` 的字面参数 `pixel_format=mjpeg` 不是合法 ROS encoding，会在打开设备后退出；
本 profile 固定使用 `mjpeg2rgb`，不要改回 `mjpeg`。

## 2. Point-LIO 与 TF 只读采样

当前外部 Point-LIO 以 `publish.scan_publish_en=true` 运行。实际采样到：

```text
/livox/lidar
/livox/imu
/cloud_registered
/aft_mapped_to_init
```

其中 `/aft_mapped_to_init` 的消息 frame 为 `camera_init`，child 为 `aft_mapped`。
实时 TF 关系为：

```text
camera_init --(/tf, dynamic)--> aft_mapped
aft_mapped  --(/tf_static, static)--> base
```

静态边的观测值为：

```text
translation = [-0.2, 0.0, 0.0] m
yaw         = -1.5708 rad (约 -90°)
```

只读查询没有发现以下实时边：

```text
camera_init -> default_cam
camera_init -> veocc_d435i_color_optical_frame
```

因此 `camera_init` 目前是 Point-LIO 的定位/里程计 frame，不应被误写成 USB 或 D435i 相机 frame；
部署不会自动广播未经确认的相机静态 TF。

## 3. D435i/MID-360 标定导入

来源文件：

```text
real_robot/calibration/orin26_d435i_mid360_targetless_v009_r009_extrinsics.json
```

约定为 `p_camera = R_lidar_to_camera @ p_lidar + t`。已导入的
`T_camera_from_lidar` 为：

```text
translation_m = [0.10970761, 0.67284314, 0.27480904]
rpy_zyx_rad   = [-1.935569001, 0.007663065, -3.131491315]
```

配置文件：

```text
real_robot/ros2_ws/src/semantic_mapping/config/projection_orin26_d435i_mid360.yaml
```

矩阵检查：`det(R)≈0.999999997`，正交误差约 `7.9e-9`，说明导入矩阵数值自洽。

## 4. 时间偏移和验收边界

提供的 D435i 外参 JSON 不含 RGB-LiDAR 时间偏移。两个 profile 均记录：

```yaml
rgb_minus_lidar_time_offset_s: 0.0
time_offset_status: assumed_zero_unvalidated
```

这表示可配置的初始假设，不是实测值；在取得 D435i 实时 `CameraInfo`、时间相关性测量和重投影验证前，
`calibration_status` 必须保持 `extrinsics_only`/`uncalibrated`，semantic mapping 不得开启。

Generic USB Camera 没有可复用的 D435i LiDAR 外参；D435i 的外参不得复制到 `default_cam`。

## 5. Controller contract

中文说明见 [`real_robot_controller_contract_zh.md`](real_robot_controller_contract_zh.md)。当前门禁为：

```yaml
approval_status: unapproved
owner_confirmation: false
allow_strive_waypoint_handoff: false
cmd_vel_direct_publish: false
emergency_stop_verified: false
```

已知接口格式为 STRIVE `/way_point` (`geometry_msgs/PointStamped`) 到外部
`/waypoint` (`std_msgs/Float32MultiArray`)；adapter 只做可配置格式/坐标转换，默认不输出。

