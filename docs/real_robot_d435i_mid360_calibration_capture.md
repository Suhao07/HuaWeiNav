# D435i--MID-360 标定数据采集

这套流程会启动或复用 Livox、Point-LIO 和 D435i 驱动，然后采集数据；不会启动 semantic mapping、waypoint adapter、局部规划器或底盘控制。启动使用机器人上已有的 `/home/orin26/code/start_livox_odom.sh` 和 RealSense launch 参数，不修改这些外部项目。

## 1. 采集前检查

在机器人工作区执行：

```bash
cd /home/orin26/HuaweiVLN
source /opt/ros/humble/setup.bash

ros2 topic list -t | grep -E \
  '/camera/d435i|/livox/(lidar|imu)|/cloud_registered_body|/aft_mapped_to_init'
```

必须有实际消息，而不是只有 publisher。脚本会对 RGB、Depth、CameraInfo、Livox、IMU、body cloud 和 odom 各取一个实际样本。

脚本优先使用以下 aligned-depth topic：

```text
/camera/d435i/d435i_camera/aligned_depth_to_color/image_raw
/camera/veocc_d435i/aligned_depth_to_color/image_raw
```

如果机器人只提供 `/camera/d435i/d435i_camera/depth/image_rect_raw`，必须先确认 RealSense 驱动已经启用 `align_depth`，然后显式使用 `--allow-nonaligned-depth`；脚本会在 manifest 中记录这是人工声明的对齐状态，不能把它当作自动验证。

## 2. 启动和录制

默认命令会自动执行以下动作：

1. 复用已有 `livox_odom` tmux session；不存在时调用 `/home/orin26/code/start_livox_odom.sh start`，启动 Livox MID-360 和 Point-LIO；
2. 复用已有 `d435i_camera` tmux session；不存在时启动 RealSense D435i，使用 `1280x720@30`、`align_depth.enable:=true`、`enable_sync:=true`；
3. 检查实际消息后开始录 bag；
4. 采集完成只停止本次脚本新建的 tmux session，复用的已有 session 不会停止。

直接录制：

```bash
bash scripts/capture_d435i_mid360_calibration.sh \
  --duration 144 \
  --phase-duration 8
```

如果只想复用已经启动的传感器、不由脚本启动：

```bash
bash scripts/capture_d435i_mid360_calibration.sh \
  --no-start-lio --no-start-d435i
```

如果机器人使用不同的 Point-LIO 启动脚本，可以显式覆盖默认命令：

```bash
START_LIO=1 \
LIO_START_CMD='<机器人现有 Point-LIO 启动命令>' \
bash scripts/capture_d435i_mid360_calibration.sh --start-lio
```

D435i 驱动的 launch 参数和命名空间属于机器人现有项目，脚本不会猜测。需要启动时由操作者提供已经验证过的命令：

```bash
START_D435I=1 \
D435I_START_CMD='<机器人现有 D435i launch 命令，需包含 align_depth.enable:=true>' \
bash scripts/capture_d435i_mid360_calibration.sh --start-d435i
```

如果更换机器人或 D435i 序列号，覆盖以下参数：

```bash
D435I_SERIAL_NO=_<serial> \
D435I_CAMERA_NAMESPACE=camera/d435i \
D435I_CAMERA_NAME=d435i_camera \
bash scripts/capture_d435i_mid360_calibration.sh
```

如当前深度 topic 名称不是 `aligned_depth_to_color`，且已由操作者确认驱动配置：

```bash
DEPTH_TOPIC=/camera/d435i/d435i_camera/depth/image_rect_raw \
bash scripts/capture_d435i_mid360_calibration.sh --allow-nonaligned-depth
```

录制期间脚本按 18 个阶段提示：3 个距离（near/middle/far）× 3 个目标朝向（front/left/right）× 2 个目标姿态（pose-A/pose-B）。每个阶段开始前将标定板/目标放到指定位置，按回车后保持约 8 秒。姿态 B 应改变标定板倾角或相机俯仰，不能只是重复姿态 A。脚本不控制机器人运动；若移动机器人，使用人工安全流程，不要通过本项目发布控制命令。

无终端交互时可以加 `--non-interactive`：

```bash
bash scripts/capture_d435i_mid360_calibration.sh --non-interactive
```

## 3. 输出文件

默认输出到：

```text
real_robot/calibration/raw_bags/d435i_mid360_<UTC>/
```

其中包括：

- `d435i_mid360_<UTC>/`：ROS 2 bag，包含 RGB、aligned depth、CameraInfo、`/livox/lidar`、`/livox/imu`、`/cloud_registered_body`、`/aft_mapped_to_init`、`/tf` 和 `/tf_static`；
- `capture_manifest.yaml`：topic、消息类型、对齐声明、时长和 bag 路径；
- `phase_log.csv`：18 个覆盖阶段的时间记录；
- `camera_info_sample.yaml`：录制前读取的 CameraInfo 样本；
- `bag_info.txt`、`ros2_bag_record.log` 和各 topic 的首条消息样本。

## 4. 后续评估入口

当前时间偏移评估器已支持实际 bag 的 topic 名称。取得 CameraInfo 中的 `fx/fy/cx/cy` 和 OCC 项目的 `T_camera_from_lidar`、`T_base_from_lidar`、`T_base_from_camera` 后运行：

```bash
python3 scripts/evaluate_rgb_lidar_time_offset.py \
  --bag real_robot/calibration/raw_bags/d435i_mid360_<UTC>/d435i_mid360_<UTC> \
  --extrinsics real_robot/calibration/orin26_d435i_mid360_targetless_v009_r009_extrinsics.json \
  --depth-topic /camera/d435i/d435i_camera/aligned_depth_to_color/image_raw \
  --lidar-topic /livox/lidar \
  --odom-topic /aft_mapped_to_init \
  --fx <fx> --fy <fy> --cx <cx> --cy <cy> \
  --held-out-split \
  --output real_robot/calibration/raw_bags/d435i_mid360_<UTC>/time_offset.json
```

这个结果是时间偏移和深度一致性评估，不自动批准标定。最终验收还要在独立片段上记录投影有效点数、median、P90、RMSE、inlier ratio 以及棋盘/标定板区域的真实 LiDAR-to-image 像素误差；阈值由项目负责人先写入验收记录。通过前，`projection_orin26_d435i_mid360.yaml` 继续保持 `calibration_status: extrinsics_only`，`require_calibration: false` 只允许用于数据链诊断。
