# D435i–MID-360 标定验收记录（2026-08-13）

## 结论

当前不能批准 semantic mapping。相机角点拟合质量良好，但 RGB–LiDAR 时间偏移的动态扫描没有形成足够可靠的跨传感器对应，LiDAR–相机的像素级投影误差也尚未定义并验收。因此投影配置继续保持：

```yaml
calibration_status: extrinsics_only
rgb_minus_lidar_time_offset_s: 0.0
time_offset_status: assumed_zero_unvalidated
```

`0.0` 只是安全占位，不是测量值；不得把本记录中的 `+0.040 s` 候选值直接写入配置。

## 已获得的数据与指标

数据来自机器人上的既有 VEOcc-Rywang 记录，全部只读访问：

- 相机：D435i serial `233522079589`，历史标定分辨率 `640x480`，`fx=606.330017`、`fy=606.105346`、`cx=311.070922`、`cy=247.274445`，5 个 7×6 棋盘内角点观测。
- 外参：`T_camera_from_lidar`，calibration id `d435i-mid360-targetless-v009-r009`。
- 动态记录：`runs/debug-motion-02/raw`，约 29.5 s，`/base_odom` 最大位移约 2.507 m；包含 RGB、aligned depth、`/livox/lidar`、`/livox/imu`、`/base_odom` 和 `/aft_mapped_to_init`。

相机棋盘 PnP 重投影 RMSE（仅验证相机观测/内参）：

| 序列 | RMSE (px) |
|---|---:|
| solve-06 | 0.307 |
| solve-07 | 0.291 |
| solve-08 | 0.401 |
| solve-09 | 0.697 |
| solve-010 | 0.262 |

定义为：

\[
 e_{\rm pnp}=\sqrt{\frac{1}{N}\sum_i\left\|\pi(K, R X_i+t)-u_i\right\|_2^2}
\]

其中 \(X_i\) 是棋盘已知三维内角点，\(u_i\) 是图像检测角点。这个量不能代替 LiDAR–相机外参验收，因为没有 LiDAR 点与棋盘角点的一一对应关系。

时间偏移扫描定义：

\[
\Delta t=t_{RGB}-t_{LiDAR}
\]

对候选 \(\Delta t\)，取时间接近 \(t_{RGB}-\Delta t\) 的 LiDAR packet，用里程计将其变换到 RGB 时刻，再投影到 aligned depth；评分为有效对应点的鲁棒深度残差。`debug-motion-02` 扫描结果的最低候选约 `+0.040 s`，但只有 60 对帧/350 个有效对应，median absolute depth residual 约 `0.271 m`，90 分位约 `1.212 m`，故：

```text
time_offset_status = unidentifiable / not accepted
```

这不是 ROS bag 接收时间差；接收时间差只能说明记录链路延迟，不能识别相机曝光/驱动与 LiDAR packet 的物理时延。

## 验收脚本与日志

- `scripts/evaluate_d435i_lidar_calibration.py`：棋盘角点 PnP RMSE。
- `scripts/evaluate_rgb_lidar_time_offset.py`：动态 bag 的时间偏移扫描；只读，不创建 ROS publisher。
- 机器人日志：`/home/orin26/HuaweiVLN/logs/calibration_eval/`。

## 仍缺少的真正 LiDAR–相机投影误差

原 VEOcc 标定程序计算的是 LiDAR 平面 RMS（米）、法向角度残差（度）和投影 overlay，不计算像素级 RMSE。对于实物融合，建议同时报告：

1. LiDAR 平面几何残差：\(e_n=\angle(Rn_L,n_C)\)，\(e_d=n_C^T(Rc_L+t-c_C)\)。
2. LiDAR 点投影到 RGB 后，到棋盘四边形/边缘的像素距离 RMSE；应报告有效投影点数、median、p90 和 inlier ratio。
3. 若使用 RGB-D，则报告 LiDAR 投影深度与 aligned-depth 的 robust RMSE/median，并在独立 held-out 序列复核。

不能把“棋盘角点 PnP RMSE”冒充上述 LiDAR–相机像素误差。

## 建议的无运动补采方案

不需要机器人移动，也不需要发布 waypoint/cmd_vel。机器人底盘保持静止，由人工手持棋盘在相机和 LiDAR 共同视场内做左右、前后、远近、轻微旋转的连续运动，持续 30–60 s。只记录：

```text
/camera/d435i/color/image_raw
/camera/d435i/aligned_depth_to_color/image_raw
/camera/d435i/color/camera_info
/livox/lidar
/livox/imu
/aft_mapped_to_init
```

要求：棋盘至少覆盖 3 个距离、3 个方位和 2 个旋转角；避免运动模糊；记录 sensor header timestamp，不要只记录 wall-clock。补采后用一半序列估计 \(\Delta t\) 和外参，用另一半做 held-out 验收。建议初始扫描范围 `[-100, +100] ms`、步长 `2 ms`，最终在峰值附近步长 `0.5 ms`。

只有满足以下条件，才能把配置改为 `calibration_status: calibrated` 并考虑开启 semantic mapping（阈值需由项目负责人确认）：

- 时间偏移曲线有唯一清晰极小值，bootstrap/held-out 不跨越验收窗口；
- 至少 3 个独立动态片段、每段 ≥1000 个有效跨传感器对应点；
- LiDAR–图像投影 median/p90 像素误差和深度残差达到项目阈值；
- 记录 calibration id、分辨率、畸变模型、日期、样本数、误差和数据路径。

在这些条件满足前，继续保持 semantic mapping 关闭是正确的安全状态。

## 2026-08-14 重新评估（仍未批准）

已使用增强版只读脚本对既有 `debug-motion-02` bag 做交替样本 held-out 复核：

```text
scripts/evaluate_rgb_lidar_time_offset.py --held-out-split
```

结果（历史 640x480 内参，仅用于复核该 bag）：

| 指标 | train | held-out（train offset = -4 ms） |
| --- | ---: | ---: |
| offset | -4 ms | -4 ms |
| median depth residual | 0.230 m | 0.283 m |
| RMSE | 0.655 m | 0.631 m |
| P90 | 1.248 m | 1.095 m |
| projected / valid depth / inlier | 193 / 192 / 176 | 197 / 195 / 175 |
| depth inlier ratio | 0.917 | 0.897 |

held-out `pass=false`（median 超过当前 0.25 m 工作阈值），所以这个既有 bag
不能把 `calibration_status` 改成 `calibrated`。增强脚本现在会同时记录有效投影
点数和深度 inlier ratio；它仍不是棋盘四边形像素距离的一一对应误差，正式验收仍
需要新的 RGB-D + LiDAR 手持动态标定数据和真实 LiDAR–图像投影误差。

## 2026-08-14 对象坐标量级防护

此前出现的 `x/y/z≈10^5–10^6 m` 对象位置与同一时段 Point-LIO 位姿发散一致，
不是 D435i 外参本身的证据。项目 semantic-mapping 源码已增加可配置的 odometry
质量门：默认拒绝非有限位姿、`max(abs(position))>200 m`、线速度超过 5 m/s 或
角速度超过 10 rad/s 的样本；对应 D435i mapping YAML 已同步到机器人。该门只
丢弃异常传感器样本，不发布或改变任何底盘控制；semantic mapping 仍需在新镜像
构建后、标定通过后才可打开。

在 semantic mapping 关闭、detector 运行期间，`scripts/check_real_robot_pose_scale.py`
对 `/aft_mapped_to_init` 做了 30 s 只读检查：2998 个样本全部有限，
`max_abs_position=0.0297 m`、位置范数 P95 为 `0.0314 m`、最大线速度
`0.0230 m/s`，pose-scale proxy `pass=true`。这证明 Point-LIO 修正后输入位姿
已回到合理量级，但不等价于对象点云/投影误差验收；实际对象列表仍需在标定通过
后再启用 semantic mapping 验证。
