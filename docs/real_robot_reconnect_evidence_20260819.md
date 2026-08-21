# Orin-26 重建、Semantic Mapping 与影子 Adapter 验收（2026-08-19）

## 安全边界

本次未批准真实底盘控制。没有发布 `/waypoint`、`/way_point` 或 `/cmd_vel`，没有启动真实 waypoint adapter，也没有执行运动指令。外部 Livox、Point-LIO 和 D435i 项目只读取/重启其运行进程，未修改其配置文件。

## 重建与数据链

- 分支：`realworld`。
- 镜像：`huawei-vln-realworld:orin-r36.5`，重新构建成功，镜像内容约 1.66 GB。
- 容器：`huawei-vln-realworld-d435i`，semantic mapping 已启动。
- mapping 输入明确为 `/cloud_registered_body` 和 `/aft_mapped_to_init`。
- `/cloud_registered_body` 实测约 9.4–9.6 Hz；`/aft_mapped_to_init` 实测约 100 Hz。
- 日志持续出现 `Fusion precheck`、投影点计数和 `Fusion retained`，未出现邻近点云超时。

## 对象节点验收

使用容器内 `rclpy` 订阅 `/huawei_vln/d435i_object_nodes_list`，12 秒内收到 4 条真实消息，约 0.5 Hz。消息 frame 为 `map`，示例标签包括 `chair`、`desk`、`cabinet`；坐标量级约为米级，未出现异常数量级。

这证明当前配置的数据流和对象节点发布可运行；由于标定文件仍是 `calibration_status: extrinsics_only`，本记录不改变标定状态，也不等同于完整标定验收。

## OCC 标定文件复用结论

只读核对了：

```text
/home/orin26/VEOcc-Rywang/calibration/d435i-v009/initial-solve-v009-r004/calibration-result.json
/home/orin26/VEOcc-Rywang/runs/debug-motion-02/raw
```

可以复用 OCC 的 `T_camera_from_lidar`、相机内参、ROS bag 和 odometry 作为评估输入；已有时间偏移评估结果的候选最低点约 `+0.040 s`，但 `identifiable=false`，median depth residual 约 `0.271 m`，因此不能写入正式时间偏移或将状态改为 `calibrated`。

OCC 结果包含平面/法向残差和 overlay，但没有真正的 LiDAR 点到 RGB 棋盘边缘的像素 RMSE。该指标仍需新的定义和 held-out 数据验收，不能用相机 PnP RMSE 替代。

## Waypoint 影子闭环

在容器内启动独立影子 adapter：

```text
/strive/test_way_point (geometry_msgs/msg/PointStamped)
    -> strive_waypoint_adapter
/strive/test_waypoint_array (std_msgs/msg/Float32MultiArray)
```

使用真实 `/aft_mapped_to_init`，输入 frame 为 `camera_init`，验证结果：

- 有效输入转换为 `[1.2046, 1.8612]`，发布到影子 topic；
- 过期时间戳被丢弃并记录 `dropping stale or pose-unavailable waypoint`；
- `/waypoint`、`/way_point`、`/cmd_vel`、`/topoplan/reached_goal` 均无发布者/未知 topic；
- 影子输出不是外部底盘接口，不产生运动。

## 剩余门禁

1. RGB–LiDAR 时间偏移和真正 LiDAR→图像像素/深度重投影误差仍未通过验收。
2. `calibration_status` 保持 `extrinsics_only`。
3. controller contract 仍为 `unapproved`；底盘所有权、阻塞/超时反馈、限速、急停和人工接管未确认。
4. 当前 CPU 负载仍需长时间温度、掉帧和资源监控后再评估是否适合持续运行。
