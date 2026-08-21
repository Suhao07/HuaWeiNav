# Orin-26 D435i 数据链路验收记录（2026-08-13）

本记录对应隔离容器 `huawei-vln-d435i-mapping-smoke` 的只读数据层测试。未启动
waypoint adapter、底盘控制器，也未发布 `/waypoint`、`/way_point` 或 `/cmd_vel`。

## 已验证链路

```text
/camera/d435i/color/image_raw
        + /camera/d435i/color/camera_info
        + /huawei_vln/d435i_detection_result
        + /cloud_registered_body
        + /aft_mapped_to_init
        -> semantic_mapping
        -> /huawei_vln/d435i_body_object_nodes_list
```

### D435i

- 实际硬件序列号：`233522079589`。
- RGB topic：`/camera/d435i/color/image_raw`。
- CameraInfo frame：`d435i_color_optical_frame`。
- 分辨率：`1280x720`，`plumb_bob`，畸变参数为全零（当前节点输出）。
- 运行内参：`fx=909.4949951`、`fy=909.1580200`、`cx=626.6063843`、`cy=370.9116516`。
- 短时观测图像约 `18–19 Hz`（驱动配置 30 Hz，受 Orin 负载影响）。

### Point-LIO 和检测

- Point-LIO 使用原项目 helper 启动，命令行覆盖 `publish.scan_publish_en=true` 和
  `publish.scan_bodyframe_pub_en=true`；没有修改原项目配置文件。
- `/cloud_registered_body` 实际收到 `body` frame 点云，短时约 `1.4–1.9 Hz`；
  `/aft_mapped_to_init` 实际收到 `camera_init -> aft_mapped` 位姿消息。
- `/huawei_vln/d435i_detection_result` 实际收到检测结果，短时约 `2.2 Hz`。

### 语义映射

映射测试使用 D435i 专用投影配置和 body-frame 输入，并将以下参数做成 profile 参数：

```yaml
cloud_input_frame: body
filter_depth_jumps: false
mask_erosion_iterations: 0
```

现场日志出现非零融合结果，例如：

```text
Fusion precheck: cloud_points=2223 projected_in_image=106 mask_hits=[8,13,8,1,11]
Fusion cloud counts: chair=8, cabinet=13, desk=8, curtain=1, chair=11
```

随后 `/huawei_vln/d435i_body_object_nodes_list` 成功输出 `cabinet`、`chair` 对象及
其 `PointCloud2` 点云（示例对象点数分别为 37、46）。这证明相机图像、检测结果、
Point-LIO 点云/位姿和语义映射已经完成数据闭环。

但本次采样的 `/aft_mapped_to_init` 位姿位置约为 `5e5–8e5` 数量级，输出对象的
`position`/`bbox3d` 也继承了这一数量级。消息格式和回调链路是通的，几何坐标合理性
却未通过验收；需要 Point-LIO 所有者检查初始化、frame/单位和运行稳定性后，才能把
对象位置用于导航或 waypoint 规划。

`mask_erosion_iterations` 的原因是：旧实现固定腐蚀 2 次，稀疏 Livox 点落在检测
mask 边界时会出现 `mask_hits>0` 但融合点数为 0；现在按机器人 profile 配置，默认
仍保留旧值 2，Orin-26 D435i 数据层 profile 使用 0。

## 尚未验收 / 不能宣称完成

- 外参文件仍是 `extrinsics_only`，RGB-LiDAR 时间偏移仍为未验证的零假设；不能将该
  数据层测试视为标定完成。
- 点云和对象输出频率受当前 Orin GPU/CPU 负载影响，需在正式部署时做性能优化和
  稳定性测试。
- Point-LIO 位姿/语义对象坐标出现异常大值，需完成坐标系、单位和初始化检查。
- 后续复测发现异常大值来自隔离建图容器与 Point-LIO 争用 CPU：停止旧 mapping
  容器并重启原有 LIO helper 后，位姿恢复到米级、LIO 处理约 1–6 ms。测试容器
  已设置 CPU 上限并固定到独立 CPU 集合；不得以无限制的 mapping smoke 结果验收。
- 检测器新增 `image_processing_interval=0.5` 和 `max_input_age_s=1.0` 参数，避免
  D435i 30 Hz 输入在 Orin 上积压成 5–10 秒旧检测。该修复已在本地提交，机器人
  网络恢复后需替换隔离容器检测进程并重新测量同步延迟。
- 控制流尚未接入：`/cmd_vel`、`/waypoint`、`/way_point` 当前均无 publisher。
  waypoint 格式转换 adapter 已实现，但真实 handoff 仍须底盘所有权、坐标语义、
  到达/阻塞/超时反馈、限速和急停流程书面批准后，才可进行无运动接口验证。
