# Clean robot workspace migration

更新日期：2026-09-08
共同基线：`realworld` / `b0e94a7`
新工作区：`/home/orin26/HuaweiVLN_deployments/realworld_2c0b783`

旧 `/home/orin26/HuaweiVLN` 未被覆盖。迁移前的 staged/unstaged patch、Git 状态和未跟踪
清单保存在 `/home/orin26/HuaweiVLN_backups/HuaweiVLN_pre_clean_20260907T151320Z/`。

## 迁移分类

| 类别 | 复用内容 | 处理边界 |
| --- | --- | --- |
| 传感器与 LIO 可观测性 | `orin26_livox_mid360_d435i.env`、`start_orin_lio_for_strive.sh`、LIO diagnostics | Livox/Point-LIO 仍由 `/home/orin26/code` 所有；只通过 ROS 参数覆盖启动，不改外部源码。 |
| 标定与投影 | OCC 初值上的 candidate JSON、D435i projection YAML、CameraInfo 读取配置 | candidate 已批准用于试运行；时间偏移 `unvalidated`，不据此宣称动态标定完成。 |
| semantic mapping 与融合 | RGB/Depth/CameraInfo、`/cloud_registered_body`、odom、投影融合、对象节点发布 | 输入固定为 `/cloud_registered_body`；修复对象图像保存目录；不回退到旧 runtime YAML。 |
| 控制边界与 shadow adapter | PointStamped→Float32MultiArray adapter、robot-specific YAML、控制门禁和 launch | `output_enabled=false`、`allow_real_motion=false`；contract 仍未批准，不接 mux、不发布 `/cmd_vel`。 |
| 部署、构建与证据 | Dockerfile/profile、ROS build、启动/检查脚本和部署 TODO | 新镜像、代码和 profile 由同一 Git 基线追踪；日志/outputs/build/install/cache 不从旧工作区整体复制。 |

## 明确未迁移

- 旧 workspace 的 dirty 状态没有直接覆盖新代码；根目录重复的历史配置/脚本只作为审计材料，
  不作为运行时输入。
- 旧 runtime projection YAML（`extrinsics_only`、旧 topic 和旧 OCC 数值）没有迁移。
- 旧 logs、outputs、colcon build/install 和缓存没有整体复制；模型资产仅复制当前 profile
  所需的五个文件。
- 外部底盘控制器、mux、急停和反馈所有权没有被假设或批准。

## 当前验证出口

- 新镜像 `huawei-vln-realworld:orin-r36.5` 已从该工作区构建，ROS overlay 7 个包成功。
- LIO-only 诊断四个 topic 均收到实际消息；独立 60 秒测量约为 cloud 9.43 Hz、odom 99.5 Hz。
- semantic mapping 已使用 candidate 启动，`/huawei_vln/d435i_object_nodes_list` 持续发布，样本
  frame 为 `map`，对象坐标为米级。
- shadow adapter 收到合成 PointStamped 并记录转换结果；`/waypoint`、`/cmd_vel` 均无 publisher。
