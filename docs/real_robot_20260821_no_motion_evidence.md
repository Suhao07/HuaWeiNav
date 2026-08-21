# Orin-26 无运动实物验收记录（2026-08-21）

本记录对应代码基线
`00c376a75f05e29b8d8a0b36a97857d53f270163`，目标机器为 Orin-26。
所有运行均保持 semantic mapping、真实 waypoint 和底盘控制关闭。

## 执行结果

| 项目 | 命令/证据 | 结果 |
|---|---|---|
| Generic RGB profile | `run_real_robot_profile.sh orin26_livox_mid360_generic_rgb check` | PASS；`START_SEMANTIC_MAPPING=false`、`dry_run=true`、lower controller=false |
| D435i profile | `run_real_robot_profile.sh orin26_livox_mid360_d435i check` | PASS；使用 `/cloud_registered_body`，mapping gate 仍关闭 |
| ROS workspace | 机器人端 `build_real_robot_ros_ws.sh` | 7 packages finished |
| Docker image | `huawei-vln-realworld:orin-r36.5` | final image ID `25048b859e0d` |
| GPU/ML | detector init smoke | `torch cuda=True`，YOLOE init PASS |
| USB camera | bounded camera smoke | `/camera/image` 与 `/camera_info` 收到；`1920x1080`、`rgb8`、`default_cam` |
| Runtime | generic 与 D435i `runtime-smoke` | dry-run `WAIT`，缺少 `object_nodes`、pose、image |
| Adapter | `run_real_robot_waypoint_adapter.sh`，15 s | `output_enabled=False`；没有控制 topic publisher |
| Offline acceptance | `check_real_robot_acceptance.sh` | `113 passed` |
| 30 min resource soak | `logs/diagnostics/resources/20260821T133641Z/` | `/aft_mapped_to_init` 约 100 Hz；`/cloud_registered_body` 无实际样本；Point-LIO 约 35% CPU；末段温度约 60--66°C、内存约 5.5 GB |

## LIO 诊断

报告：`logs/diagnostics/lio_dds_20260821T133422Z.md`

发现 publisher endpoint，但以下 topic 在独立 header sample 中均超时：

- `/livox/lidar`
- `/livox/imu`
- `/cloud_registered`
- `/aft_mapped_to_init`

只读参数查询结果：

```text
publish.scan_publish_en=False
publish.path_en=True
publish.scan_bodyframe_pub_en=False
```

因此这轮不启动 semantic mapping，也没有重启或修改机器人原有 Livox/Point-LIO。

## 资源监控

监控命令：

```bash
RESOURCE_MONITOR_DURATION_S=1800 \
  bash scripts/monitor_real_robot_resources.sh
```

输出目录：`logs/diagnostics/resources/20260821T133641Z/`

该目录包含 `tegrastats.log`、`process.log`、`cloud_hz.log`、`odom_hz.log`、
`core_pattern.txt` 和 `run.info`；监控进程已正常退出。

## 安全结论

本轮没有发布 `/way_point`、`/waypoint`、`/cmd_vel`，没有启动语义建图，没有接管
底盘，也没有改变机器人原有 LIO/相机项目的配置。
