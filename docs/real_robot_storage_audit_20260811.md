# Orin-26 存储审计

> 审计日期：2026-08-11
> 范围：`/home/orin26`、Docker 镜像/容器；全部为只读检查。
> 本报告不授权删除、prune、移动或压缩任何项目数据。

## 总览

| 范围 | 容量 |
| --- | ---: |
| 根文件系统 | 233G 总量，206G 已用，15G 可用（94%） |
| `/home/orin26` | 168G |
| `/home/orin26/code` | 112G |

## 按项目统计

| 项目/目录 | 占用 | 主要内容 |
| --- | ---: | --- |
| `/home/orin26/code/Urban-Nav-SR` | 110G | 其中 `Policy_part` 约 109G |
| `Policy_part/bags` | 98G | `staging` 48G，`auto_import` 51G，ROS bag/MCAP |
| `Policy_part/onnx_ckpts` | 5.4G | NavFlow ONNX/TensorRT 相关模型 |
| `Policy_part/vis` | 3.7G | 可视化视频，`vis/combined` 3.4G |
| `Policy_part/test_runs` / `test_results` | 1.0G / 416M | 测试输入、推理结果 |
| `/home/orin26/VEOcc-Rywang` | 37G | EmbodiedOcc 代码、部署、标定、运行结果 |
| `VEOcc-Rywang/calibration/d435i-v009` | 15G | 多组 D435i 标定 ROS bags/raw 数据 |
| `VEOcc-Rywang/runs` | 15G | `postmount-long-01` 12G，两个 debug run 3.4G |
| `VEOcc-Rywang/runtime` | 4.6G | 其中 PyTorch ARM64 tar 约 4.8G |
| `VEOcc-Rywang/deployments` | 2.6G | 多个部署版本及 checkpoint/assets |
| `/home/orin26/HuaweiVLN` | 1.9G | 当前实物部署工作区，`assets` 约 1.8G |
| `/home/orin26/code/HuaWeiNav` | 1.8G | 原始代码工作区，含另一份 `real_robot` 资产副本 |

## 非项目占用

| 目录 | 占用 | 说明 |
| --- | ---: | --- |
| `/home/orin26/.cache` | 3.0G | 多份 NavFlow ONNX/TensorRT engine cache，单项约 78–682M |
| `/home/orin26/.ros` | 1.3G | 主要为 ROS 日志（约 1.3G） |
| `/home/orin26/miniforge3` | 3.7G | Conda/Mamba 环境 |
| `/home/orin26/.local` | 1.1G | 本地 Python/工具安装 |
| `/home/orin26/route_bag` | 1.5G | 独立路线 bag |

## 最大数据文件样例

只读扫描发现最大的文件主要是 ROS bag/MCAP，而不是当前 HuaweiVLN 源码：

```text
18.6G  Urban-Nav-SR/Policy_part/bags/auto_import/approved/nav_20260602_235343/*.mcap
11.2G  Urban-Nav-SR/Policy_part/bags/staging/nav_20260727_185823/*.mcap
9.8G   Urban-Nav-SR/Policy_part/bags/staging/nav_20260616_120018/*.mcap
9.1G   Urban-Nav-SR/Policy_part/bags/staging/nav_20260612_182610/*.mcap
9.0G   VEOcc-Rywang/runs/postmount-long-01/raw/raw_0.db3
4.8G   VEOcc-Rywang/runtime/pytorch-25.02-py3-igpu-arm64.tar
1.2G   VEOcc-Rywang/deployments/.../assets/checkpoints/finetune_scannet_depthanythingv2.pth
```

## Docker 占用

`docker system df -v` 显示 20 个可计费镜像，报告的共享后总占用约 25.36G；各镜像的
逻辑大小不能直接相加，因为大量基础层共享。当前有 **7 个带 VEOcc 标签的可直接
引用镜像**：`r005f`、`r005e`、`r005d`、`r005c`、`r005b`、`r005`、`r004`。
`docker system df -v` 另外列出 2 个无标签的 VEOcc 顶层结果；完整
`docker image ls -a` 则能看到 VEOcc 这一构建链共 38 条 16.4–16.9G 的镜像记录，
其中 31 条是无标签中间层，不能按 31 个独立运行镜像相加。

主要镜像包括：

- `huawei-vln-realworld:orin-r36.5`：7.18G，独有层约 1.68G。
- 多个 `veocc-v007-r005*`：每个逻辑大小约 16.9G，独有层约 5.1G。
- `nvcr.io/nvidia/pytorch:25.02-py3-igpu`：逻辑 16.4G，独有层约 4.84G。
- 多个无 tag 的历史构建镜像仍存在；当前 Docker build cache 为 0B。
- 已停止容器合计约 586M；当前没有 `huawei-vln-realworld` 运行容器。

## 历史记录关联

历史命令与目录对应关系如下：

- `Urban-Nav-SR/Policy_part`：USB 相机、NavFlow、VINS/RTK、AgileX bridge、PD
  controller、`ros2 bag record`，因此其 `bags` 是最大占用来源。
- `VEOcc-Rywang/deployments/.../source`：D435i 标定命令
  `capture_calibration_pose.py`，对应 15G calibration 数据。
- `start_livox_odom.sh`、`ros2 topic echo` 等命令只说明历史使用，不代表本次执行。
- 历史中的 `tmux kill-session`、`start_livox_odom.sh stop/restart`、`rm -r` 命令本次均未执行。

## 当前结论与安全建议

1. 最大空间来源是 Urban-Nav-SR 的 ROS bag（98G），其次是 VEOcc 标定/运行 raw
   数据（约 30G）。清理前必须由对应项目所有者确认保留集和备份策略。
2. VEOcc Docker 历史镜像和 HuaweiVLN/原始 HuaWeiNav 的模型副本都可能是有意隔离，
   不应直接执行 `docker prune` 或删除资产。
3. 当前只剩约 15G 可用空间，不建议立即重新构建 7G 级别镜像。
4. 可在得到确认后分别制定 bag 归档、旧 TensorRT cache、ROS log 和旧 Docker 镜像
   的回收计划；本次审计没有改变任何文件。
