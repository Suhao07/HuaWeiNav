# Orin 实物部署实现与验证记录（2026-09-23）

软件实现已构建并完成隔离验证；实车状态合同、运动闭环和系统验收仍未完成。
本轮现场操作仅订阅遥测和查询消息定义，没有切换底盘 mux 或发布底盘速度。

## 状态与控制权实现

`agilex_platform.py` 使用实际 `/mux_vel/selected` 反馈确认选源。
服务响应只确认调用结果；同一连接中已有的同源反馈有效，适配 selected 的
事件发布语义。断连清除观测，每次授权切换清除旧速度缓存。

硬件就绪条件包含 `/robot_status` 的 control_mode、base_state、fault_code
许可集合，以及本机单调时钟测得的遥测接收年龄。物理控制合同需提供：

- `controller_contract.agilex_rosbridge.hardware_state.allowed_control_modes`
- `controller_contract.agilex_rosbridge.hardware_state.ready_base_states`
- `controller_contract.agilex_rosbridge.hardware_state.ready_fault_codes`
- `controller_contract.agilex_rosbridge.hardware_state.telemetry_timeout_s`

前三项为经本车确认的非空枚举列表，超时为正数。当前尚未填入实车许可值。
硬件状态失效或过期时撤销授权、清除命令并在连接可用时发送零速；恢复后需
显式重新授权。选源过程中的瞬时故障也会使本次自动授权失败。
桥接发零不能证明断网停车，底盘自身 watchdog 仍需实测。

## 原生 viewpoint 生产者

`strive_sysnav_viewpoints` 复用原 SysNav 的占据栅格、点云存储、射线检测和
代表点模型。观测覆盖变化或对象可见性触发候选点，原模型执行 2 米去重，
仅实际新增节点时发布 ViewpointRep。静止环境无需持续产生新 ID。

生产者只承担观测，不迁入原完整规划器的运动输出和 x86 OR-Tools 二进制。
输入点云和里程计必须处于配置的 world_frame，时间差默认最多 0.25 秒；
相机高度默认沿用原模型 0.265 米，需现场标定。聚合入口默认启用生产者，
已有外部生产者时设置 `start_viewpoint_provider:=false`。

原 SysNav HEAD：`0fa15cc8bc18be7409272fb65fa514a9a5bca6b0`。
原始工作树 sensor_coverage_planner_ground.cpp SHA256：
`42d3495631f7139d5cf0e17ed24c8a920caae3b3d62a2f4cbcdf99fdb4ccbb10`。
迁入模块的原始哈希、版权和适配说明见包内 SOURCE_SHA256SUMS、LICENSE、SOURCE_NOTICE.md。

## 部署与验证

Orin 工作空间：`/home/orin26/HuaweiVLN_deployments/realworld_2c0b783`。
最终镜像：`huawei-vln-realworld:orin-r36.5-agilex`。
镜像 ID：`sha256:b1e6b46af5187a8f35dbac337e1da3849eff07537b557be728fddacde65f7938`。
安装目录：`/workspace/STRIVE/real_robot/ros2_ws/install`。

Docker 构建复用 fix 基础镜像，清除镜像内继承的旧 ROS 产物后构建 8 个包，
耗时 4 分 10 秒。随后仅 COPY 已验证的启动脚本修复，复用全部编译产物。
仓库完整 Dockerfile 的源码也包含该修复，后续构建无需额外临时层。
`START_LOWER_STACK=1` 经合同校验后由聚合 launch 启动原生控制栈；独立外部
控制器路径仍要求 LOWER_CONTROLLER_CMD。原生路径、外部命令缺失、合同失败
三个启动分支检查通过。

| 验证 | 结果与范围 |
| --- | --- |
| AgileX 边界回归 | 13 项通过，涵盖真实选源反馈顺序、状态失效、遥测过期和恢复授权 |
| 离线验收 | 126 项通过 |
| 当前源码 overlay | 显式指定源码工作空间构建 8 包；导入路径和 CMake 源目录已核对 |
| 原生规划 HIL | goal_reached，9 条真实规划路径；里程计由测试模拟 |
| 原生安全仲裁 HIL | goal_reached，153 条路径、154 条最终命令，其中 152 条非零；无实车运动 |
| 最终镜像隔离验证 | 禁网、无宿主源码挂载；安装路径、4 项规划资产、正式入口参数解析通过 |
| 最终镜像 viewpoint→bridge | 合成输入生成 ID [0, 1]，静止去重、对象 42 关联通过 |

HIL 直接启动已安装的节点可执行文件，清理阶段持有真实节点 PID，解决
只结束 ros2 run 父进程后子节点残留的问题。源码已有的 ConstPtr 弃用和
setuptools tests_require 警告仍可见。

验证日志保存在本地及 Orin 的 `logs/current_source_20260923/` 和
`logs/image_validation_20260923/`，运行日志不纳入 Git。
早期曾因基础镜像 STRIVE_REAL_ROBOT_WS 指向旧源码而得到 7 包快速构建结果；
该结果已排除，以上证据来自显式源码目录和最终镜像的后续验证。

构建后根分区可用 64,044,326,912 字节（约 59.65 GiB），使用率 73%。
构建前只有 df -h 的 60G 粗粒度记录，无法精确归因整次磁盘增量。
本轮没有清理旧镜像、工作空间或数据。

## 现场证据与剩余工作

新镜像只读采集 8 秒收到 398 条 robot_status、397 条 ranger_status 和
1 条 selected。最后样本：control_mode=3、base_state=2、fault_code=4，
线速度及角速度为零，故障原文为“遥控器失联保护 / Remote control lost
connection protection status”，selected 为 `/run_management/virtual_joy`。
后续复查仍为同一故障。消息类型分别为 robot_status_core/RobotStatus、
ranger_msgs/RangerStatus、std_msgs/String；所查消息定义没有给出控制模式枚举。
单次采样无法解释全部控制模式或急停含义。

同一镜像在 host 网络、ROS_DOMAIN_ID=0、UDPv4 下订阅 /aft_mapped_to_init，
8 秒未取得消息，timeout 返回 124。没有据此改写坐标系名称或假设 TF 对齐。

尚需完成：

1. 现场遥控器、急停操作记录及实际底盘驱动枚举依据；补齐本车合同、限速和 watchdog。
2. 恢复真实传感器流，验证 TF、时间戳、相机参数及真实 viewpoint 与对象关联。
3. 台架/低速验证实际选源、方向、人工接管、输入超时、进程故障和断网后的停车时间。
4. 完成一次真实 ObjectNav 目标任务，以及 detector、mapping、Point-LIO、viewpoint、
   导航运行时共同运行至少 30 分钟的验收。

现场阻塞解除前，保留已验证镜像与代码；软件测试不替代上述实车验收。
