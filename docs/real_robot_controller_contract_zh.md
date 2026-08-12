# Orin-26 真机底盘控制契约（中文说明）

本文是 `real_robot/control/orin26_controller_contract.yaml` 的填写和验收说明。
YAML 的字段名保持英文，便于程序门禁读取；字段注释、测试步骤和审批记录使用中文。
当前契约必须保持 `approval_status: unapproved`，因为底盘所有权、阻塞/超时反馈和急停尚未由底盘所有者独立确认。

## 所有权边界

```text
STRIVE /way_point (geometry_msgs/PointStamped)
  -> waypoint adapter（只做格式/坐标/新鲜度校验）
  -> 外部 /waypoint (std_msgs/Float32MultiArray)
  -> 外部局部规划器/PD 控制器
  -> 外部底盘桥接与 mux
  -> /cmd_vel（STRIVE 禁止发布）
```

STRIVE 只拥有高层目标和 dry-run 逻辑；外部控制器所有者负责启动、监控、速度限制、
急停和人工接管。adapter 默认关闭，不能绕过局部规划器，也不能创建 `/cmd_vel` 发布者。

## 已观察但未批准的接口

| 项目 | 当前观察 | 验收状态 |
|---|---|---|
| STRIVE 输出 | `/way_point`，`geometry_msgs/msg/PointStamped` | 已知 |
| 外部输入 | `/waypoint`，`std_msgs/msg/Float32MultiArray` | 消息格式已知，坐标语义待书面确认 |
| 目标格式 | 历史代码按 ego-frame `[x, y]` 展平数组使用 | 待所有者确认 |
| 到达反馈 | `/topoplan/reached_goal`，`std_msgs/msg/Bool`，仅确认 `true=到达` | 阻塞/超时/取消未定义 |
| 速度上限 | 历史记录 `1.5 m/s`、`0.5 rad/s` | 不能作为首次运动限制 |
| 急停 | 话题/服务、断言值、复位流程均未核实 | 未通过 |
| 控制器所有者 | 未取得书面确认 | 未通过 |

## 分阶段验收（不改变真实运动门禁）

### 0. 静态与影子验证（无运动）

1. 记录控制器所有者、启动命令、命名空间和进程所有权。
2. 记录 waypoint 类型、坐标系/单位/轴向、到达/阻塞/超时/取消反馈、心跳和 watchdog。
3. 保持 `STRIVE_DRY_RUN=true`、`BLOCK_LOWER_CONTROLLER=1`、`WAYPOINT_ADAPTER_OUTPUT_ENABLED=false`。
4. 只向 `/strive/test_way_point` 或非订阅影子话题写入测试数据；确认 STRIVE 对 `/cmd_vel` 的发布者数量为 0。
5. 解决 `camera_init`、`base`、`map` 的坐标关系后，才允许做接口格式验证。

### 1. 台架验证（轮子离地或电机输出硬件禁用）

由底盘所有者启动控制器，确认刹车/电机禁用且无法产生车轮运动。验证正常目标、过期目标、取消/保持、
错误坐标系和错误数组长度，并记录每种反馈。此阶段也不能打开真实速度输出。

### 2. 有人持急停的低速验证

仅在清空区域、第二人持急停、机械限位/牵引保护到位后进行。初始建议限制：

| 限制 | 首次值 |
|---|---:|
| 最大线速度 | 0.15 m/s |
| 最大角速度 | 0.15 rad/s |
| 最大线加速度 | 0.10 m/s² |
| 最大角加速度 | 0.20 rad/s² |
| 单次目标距离 | ≤ 0.50 m |
| 目标超时 | 10 s |
| 目标过期 | 1 s |
| 心跳超时 | ≤ 0.5 s |

测试顺序：前进、停止/保持、后退、原地旋转、障碍阻塞、超时、急停断言和人工接管。
任何反馈缺失、位姿过期、心跳丢失、坐标不一致或急停不确定都立即失败并关闭交接。

### 3. 受监督部署

阶段 2 通过后，由所有者提出更高限制，并在契约中记录准确数值、日期、操作者、测试日志和回滚命令。
首次部署仍只允许 waypoint 交接，速度发布和 mux 所有权留在外部控制器。

## 批准条件

只有以下条件全部满足，才可将 `approval_status` 改为 `approved`，并由所有者签名/填写 `approval_reference`：

- 所有者和控制器所有权已确认；
- `/waypoint` 类型、坐标语义、单位和 adapter 转换已确认；
- 到达、阻塞、超时、抢占/取消和心跳值已实测；
- 速度、角速度、加速度和目标 watchdog 已确定；
- 急停接口、触发值、复位步骤和人工接管已独立测试；
- `allow_strive_waypoint_handoff: true`，且 `cmd_vel_direct_publish: false` 保持不变；
- 已完成台架与低速验收并保留日志。

## 回滚

由底盘所有者执行：触发已验证急停，关闭 waypoint 交接，停止本部署容器，确认外部底盘进入安全状态。
不得把 `git reset`、Docker 清理或批量 kill 当作运动安全措施；不得停止或修改外部 Livox/Point-LIO 项目。
