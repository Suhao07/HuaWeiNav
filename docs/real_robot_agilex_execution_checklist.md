# HuaweiNav 复用 FWTW 松灵接口执行清单

核对日期：2026-09-21。范围：本地代码与已有部署记录；本次没有连接机器人或发送控制。

目标：在已有 SysNav 下层和安全仲裁之上，复用 FWTW 的 rosbridge、遥测和 mux
切换，实现可观测、可接管、可验证的实物闭环。本文是待办清单，不是实物验收结论。

文件/函数级实施见 [代码实现计划](real_robot_agilex_implementation_plan.md)。
该计划将类型适配收敛到 bridge 回调，沿用已有 action 接管终态，并限定必要测试范围。

## 1. 当前实现与证据边界

| 部分 | 当前事实 | 还缺什么 |
|---|---|---|
| D435i、MID-360、Point-LIO、检测与语义建图 | 已有真机数据链记录；低速融合已接受 | 修复后完整组合的连续长测 |
| SysNav localPlanner、pathFollower、motion server | HuaweiNav 已有代码、launch 和离线测试 | 本车输入坐标、运行配置与实际运动验收 |
| SafetyVelocityMux | 已有自动/人工仲裁、软件急停、命令和传感器 watchdog | 松灵模式/故障联动，以及实际急停和断链验证 |
| 松灵 rosbridge 发布、遥测、选源 | FWTW 正式仓库已有实现 | 尚未成为 HuaweiNav 已验收的底盘后端 |
| 速度消息 | HuaweiNav 最终输出 TwistStamped；FWTW bridge 输入 Twist | 安全仲裁后的最小类型适配 |
| controller contract | Orin 合同仍是旧外部 waypoint 控制器的未批准观察记录 | 新控制链的完整字段、真实边界和审批证据 |
| SysNav viewpoint | 已有 bridge；部署记录中真实 provider 尚未验收 | 实际 viewpoint 输入和任务级闭环 |

代码状态：第一批通信接入、第二批软件模式联动、lower planner 独立点云配置与 terrain frame 参数化、launch/profile
接线已经实现。ROS overlay、目标设备运行、底盘模式字段和实物安全试验仍未完成。

已落地的代码边界：

- `agilex_platform` 订阅 `/cmd_vel`（`geometry_msgs/msg/TwistStamped`），按分量限幅后通过 rosbridge 发布 `/navflow_cmd_vel`（`geometry_msgs/Twist`）。`remote_control_enabled=false` 时只做连接和遥测观察，不执行 advertise、mux 服务或速度发布。
- 同一节点订阅 `/mux_vel/selected`、`/ranger_status`、`/robot_status`、`/navflow_cmd_vel`、`/base_cmd_vel`，输出 `/platform/chassis/mux_selected`、`/platform/chassis/ranger_status`、`/platform/chassis/robot_status`、`/platform/chassis/navflow_cmd_vel_received`、`/platform/chassis/base_cmd_vel_received` 和 `/platform/chassis_selected`。
- `/platform/set_autonomy` 是唯一模式入口：`true` 先完成 `/mux_vel/add`、再 `/mux_vel/select`，成功后才发布自动授权；`false` 发送零速度并选择 `/run_management/virtual_joy`。physical 节点入口复用 `controller_contract.py` 校验批准合同。
- rosbridge 每次重连先选择人工源；请求超时、正常退出和断链收尾都会发送零速度并尝试恢复人工源。`/platform/chassis_ready` 当前只表示“连接存在且软件 selected 为自动”，不代表已确认电机或急停状态。
- 遥测 JSON 保留原始 rosbridge message、消息体和本机 wall/monotonic 接收时间；physical 合同同时约束 bridge 的限速与命令 watchdog。
- physical bridge 配置中的限速和 command timeout 必须为正值；shadow 可以保留调试配置，但不执行控制发布。
- `SafetyVelocityMux` 在 `require_chassis_ready=true` 时把 readiness 作为自动输出门；`lower_cloud_topic` 将下层公共坐标点云与感知 profile 的点云配置分开，`terrain_frame` 统一声明 terrain 输出 frame。
- `AGILEX_COMMAND_TIMEOUT_S` 从 profile 经 Docker、SysNav 脚本和 aggregate launch 传入 bridge，作为同一份部署 watchdog 配置。
- physical 聚合入口同时沿用 `ALLOW_REAL_MOTION`、`ENABLE_LOWER_CONTROLLER` 和 `BLOCK_LOWER_CONTROLLER`；新 bridge 门禁不会替代已有实车运动门禁。
- `docker_en.sh` 与专用 Docker wrapper 传递同一组 AgileX、点云、readiness 和运动门禁参数，profile 入口不会产生第二套默认值。

本地已完成 fake rosbridge 边界测试（subscribe、advertise、add、select、零速和 WebSocket JSON 帧）。仍需在目标 ROS 环境执行 overlay 构建、实际消息类型/QoS 核对和连接/断链观测。物理急停、底盘 watchdog、硬件控制模式和首次低速运动不由本地检查代替。

当前软件入口：只读 bridge 使用 `start_agilex_platform:=true`、`agilex_remote_control_enabled:=false`；需要连同
SysNav lower stack 做无运动影子联调时，保持 `STRIVE_DRY_RUN=true`、
`STRIVE_LOWER_CONTROLLER_ENABLED=false`，脚本会将 `require_controller_contract` 关闭但不会打开远端控制。
physical 还需显式设置 `agilex_remote_control_enabled:=true`。启动 aggregate launch 后调用
`ros2 service call /platform/set_autonomy std_srvs/srv/SetBool "{data: true}"`
请求自动，`data: false` 请求人工。默认 profile 保持该节点关闭；物理启用还需要批准后的合同。

源码定位：

- HuaweiNav：[安全 mux](../real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/safety_velocity_mux.py)、[下层 launch](../real_robot/ros2_ws/src/strive_sysnav_motion/launch/sysnav_lower_stack.launch.py)、[合同校验](../real_robot/control/controller_contract.py)、[现有 Orin 合同](../real_robot/control/orin26_controller_contract.yaml)。
- FWTW 参考副本：`/Users/ricksu/Documents/ChatGPT/FWTW ICRA2027/work/control_update/tools/agilex_api/`；复用其中的 `ros2_cmd_vel_to_agilex_ws.py` 和 `keyboard_mux_switch.py` 协议行为。该路径不属于部署依赖。
- FWTW 有未提交的配置/启动器修改，复用时不能只记录 HEAD；必须记录实际文件哈希。`Documents/ChatGPT/FWTW ICRA2027` 是历史副本，不作为发布来源。
- HuaweiNav 现有进度见 [部署清单](real_robot_deployment_todo.md)。旧清单中的外部 `/waypoint` 路线不能与本文 SysNav 原生下层路线同时启用。

## 2. 本清单采用的实施路线

采用已有 SysNav 原生下层作为本轮接入路线；这是实施选择，尚不是本车后端验收结论。

```text
导航任务 -> ExecuteWaypoint / motion server -> /way_point
  -> localPlanner -> pathFollower -> /cmd_vel/autonomy (TwistStamped)
  -> SafetyVelocityMux -> /cmd_vel (TwistStamped)
  -> 最小类型适配 -> 独立底盘发送话题 (Twist)
  -> rosbridge -> 松灵 /navflow_cmd_vel -> mux_vel -> /base_cmd_vel

人工源 /run_management/virtual_joy --------------------> mux_vel
松灵 selected/status -> 遥测与状态适配 -> HuaweiNav 接管/故障状态
```

独立底盘发送话题名称在 A1 中固定；不把 FWTW 的候选速度绕过 HuaweiNav mux 直接接车。
本轮以松灵原有人工源完成驾驶接管，不再增加一套 HuaweiNav 手柄驾驶链。

## 3. 按依赖执行

### A1 — 固定接口、复用版本与唯一输出归属

- [ ] 记录 HuaweiNav/FWTW commit、实际复用文件 SHA256 和本地未提交差异。
- [ ] 固定 ROS2 仲裁输出、Twist 适配输出、松灵自动源、人工源、状态话题及消息类型。
- [ ] 确认底盘地址/网卡；当前 FWTW 配置是 `ws://192.168.1.102:9090`、`wlP1p1s0`，不能据此认定现场一定相同。
- [ ] 固定 SysNav 原生下层为本轮唯一运动后端；启动时禁止旧外部 PD/waypoint 控制器并行输出。
- [ ] 固定 `/navflow_cmd_vel` 发布归属，禁止 HuaweiNav 和 FWTW 两套自动桥接同时向该话题发送。
- [ ] 只读核对底盘状态字段、消息类型和实际样本；明确硬件模式、故障码、急停、底盘通信 watchdog。未知枚举不得猜测。

产物：一个接口表、版本/哈希清单、现场待确认字段表。
验收：每条控制边都有明确生产者/消费者、类型、单位和坐标；未知项显式保留。

### A2 — 先接入只读遥测

依赖：A1 的网络与话题定义；不依赖运动合同批准。

- [ ] 复用 FWTW relay，订阅 `/ranger_status`、`/robot_status`、`/mux_vel/selected`、`/navflow_cmd_vel`、`/base_cmd_vel`。
- [ ] 保留原始 JSON、原始 Header（若有）、Orin 接收时间及单调时间；在 HuaweiNav 中固定 relay 输出命名。
- [ ] 为状态适配提供新鲜度；断连不复用旧状态，不伪造零反馈。
- [ ] 区分候选命令、发送命令、mux 后命令和反馈速度；将它们加入同一次录制。
- [ ] 验证迟加入订阅者能得到 mux 当前状态，重连后的旧状态不会被误认为新鲜。

产物：可单独启动的遥测节点与录制配置。
验收：可以只读看到底盘状态和软件选源；空闲时命令话题可为空；relay 不执行 advertise 速度、选源或运动操作。

### A3 — 最小速度类型适配和传输接入

依赖：A1；先在 fake rosbridge 或隔离网络验证。

- [ ] 在 SafetyVelocityMux 之后实现 `TwistStamped.twist -> Twist`，或给现有 bridge 增加一个明确的输入类型选项，二选一，不同时新增两套路径。
- [ ] 明确上游 frame 与底盘轴向；类型转换不承担隐式旋转、速度缩放或参考点变换。
- [ ] 沿用 FWTW 已核对的 rosbridge JSON 语义（advertise/publish、mux add、每连接心跳和重连）。HuaweiNav 将最小客户端随节点发布，避免把历史 FWTW 工作树路径变成部署依赖；不再启动第二个 bridge。
- [ ] 独立启动为 shadow：即使 enable file 存在，也不得连接控制发布链。
- [ ] 统一控制器、SafetyVelocityMux、bridge 的限速和超时；桥接上限不高于本车合同。FWTW 的 1.5 m/s、0.8 rad/s、1 s 不直接成为 HuaweiNav 首次运动配置。
- [ ] 验证停止上游、杀死适配器、杀死 bridge、网络中断的区别。软件停止进程不等于底盘已经停车。

产物：一个类型适配点、复用 bridge、集中配置。
验收：隔离环境中六个 Twist 分量转发正确；只有仲裁后的速度可到 bridge；上游超时归零；shadow 无控制网络副作用。

### A4 — 联通人工接管、自动恢复和故障状态

依赖：A2、A3；只实现本车已确认的状态字段。

- [ ] 将松灵人工选源映射到 HuaweiNav 人工接管，暂停/终止活动导航目标并确认 action/runtime 状态传播。
- [ ] 人工接管时停止 bridge 自动发布并调用松灵选源；避免只修改 `/platform/manual_takeover` 而底盘 mux 仍留在自动源。
- [ ] 自动恢复时重新确认状态新鲜、底盘允许上位机控制、无急停/故障、传感器与候选有效，然后选自动源并开放发布。
- [ ] 明确恢复策略：建议显式恢复自动，不因遥测重连或 `manual_takeover=false` 自动续跑旧目标；清理接管前缓存和待执行目标。
- [ ] 将底盘故障、遥测断流、控制连接异常映射到阻止自动的状态，记录可诊断原因；不把未知状态按正常处理。
- [ ] 固定状态写入者，避免遥测适配器、键盘程序和启动器相互覆盖 Bool 状态或形成反馈循环。
- [ ] 修正复用键盘工具的启动语义：FWTW 当前启动直接 AUTO，HuaweiNav 的启动策略必须明确写入代码、帮助和测试，不能保留“默认人工”的失实注释。
- [ ] 明确 `/platform/safe_hold` 只关闭自动，不能替代压过人工流的急停；急停复位不能自动恢复旧命令。

产物：模式转换表与一个本车状态适配实现。
验收：AUTO→MANUAL→HOLD→显式恢复全过程，上层任务状态、软件仲裁、松灵选源一致；异常/重连不意外恢复运动。

### A5 — 核对 SysNav 输入坐标、QoS 与执行反馈

依赖：A1；可与 A2–A4 的离线工作独立推进。

- [ ] 核对实际 odom、点云、waypoint、路径和底盘 frame；验证 `map`、`camera_init`、`body`、`base`、`base_link` 的变换。
- [ ] 专门核对 `cloud_registered_body` 能否供 terrain/local planner 使用；当前感知 profile 使用机体系点云，不代表下层允许直接用该点云替代世界系点云。不得仅靠 topic remap 改变坐标语义。
- [ ] 验证各传感器订阅 QoS 与驱动匹配，尤其 SafetyVelocityMux 的 odom/点云新鲜度输入。
- [ ] 验证 motion server、local planner、path follower 的 reached、blocked、timeout、cancel/preempt、manual takeover 和 fault 反馈。
- [ ] 用实际反馈确认速度单位、前进/转向符号、底盘可执行运动学及停止判据。

产物：本车 lower-stack 配置、frame/QoS 表、反馈映射。
验收：确定性回放可追踪目标→路径→候选→安全输出→终态；坐标不靠改 frame 字符串“修复”。

### A6 — 完成本车合同和统一启动/停止

依赖：A1–A5 已有明确实现与接口；现场验收结果按 A8 补录。

- [ ] 从新版模板补齐唯一 owner、自动/人工/最终话题、传感器 watchdog、waypoint/action/frame、容差、超时、权威反馈、限速/加速度、急停和接管流程。
- [ ] 为 rosbridge 地址、底盘选源、状态新鲜度与断链行为建立同一份可审阅配置，避免脚本常量与合同各自漂移。
- [ ] 新合同草案保持 `unapproved`；按阶段记录实际允许的试验范围，生产批准不能用静态填字段替代。
- [ ] 明确验收与批准的顺序：电机禁用的台架试验先形成证据；首次地面低速试验使用负责人已确认的受限合同与范围；验收后再批准相应生产范围，不关闭合同校验绕过门禁。
- [ ] 统一 launch/profile：只读、shadow、受控 physical；默认无真实运动，现有安全门禁继续生效。physical 聚合入口还需 `ALLOW_REAL_MOTION=true`、`ENABLE_LOWER_CONTROLLER=true`、`BLOCK_LOWER_CONTROLLER=false`。
- [ ] 启动连接先选择人工源，重启不得继承旧自动授权；停止时发零、关闭自动发布、请求恢复人工并确认结果，失败明确报错。

产物：本车合同、部署 profile、统一启动/停止入口。
验收：合同缺失/不匹配时 physical 被拒绝；shadow 正常可用；控制停止不误停外部 LIO/相机。

### A7 — 离线与无运动集成验收

依赖：A2–A6。

- [ ] 运行现有 motion safety、controller contract、SysNav migration、motion action/runtime 相关回归与项目 acceptance。
- [ ] 补充真实新增边界测试：Twist 类型适配、shadow 零副作用、选源失败不使能、人工接管中断任务、重连不恢复旧命令、遥测过期禁止自动。
- [ ] fake rosbridge 覆盖 advertise/add/select/subscribe/心跳/断连及服务失败；验证服务响应和实际 selected，不只看发送日志。
- [ ] 无运动集成验证传感器→规划→仲裁→模拟底盘反馈，并检查唯一发布者和 mode 状态一致性。

产物：带版本、命令和结果的回归报告。
验收：全部目标用例通过；明确区分离线通过、真实订阅通过和物理控制通过。

### A8 — 台架与低速实车验收

依赖：A7、对应阶段的现场条件和合同确认。此项必须现场执行，不能由本地代码审计勾选。

- [ ] 电机禁用或轮子离地：逐级确认安全输出、松灵自动输入、mux 后输出、底盘状态；确认软硬模式和遥控接管。
- [ ] 先写明停止响应时间、停止距离、目标误差和状态延迟的通过阈值，再执行试验，禁止事后按结果调整阈值。
- [ ] 按已确认的受限配置做短距离前进、停止、转向；后退仅在本车及规划器明确支持时测试。
- [ ] 测试人工接管、恢复自动、取消、目标过期、堵转/阻塞、里程计/点云断流、遥测断流、桥接退出、断网和硬件急停。
- [ ] 验证断网时底盘自身 watchdog；记录实际停车，而非把“发送零速度”日志当作通过。
- [ ] 验证停止/回滚后没有旧命令恢复，外部传感器服务保持正常。

产物：同一时间线的 ROS bag/遥测、选源记录、实测停止与模式切换结果、负责人确认。
验收：每个测试都有明确通过/失败和证据；失败保持受限状态，不扩大运行范围。

### A9 — 完整 ObjectNav 任务闭环

依赖：A8；viewpoint provider 接入与感知长测可提前在无运动模式完成。

- [ ] 启动并验收真实 SysNav viewpoint provider，解除缺少 `/viewpoint_rep_header` 导致的 WAIT；不以对象中心替代可执行 viewpoint。
- [ ] 完成 detector + semantic mapping + Point-LIO 修复后整栈连续至少 30 分钟测试，记录 TF、丢帧、资源和温度。
- [ ] 执行自然语言目标→语义地图→viewpoint→waypoint/action→底盘→到达反馈→新鲜视觉证据验证。
- [ ] 人工介入、目标失败和重新规划正确结束/恢复任务；一次运行完整保存配置、代码哈希和结果。

验收：至少一个真实任务完整闭环，并保留失败与接管案例；底盘可控与完整 ObjectNav 成功分别记录。

## 4. 执行顺序与完成定义

```text
A1 接口与来源
  ├─ A2 遥测 ─ A3 类型适配/桥接 ─ A4 模式联动 ─┐
  └─ A5 坐标/QoS/反馈 ──────────────────────┤
                                      A6 合同/启动
                                          ↓
                                      A7 离线/无运动
                                          ↓
                                      A8 台架/低速
                                          ↓
                                      A9 ObjectNav
```

第一交付点是 **A1–A7：可审阅、可回放、默认不运动的底盘接入实现**。
第二交付点是 **A8：有物理证据的底盘控制闭环**。
第三交付点是 **A9：完整语义导航实物闭环**。

不重复建设传感器驱动、第二个 WebSocket bridge 或已有 SysNav 规划器；当前节点只携带 FWTW
已核对操作所需的最小协议实现。不把已接受的低速相机–LiDAR
外参重新列为本轮 P0。动态时间偏移、高速融合和模型性能增强不属于当前最小接入范围。

负责人分工：代码侧负责 A1–A7 的实现与可复现报告；底盘/现场负责人提供真实模式、急停和
watchdog 语义并参与 A6/A8；任务验证负责人完成 A9。具体人员与现场阈值待项目指派/确认。
