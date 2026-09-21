# HuaweiNav 松灵底盘接入：代码实现计划

日期：2026-09-21。依据：[执行清单](real_robot_agilex_execution_checklist.md)及本地源码。
本次交付包含必要代码实现与可离线审计的启动接线；ROS overlay、松灵真实应答和实物安全行为仍待目标设备验证。

## 1. 本轮交付与实现原则

本轮完成：复用底盘通信、读取遥测、接入仲裁后速度、统一软件选源与接管状态、接入现有部署入口。
交付到默认 shadow、可独立验证的底盘后端。实际地面运动按既有执行清单现场验收。

当前实现进度：第一批通信接入、第二批软件模式联动的代码已经落地；第三批完成了
lower planner 与感知点云话题分离并参数化 terrain 输出 frame，第四批完成了 launch/profile 接线、底盘节点合同门禁和未批准合同草案。
现场底盘字段、急停、watchdog、QoS 和 ROS overlay 构建仍待目标设备确认。

- 只实现 Orin 这台车当前使用的 rosbridge 接口，沿用现有 motion ROS 包。
- 控制决策由 SafetyVelocityMux 负责；bridge 负责传输；模式节点负责远端选源与本地模式同步。
- `remote_control_enabled=false` 是默认 shadow 边界；physical 运行同时要求显式打开该参数和现有合同门禁。
- 信任内部 ROS 消息和已解析配置。配置入口验证参数，外部协议入口验证必要字段。
- 检查覆盖可发生的边界：消息断流、连接中断、服务失败、模式切换、坐标不一致。
- 网络异常携带上下文报告并按既有机制重连；配置错误、协议错误和程序错误保留 traceback。
- 根因修复落在产生问题的模块。复用已有状态、action 和 launch，不建立通用底盘框架。
- 注释解释接口语义、顺序约束、坐标和数学模型；函数体保持直接。

## 2. 已确认的根因与最小解法

| 事实 | 根因 | 本轮解法 |
|---|---|---|
| HuaweiNav 输出 TwistStamped，FWTW bridge 订阅 Twist | 接口类型不同 | HuaweiNav bridge 固定订阅 TwistStamped，在 callback 取 `.twist`；省去独立转换节点 |
| 松灵选源与 HuaweiNav 人工接管独立 | 两个控制域缺少状态连接 | 增加一个具体的 AgileX 模式节点，复用现有选源函数与 SafetyState 消费链 |
| motion server 已处理 MANUAL_TAKEOVER | action 终态已具备实现 | 保留现有实现，验证接管能到达 runtime；有实际失败再定位修改 |
| FWTW 键盘主程序启动即 AUTO，注释称启动人工 | 文档与实际入口行为分离 | HuaweiNav 提供统一模式服务；节点重连先选人工源，自动切换要求显式服务调用 |
| 遥测保存 JSON，未驱动本地仲裁 | 观测和控制状态尚未对接 | 模式节点解析固定外部字段并提供带接收新鲜度的 readiness |
| terrainAnalysis 计算 point - vehicle_position | 输入要求公共世界坐标，感知 profile 却使用 body cloud | 分离感知点云和下层点云配置，使用已存在的世界系点云并统一输出 frame |
| Orin 合同描述旧外部 waypoint 控制器 | 合同对应的后端已经不同 | 新增本轮原生下层合同草案，保留历史观察记录 |

实现参考：本地检查的 FWTW 工作树为 `/Users/ricksu/Documents/ChatGPT/FWTW ICRA2027/work/control_update/tools/agilex_api/`。
该路径只是当前机器上的参考副本，不是部署依赖；部署时仍需记录实际来源哈希和未提交差异。代码归属/许可证随来源一并保留。

## 3. 固定数据流与状态归属

```text
pathFollower -> /cmd_vel/autonomy (TwistStamped)
  -> SafetyVelocityMux -> /cmd_vel (TwistStamped)
  -> AgileX bridge: msg.twist -> rosbridge /navflow_cmd_vel
  -> 松灵 mux -> /base_cmd_vel

松灵遥测 -> /platform/chassis/* (String JSON)
  -> AgileX mode node -> /platform/manual_takeover
                      -> /platform/autonomy_enable
                      -> /platform/chassis_ready

操作者 -> /platform/set_autonomy (std_srvs/SetBool)
       -> AgileX mode node -> 松灵 /mux_vel/select
```

上面新增话题是本计划选定的 HuaweiNav 接口，不代表已有接口。
`set_autonomy(true)` 请求自动，`false` 请求人工；服务返回操作结果和失败原因。
软件 HOLD 沿用 `/platform/safe_hold`；硬件急停状态映射取决于实际字段确认。

| 状态/命令 | 写入者 | 消费者 |
|---|---|---|
| 自动候选 | pathFollower | SafetyVelocityMux |
| 仲裁后速度 | SafetyVelocityMux | bridge |
| 松灵自动速度 | bridge | 松灵 mux |
| 本地选源状态与自动授权 | AgileX mode node | SafetyVelocityMux |
| 本地硬件 readiness | AgileX mode node | SafetyVelocityMux |
| 原始底盘遥测 | telemetry relay | 模式节点与录制器 |
| SafetyState | SafetyVelocityMux | 已有 motion server/status provider/runtime |

本车人工驾驶使用松灵 `/run_management/virtual_joy`。现有 `/cmd_vel/manual` 保持已有能力，
本次部署将它与松灵人工源区分：选中松灵人工时，bridge 不再发送自动源。

## 4. 分批代码修改

### 第 1 批：复用通信与遥测，接通仲裁输出

实现位置：`real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/agilex_platform.py`。

| 文件 | 必要工作 |
|---|---|
| `agilex_platform.py` | 单一 rosbridge 连接拥有发布、遥测订阅、mux 服务调用和模式状态；固定 `TwistStamped.twist -> geometry_msgs/Twist` 边界 |
| `agilex_platform.launch.py` | 编排 bridge 参数，不复制通信逻辑 |
| `setup.py`、`package.xml` | 注册节点入口和 `std_srvs` 运行依赖 |
| `SOURCE_NOTICE.md` | 来源、实际参考路径与本地修改摘要 |

具体规则：

1. 类型适配放在 `CmdVelBuffer.callback`：取 `.twist`，更新 monotonic 接收时间，Header 供诊断。
2. HuaweiNav 这一版本只有 TwistStamped 输入，省去兼容类型选择与额外中间话题。
3. 消息各速度分量按现有 rosbridge 格式发送；bridge 保留合同上限裁剪，不增加平滑控制器。
   physical 模式下 bridge 的线速度、角速度和命令 watchdog 不能超过批准合同。
4. 模式服务取代独立文件开关；shadow 只订阅和记录，physical 每次连接先选人工源，自动源只由显式 `set_autonomy(true)` 请求选中。
5. 首轮保持来源传输代码可追溯；没有确认两个实际调用点需要共享实现，不提取通用层。
6. 处理 `OSError`、明确的 WebSocket 关闭/超时异常；未预期异常保留 traceback 并停止 worker。
7. mux add 的失败保留日志，select 应答作为远端 source 可用性的最终判据。

完成标准：shadow 可接收 HuaweiNav 输出并记录；fake rosbridge 能接收转换后的消息和心跳；生产数据链保持关闭。

### 第 2 批：实现一个模式节点并补齐仲裁边界

`agilex_platform.py` 已同时承担 bridge 和模式服务；修改现有 `safety_velocity_mux.py`，并在 `package.xml` 增加 `std_srvs`。

模式节点提供 `/platform/set_autonomy`，复用 mux 连接/选源函数。
选源操作串行执行；ROS 接收回调与网络等待分开，使用 ROS callback group/executor 的现成能力。
保持一个选源连接拥有者，服务与状态变化通过同一处模式转换处理，避免并发读同一个 socket。

节点内的职责边界：

- `request_autonomy(enabled)`：协调一次模式转换并给出服务结果。
- `update_chassis_status(message)`：处理固定的状态字段、接收时间和连接状态。
- `update_selected(message)`：记录实际选源，识别来自外部的人工接管。
- `publish_readiness()`：输出当前已确认的底盘自动控制条件。

函数名称可随既有项目风格调整，职责保持上述范围。直接使用字段和普通函数，不为这些职责定义接口基类。

| 事件 | 执行顺序 | 最终状态 |
|---|---|---|
| shadow 节点启动/重连 | 只连接和订阅，不执行远端控制操作 | HOLD |
| physical 节点启动/重连 | 关闭自动授权，先选择人工源 | HOLD |
| 请求人工 | 关闭自动授权，发送自动源零命令，选择人工源，发布接管状态 | MANUAL；远端失败时返回错误并保持本地关闭 |
| 请求自动 | 选择自动源并确认成功，再更新本地模式 | 自动已授权；实际速度仍由 mux 的目标/传感器/命令条件决定 |
| 外部切人工 | 关闭自动授权，发布人工接管 | MANUAL，沿已有 SafetyState 链结束活动 action |
| 状态断流/连接故障 | 关闭自动授权，readiness=false | 故障/过期；保留原因 |
| 连接恢复 | 接收本连接的新状态，保持自动授权关闭 | 等待显式恢复 |
| 正常退出 | 关闭授权，发送零速度；连接仍可用时请求人工源 | 记录实际选源结果 |

补充时序约束：

- 自动恢复后只接受本次授权之后收到的自动候选，清理 SafetyVelocityMux 的旧候选缓存。
- 同一话题重复 `true` 是幂等操作；授权上升沿才开始新的候选接收周期。
- 恢复授权与提交新导航目标分开。motion server/runtime 已有终态处理按实测保留，避免缓存目标自动重放。
- mode 节点停止发消息时，mux 通过 readiness 的接收超时关闭自动并撤销本地授权；mode 节点重新上线后要求新的显式授权。
- `/mux_vel/selected` 可能是变化才发布的状态；它的有效性按连接代际与实际状态证据处理，不要求机械地每周期收到 selected。

`SafetyVelocityMux` 最小改动：

1. 新增本车启用的 chassis readiness 输入与接收超时；复用现有 STALE_INPUT/CONTROLLER_FAULT 状态。
2. 在自动分支检查 readiness；硬件人工模式仍由松灵执行，软件急停保留最高优先级。
3. 在接管、底盘失联和新的授权周期清理自动缓存并撤销旧授权，保留原有限速策略。
4. 以已有 SafetyState 发布失败原因；先验证 status provider/action/runtime 的现有传播。

完成标准：服务结果、实际远端 selected、本地 SafetyState 和 action 终态一致；断流与重连有可解释结果。

### 第 3 批：修正下层点云与 frame 配置

主要修改：`terrainAnalysis.cpp`、`sysnav_lower_stack.launch.py`、`strive_real_robot_stack.launch.py`、Orin profile。

- 增加独立 lower-cloud 配置，保留 semantic mapping 已使用的 body cloud。
- 下层优先使用已有 `/cloud_registered` 世界系输出；按运行证据核对带宽和 Point-LIO 负载。
- 令 terrain 输出 frame 与输入世界坐标一致；`terrainAnalysis.cpp` 已改为读取 `terrainFrame`，由 launch/profile 传入。默认仍为原有 `map`，现场 TF 核对后再改为实际 frame。
- 核对 localPlanner/pathFollower 的输入输出 frame 和底盘参考点，按同一个世界系贯通配置。
- 读取实际驱动 QoS 后选择匹配的订阅 QoS。只修改确实不匹配的订阅。

若现有世界系输出无法满足下层需求，先记录具体差异，再确定唯一转换点。新增点云转换节点需有实际输入缺口作为依据。

完成标准：非零位置、非零 yaw 的回放中，障碍物相对车辆的位置正确，terrain/path frame 表达实际坐标。

### 第 4 批：配置、合同与现有部署入口

| 文件 | 改动 |
|---|---|
| `real_robot/control/orin26_agilex_controller_contract.yaml` | 新合同草案；填写确认的原生下层接口，未知字段保留待确认，状态 unapproved |
| `real_robot/ros2_ws/src/strive_sysnav_motion/launch/agilex_platform.launch.py` | 提供本车 URI、网卡、遥测、bridge 参数和 physical 合同门禁入口 |
| `setup.py`、`package.xml` | 注册节点入口、`std_srvs` 依赖和 launch 文件 |
| `strive_real_robot_stack.launch.py` | 复用现有 lower-stack 门禁接入底盘节点 |
| `real_robot/profiles/orin26_livox_mid360_d435i.env` | 接入本车配置路径和独立下层点云参数，保留默认 shadow |
| `scripts/run_real_robot_profile.sh` | 在现有入口接入配置检查，复用已有门禁 |

参数归属：限速/加速度上限在 controller contract；机器人连接和话题绑定在本车配置；
launch 从这些来源传递参数。实际端点检查发生在节点入口，不在 shell、launch、callback 各复制一遍。
若当前模板校验已覆盖新合同字段，则保留 `controller_contract.py` 原实现。

沿用 HuaweiNav 现有真实运动开关。FWTW 的环境变量接线归并到现有入口，
模式授权统一由 `/platform/set_autonomy` 表达，避免多个同义开关要求用户重复确认同一件事。
physical 聚合入口同时要求 `ALLOW_REAL_MOTION=true`、`ENABLE_LOWER_CONTROLLER=true`、
`BLOCK_LOWER_CONTROLLER=false`，并保留本轮的 `START_AGILEX_PLATFORM`、
`STRIVE_DRY_RUN`、`STRIVE_LOWER_CONTROLLER_ENABLED` 和批准合同门禁。
`docker_en.sh` 与专用 Docker wrapper 都传递同一组 AgileX、点云、readiness 和运动门禁参数。

完成标准：同一入口可启动只读/影子底盘后端；physical 使用已有合同校验；停止按控制依赖顺序收尾。

## 5. 不确定项与处理顺序

| 待确认内容 | 影响 | 当前可继续的工作 |
|---|---|---|
| `/ranger_status` 的完整 schema、control_mode/fault/急停枚举 | 自动 readiness 的真实判据 | 完成原始遥测、类型适配、软件 selected 联动；硬件映射等待真实样本 |
| 硬件急停与上位机模式的优先级、底盘 watchdog | 地面运动与断网验收 | 完成软件接口与故障传播，现场记录真实行为 |
| 松灵服务重复 add、select 应答语义 | 选源成功与失败判定 | 复用现有流程，采集真实应答后固定分支 |
| 机器人实际部署版本与本地工作树差异 | 复用版本确定 | 固定本地文件哈希，部署时核对远端 |
| 世界系点云发布能力、frame/TF、QoS | 下层规划输入 | 已确认 terrain 要求公共世界坐标，先做配置分离与离线回放 |
| 首次运动速度、加速度、停止阈值 | physical 合同批准 | 保持合同草案，现场确定受限数值 |

软件接入与现场确认独立推进。所有未确认事实写入交付报告，明确影响范围。

## 6. 注释与核心数学模型

公共节点/函数 docstring 写清用途、输入输出、单位和实际可能抛出的异常；私有回调只说明必要语义。
中文注释集中在模式切换顺序、时钟域、frame 和硬件边界，避免逐行翻译代码。

类型适配注释示例：

```python
# 底盘接收无 Header 的 Twist；速度已在 SafetyVelocityMux 完成仲裁。
# Header 仅用于诊断，通信超时使用本机 monotonic 接收时间。
twist = msg.twist
```

watchdog 模型：接收时间为 t_recv，当前单调时间为 t_now，合同超时为 τ：

```text
fresh = (t_now - t_recv <= τ)
```

模式节点内部计时使用自己的 monotonic 接收时间；不对跨主机 monotonic 时间做减法。
ROS 时间用于消息时间戳和 TF；系统毫秒时间只按底盘 ping 协议填 `timeStamp`。

terrain 的空间模型：世界系点云 p_W 与机器人位置 t_W 必须在同一坐标系：

```text
d_xy = sqrt((p_W.x - t_W.x)^2 + (p_W.y - t_W.y)^2)
p_W = R_WB · p_B + t_WB    # 仅在实际需要 body→world 转换时执行
```

速度消息转换模型是恒等映射 `v_out=v_in, ω_out=ω_in`；坐标变换发生在已确认的控制参考系边界。
角度为 rad，线速度为 m/s，角速度为 rad/s。模型注释放在实现该计算的函数旁。

## 7. 精简验证方案

新增测试按行为分为三个组，可用参数化覆盖事件；复用现有 pytest 基础设施。

| 组 | 必需验证 | 执行方式 |
|---|---|---|
| 传输边界 | TwistStamped 内容正确、shadow 无控制发送、超时零速度、错误可见 | 一个测试模块，局部 fake 连接，无通用模拟框架 |
| 模式转换 | 自动成功/失败、人工接管、授权后新候选、断流/重连、mode 节点退出 | 扩展现有 safety 测试并增加具体 mode 用例 |
| 集成链 | frame/QoS、selected→SafetyState→action 终态、唯一速度发布者 | ROS 环境一次组合测试与一个非零位姿回放 |

每批运行改动相关用例。四批完成后运行一次既有 real-robot acceptance 和 ROS overlay 构建。
构建或集成失败后，只重跑受影响部分。文档、参数转抄与函数调用顺序不单独写镜像测试。
本机缺 ROS/硬件时明确报告未执行项，将对应构建/集成放在既有 ROS 环境执行。

现场阶段保留执行清单的台架、低速、人工接管、急停与断网测试；这些行为的通过依据来自真实反馈。

## 8. 提交与交付顺序

1. 通信复用、遥测和 TwistStamped 输入：一个可独立审阅的提交。
2. 模式服务和 SafetyVelocityMux 联动：一个提交，附状态转换测试。
3. 下层世界系点云与 frame 配置：一个提交，附回放证据。
4. 合同草案、launch/profile 和部署说明：一个提交，附组合构建/测试结果。

除实际定位出缺陷外，保持 motion server、sysnav_runtime、导航算法与训练代码原状。
不创建新的 controller 框架、通用事件总线、多底盘插件层或复制一套 FWTW episode runner。

代码阶段完成后报告：改动文件、完成行为、通过的验证、现场待确认项和可复现启动方式。
随后按执行清单完成实车与 ObjectNav 验收；本代码计划的完成不自动表示实车验收完成。
