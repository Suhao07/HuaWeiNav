# STRIVE 任务合同与 Harness

本文是维护中的运行合同，说明任务、观测、验证、决策和运动的职责边界，不把某次提交、测试数量或历史 episode 成绩当作当前验收结论。指令语义和证据细节见 [指令编译与验证合同](instruction_adapter.md)。

## 1. 状态所有权

| 所有者 | 合同 | 不负责 |
| --- | --- | --- |
| [`instruction/TaskSpec`](../instruction/contracts.py) | 同一 revision 内不可变的目标、数量、顺序和约束；需求 activity/material/resource 与候选设施假设分离 | 检测词、运动状态和运行进度 |
| [`task/TaskSession`](../task/session.py) | 当前合同、任务服务、独立的 search revision；解释修订创建新 task revision | 改写物理地图或原始标签 |
| [`TaskProgress`](../task/state.py) / [`reducer`](../task/reducer.py) | 唯一权威进度；按有效验证事件接受实例、推进顺序与完成状态 | 把导航到达当作语义成功 |
| [`scene/`](../scene/contracts.py) | 观测快照、实例与视点；任务查询只读原始观测 | 用任务要求替换观测事实 |
| [`verification/`](../verification/contracts.py) | 三值判断、predicate/对象对作用域、图像内容和版本依赖 | 直接修改任务义务或下发运动 |
| [`MotionSupervisor`](../motion/supervisor.py) | 唯一执行槽位、控制 epoch、请求幂等及执行端终态 | 推断语义成功或自行解除安全 HOLD |

`InstructionPlan`/legacy spec 只是 [`TaskSpec 的兼容投影`](../instruction_adapter/task_compat.py)，不能作为另一套进度状态。运行关系为 `q_next = Reduce(q, verified_event; TaskSpec_r)`；对目标 i，计数是不同已接受 UID 集合的基数 `N_i = |A_i|`。

## 2. 决策入口与工具

[Habitat runner](../objnav_benchmark_with_process_obs.py) 提供 `--decision_owner baseline_guided|legacy|agent`，默认 **`baseline_guided`**。它与 `--instruction_adapter_backend` 的解析后端选择是两个独立开关。

| 模式 | 决策边界 |
| --- | --- |
| `baseline_guided` | [GuidedNavigation](../backends/habitat/guided_runtime.py) 保留 baseline 搜索、候选接近、补拍、局部运动和验证闭环；Agent 在恢复及周期性搜索复核等事件上从领域提供的合法选项中选择，不逐步接管所有动作 |
| `legacy` | 保留原导航策略入口，便于兼容和对照；不是没有任务验证的同义词，也不代表某个已冻结历史版本 |
| `agent` | [自由 Agent 工具循环](../backends/habitat/agent_runtime.py) 选择工具序列；仍由领域服务实施运动准入、验证、进度与完成检查 |

[`NavigationTools`](../harness/tools.py) 暴露八个工具：`observe`、`query_scene`、`query_viewpoints`、`submit_motion`、`motion_status`、`cancel_motion`、`verify_target`、`propose_finish`。参数由 [`AgentRuntime`](../harness/runtime.py) 的类型化 schema 约束；模型提交已观察的实例/视点 ID，不自行捏造坐标或完成回执。

`propose_finish` 只读取 reducer 的完成状态与运动静止/占有状态，不写进度。一个候选可接近、已确认语义、任务完成、后端 STOP、官方 evaluator 成功，是不同层面的事实。

运行时加载五份 Skills：

- [understand_instruction](../skills/understand_instruction.md)：理解需求，区分解释与检索修订。
- [search](../skills/search.md)：搜索与候选探索。
- [active_observation](../skills/active_observation.md)：为未知谓词获取新证据。
- [recovery](../skills/recovery.md)：基于真实执行及证据失败恢复。
- [completion](../skills/completion.md)：按领域完成合同收束。

Skills 是决策指导，不授予绕过 verifier、reducer 或运动所有者的权限。

## 3. 修订、快照与迟到结果准入

- 初次语言解析使用 [`InitialInstructionCompiler`](../instruction/compiler.py)：首次解析加最多一次结构修正，真实请求计预算；`parse_retries=0` 将修正权留给上层。provider 错误不当作语义错误修正。
- `search_updates` 只替换检索词并推进 search revision；不改变用户需求、任务进度或证据事实。DINO/SAM 按 [基类 + 当前检索词](../cv_utils/sam.py) 更新词表。
- `interpretation_update` 必须引用原始指令及本轮给出的证据，保持显式需求，编译完整解释；只有 language 来源任务能采用新的 task revision。解释修订与检索修订互斥。
- 工具模式只允许在 `observe`/`query_scene` 配合修订；解释采用后先返回结果，再重选动作。无效解释保留现有任务，不变成候选反证。
- Agent 响应准入检查当前任务、search revision、进度、快照与运动状态；guided choice 在最终提交前还通过 `choice_guard`/`idle_guard` 再检查 control epoch 与空闲条件。
- 解释修订归档旧任务、进度与判断，清理任务局部选择、补拍及先验查询状态；[Habitat 安装逻辑](../backends/habitat/agent_runtime.py) 不清空物理地图或定位。

对象 UID 表示实例身份，[几何依赖](../scene/dependencies.py) 独立描述相关对象和坐标系版本。[归档图像](../verification/images.py) 与请求使用同一次读取的字节；哈希、任务版本及当前观测绑定均须有效。不能将历史关系 accepted、旧图像或稳定 UID 当成无条件的当前 STOP 许可。

## 4. 运动所有权与后端差异

`MotionSupervisor` 保证每个机器人最多一个活动运动请求：`|M_active(robot)| <= 1`。

- 请求带 `request_id` 和观测到的 `control_epoch`；同一请求的重试幂等，不能用同一 ID 提交另一目标。
- 提交前占位；提交/查询结果未知时保留占有，不能认为超时就已停止，也不能直接再派新目标。
- cancel、HOLD 或 resume 改变控制 epoch，使旧提案失效。取消和 STOP 都需要执行端终态确认；HOLD 只能由安全所有者或授权操作显式解除。
- 到达只允许进入后续取证。缺少运动后的当前观测、候选绑定或所需关系/功能证据时，继续等待、补拍或搜索，不能推进成功。

[Habitat 后端](../backends/habitat/agent_runtime.py) 用原导航器/局部规划器执行离散动作；[Isaac 后端](../backends/isaac/runtime.py) 也复用任务、工具和运动所有者，但其 freemap 检查的运动学相机移动不等于动力学机器人行驶。

[ROS InstructionRuntime](../real_robot/sysnav_runtime.py) 复用任务/验证接口与唯一 MotionSupervisor；部署通过推理 worker 把慢决策和验证移出控制 timer。迟到结果按输入版本、control epoch 和相关对象快照依赖准入，运动到达后还需新观测屏障。不要把 Habitat 三种决策模式或自由 Agent 工具入口直接宣称为 ROS 已部署能力。

实物启动、安全开关和参数以 [正式实物入口](real_robot_navigation_entry.md)、[控制合同](real_robot_controller_contract_zh.md)、[参数说明](real_robot_parameter_tuning.md) 为准。**bridge-off 验证只证明真实输入下的规划与速度输出链，不证明底盘已经行驶、到达或完成指令。**

## 5. 预算、审计与评测

`--agent_max_model_calls` 约束 Habitat 非 legacy 路径的初始语言解析/结构修正及后续 Agent 决策总量；Agent 从 `initial_model_calls` 开始累计。其它感知/房间语义、概念匹配、关系和最终验证调用单独统计，视角控制也有自己的预算；当前没有“所有模型共用一个上限”的合同。预算耗尽不产生接受或成功。

[`lvlm_call_tracker`](../llm_utils/lvlm_call_tracker.py) 在发送前记录真实消息、内联图像、模型、生成参数与响应 schema，用 request ID 关联返回；工具 trace 记录提案、结果、修订及失败。真实调用与 cache hit 分开，批量请求不按候选数量虚增；工程失败须保留，不伪装为模型否定判断。产物根目录以运行入口参数为准，详细证据合同见 [指令文档](instruction_adapter.md)。

部署从 [交付 README](../README.md) 开始，运行操作见 [HM3D 评测指南](hm3d_docker_benchmark_guide.md)，任务来源和 success distance 见 [Benchmark provider](benchmark_providers.md)。比较不同模式时固定代码、数据 split/episode、初始观测、模型、镜像和执行预算，记录策略差异；策略变化不能解释成纯目录重构收益。

官方 success/SPL、系统 `instruction_success`/`final_stop_success`、执行到达与工程终止原因分别报告。本文不声明最新通过测试数、模型在线表现或真机闭环完成；这些需要对应冻结版本的独立回放、benchmark 或现场证据。
