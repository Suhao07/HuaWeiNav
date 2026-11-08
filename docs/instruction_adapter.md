# STRIVE 指令编译、检索与验证合同

本文维护当前指令链的语义与证据边界，不记录阶段进度或单次实验成绩。环境、镜像与启动步骤见 [从零部署入口](../README.md)、[模型服务部署](lvlm_server_deployment.md)；高层决策和运动所有权见 [Harness 合同](harness_refactor_design.md)。

## 1. 事实源与兼容边界

```text
自然语言 / 后端显式任务输入
  -> TaskSpec（同一 revision 内不可变）
  -> runtime_plan + detector grounding（兼容投影与检索词）
  -> 场景候选 / 接近取证 / 约束及最终验证
  -> TaskProgress + reducer
  -> 完成检查与后端 STOP
```

- [`TaskSpec`](../instruction/contracts.py) 是语义事实源：原始指令、目标、角色、数量、顺序、完成模式和约束不能被检测结果或导航方便性覆盖。
- [`TaskSession`](../task/session.py) 持有当前任务 revision、[`TaskProgress`](../task/state.py)、搜索假设和证据服务；[`apply_verification`](../task/reducer.py) 是验证驱动进度的唯一写入口。新任务 revision 初始化自己的进度，不沿用旧解释的成功许可。
- [`InstructionPlan`](../instruction_adapter/contracts.py) 由 [`runtime_plan`](../instruction_adapter/task_compat.py) 生成，是供查询、grounding、验证和 legacy spec 使用的可丢弃兼容投影，不是第二份语义事实源。其 `diagnostics.task_spec` 保存任务合同的序列化副本。
- 场景快照保留原始观测；任务匹配不修改地图标签。检测类别、搜索先验、候选设施和几何可达性都不能自行成为任务完成事实。

只有 terminal 目标参与完成统计。anchor/support 是参考对象或搜索线索；数量按不同已接受实例 UID 集合计数：`N_i = |A_i|`。顺序任务只允许当前目标的验证事件更新进度，包括补拍和拒绝；all/exhaustive 与 any-success 分别按所有目标或任一目标完成计算。运动到达不增加数量。

## 2. 初始解析与需求建模

### 输入边界

[`compile_instruction_plan`](../instruction_adapter/compiler.py) 将输入适配与语义编译分开：

- 只有后端显式提供的 episode metadata 才进入 [Habitat 任务输入适配器](../backends/habitat/task_input.py)，不让语言编译器访问隐藏 GT。
- 普通 ObjectNav 通过 `adapt_objectnav_task` 安装窄化的 object-goal 任务；不隐式引入 room、relation、sequence 等自然语言语义。
- 自定义自然语言走严格编译。不能解析、含歧义或不支持的硬约束，不得以 dataset 类别静默替换原任务。显式选择的 `legacy`/`rules` 解析后端仍有兼容 fallback，不应与严格语言路径混同。

### 初始调用和修正预算

[`InitialInstructionCompiler`](../instruction/compiler.py) 用于 Habitat 非 legacy 决策入口的初始语言解析，以及 [Isaac 运行入口](../backends/isaac/runtime.py)：

1. 首次解析；只有结构无效时，才允许最多一次完整结构修正。
2. 每次真实请求发出前增加调用计数；修正同样消耗预算，预算不足不发请求。`ambiguous`、`unsupported` 等结果不是反复修正的理由。
3. 显式使用 `parse_retries=0`，让上层编译器管理结构修正，避免 provider 适配器再暗中修一轮。它不表示取消所有传输层重试。
4. `StructuredOutputError` 携带原始响应供结构修正；provider/传输异常记录后传播，拒绝响应也不是语义修正。离线不可解析时不授予任务准入。
5. 此路径安装 plan 时关闭额外的编译期模型 grounding；`initial_model_calls` 传给 Agent 继续累计。这里共享的是**初始解析/修正与 Agent 决策预算**，不是所有感知、关系和最终验证模型的统一上限。

严格结果通过支持的约束类型检查后才含可执行 `TaskSpec`。相关 schema 和 prompt 见 [`prompting/schemas.py`](../prompting/schemas.py)、[`prompting/templates.py`](../prompting/templates.py)；provider 协议见 [`cognav_llm_adapter.py`](../llm_utils/cognav_llm_adapter.py)、[`self_hosted_llm_adapter.py`](../llm_utils/self_hosted_llm_adapter.py)。

### activity、material 与 resource

隐式功能目标先描述需求，而不是猜一个设施名作为唯一答案：

| 内容 | 当前字段与含义 |
| --- | --- |
| activity | 用户要进行的活动 |
| material | `manipulated_objects`：活动操作的材料/对象，不自动成为导航终点 |
| resource | `required_resource`：活动所需能力或资源，是 terminal 目标的语义要求 |
| 设施假设 | `destination_hypotheses`：可能提供资源的设施，只用于检索与候选探索 |

[`instruction/parser.py`](../instruction/parser.py) 将设施假设从目标 `demand` 中分离，作为搜索先验；编译器为 demand 补充硬 `affordance` 约束。运行时设施候选仍标记为 `requires_instance_verification`。某设施可能提供资源，不等于这个已观察实例确实可用；拒绝一个设施实例也不拒绝整个资源需求。

## 3. 检索词与任务解释分别修订

[`perception/detector_adapter.py`](../perception/detector_adapter.py) 将目标及关系参考概念映射为 `ConceptQuery`/检测词，不能添加或删除用户义务。词表映射不是身份确认。

- `TaskSession.search_revision` 只版本化检索假设。`revise_search` 替换指定 concept 的 `detector_terms`，不改 `TaskSpec`、进度或已存证据。
- 任务解释修订需完整结构、原始指令及已提供证据的引用；只对 language 来源任务开放。经编译后采用 `task_revision + 1`，归档旧任务、进度与判断，清理任务局部选择及证据索引。
- 两类修订互斥；采用新解释后重新选动作，不把旧解释下选出的运动直接执行。仅检索词变化不得伪装为语义 revision。
- 修订受任务、搜索、快照及控制状态准入约束；解释修订要求运动空闲且未 HOLD。它不清空物理地图、定位或原始观测。

动态 DINO/SAM 词表由 [`cv_utils/sam.py`](../cv_utils/sam.py) 按“固定基类 + 当前检索词”去重重建；旧动态词不会永久追加到基类，也不重新加载模型。Habitat 观测链使用 `TaskSession.detector_terms`，Isaac 也将基础背景词与任务检索词分开。此能力不要外推为所有 ROS 检测器都支持同样的热更新协议。

## 4. 候选接近、身份与有向关系

必须区分三个权限：**候选许可接近 ≠ 语义已确认 ≠ 允许 STOP**。

[`query_candidates`](../scene/query.py) 允许 `candidate_eligible` 的实例进入接近取证；身份或关系尚未知，不要求先证明再接近。目标候选优先；没有可用目标候选时可从 non-terminal anchor 搜索。没有看到 anchor 是 `unknown`，不是关系反证，也不必禁止接近孤立目标。

[`RuntimeConceptMatcher`](../instruction_adapter/concept_matcher.py) 的检索/匹配结果不能替代最终验证。关系验证通过 [`ConstraintEvaluator`](../instruction_adapter/constraints.py)、[`DynamicRelationService`](../instruction_adapter/relation_verifier.py) 按需执行，不预计算所有物体对：

1. 为 subject/reference 分别绑定实例 UID、所需概念和同帧区域；不能用另一物体的图块替代参考对象。
2. 几何预筛排除明显不可能的对象对，但几何通过不证明语义成立。仅显式允许视觉覆盖的绑定证据可绕过几何预筛，不是任意图片覆盖安全条件。
3. VLM 分别输出 `subject_identity`、`reference_identity`、有向 `relation` 的 `true / false / unknown`。只有两端身份都为 true，才可用 relation 的判断确认该边；身份 unknown 不能被“看起来相邻”补足。
4. 关系缓存依赖对象对、方向、图像内容及绑定信息；更新后的证据必须重新匹配依赖。无共视绑定图像或验证不可用时，不把几何可能性提升为 true。

拒绝作用域必须保持最小：

- `cup_uid on table_uid` 失败属于该有向对象对；不拒绝所有 cup/table，也不自动拒绝 cup 实例的其它配对。
- 属性或身份反证属于具体实例；新实例/新证据可继续验证。
- `AnchorSearchLedger` 表示某 anchor 实例的搜索尝试，不是概念级语义反证。
- `RelationPairLedger` 保存对象对结果；历史 accepted 边只是带依赖的判断，不是永久通行证。

## 5. 当前证据与最终验证

[`FinalInstructionVerifier`](../instruction_adapter/verifier.py) 处理原始指令、当前 active target 的语义义务和实际候选图像。预测标签或接近距离不能替代视觉身份判断；room/attribute 等由约束层委派后仍需证据验证，“已排队验证”不等于约束成立。

### 实例身份与依赖

- 原生地图对象的稳定 UID 来自 [`mapping_utils/representation.py`](../mapping_utils/representation.py)，不是每次用质心重算的几何哈希。`candidate_uid_from_object` 优先使用原生 UID；几何量化签名仅用于没有 UID 的 legacy 输入。
- [`object_dependencies`](../scene/dependencies.py) 独立跟踪坐标系/重置版本及相关实例的标签、几何和房间。稳定身份不意味着几何永远有效；无关对象更新也不应自动使该实例判断过期。
- [`image_input`](../verification/images.py) 一次读取图像，以同一份字节生成请求与 SHA-256 内容归档。文件路径相同不证明内容相同；归档缺失或内容哈希不符时，旧判断不可复用。
- [`EvidenceStore`](../verification/store.py) 按指令、任务 revision、predicate、候选 UID 和 anchor UID 索引；[`VerificationResult`](../verification/contracts.py) 保留证据引用及依赖。历史记录可供审计，但不能直接宣称当前谓词成立。
- 准入是分层的：后端负责当前观测身份、相关对象/控制版本检查；reducer 检查任务 revision、候选 UID、图像依赖与顺序目标。不能声称 reducer 单独检查了所有实时几何。

### affordance 的视觉支持

硬 affordance 必须有**实际发送的当前候选绑定图像**：`observation_identity` 包含当前 `candidate_uid`、`task_revision`、非空 `observation_id`，且对应图像区域有效。每条 affordance 的非 unknown 判断还需视觉理由及 `supporting_observation_ids`，引用范围只能是本次实际发送的候选观测。

缺少当前候选图像，直接要求补拍；空支持、捏造 ID 或缺失理由降为 unknown。历史绑定图像可辅助，但不能替代本次当前候选图像的准入。功能资源的常识、设施标签、提示词中出现“可用”，均不构成 affordance 证据。

### 接受与 STOP

在安装任务的统一验证链中，当前结果须同时满足 `satisfied`、`decision=accept`、`semantic_satisfied`、`view_sufficient_for_stop`，并通过硬约束与当前证据准入，才交 reducer 接受实例。完成数量/顺序要求后，后端再按运动状态执行完成/STOP。

`hard_stop_constraints` 属于 planner/geometry：距离、当前目标投影以及提供时的实例表面对应关系。当前实现会将硬约束未满足的 accept 降为 `need_better_view`；即使记录了 planner infeasibility，也不能据此声称当前代码允许绕过硬条件停止。

`STRIVE_FINAL_VERIFIER=auto` 按调用侧有无 plan 启用；`1` 显式启用，`0` 使验证不可用。关闭、`LLM_OFFLINE=1` 或 Pydantic 不可用时得到 unknown/uncertain，**不 fallback accept**；provider 异常传播为工程失败。无 plan 的 legacy 路径不能作为指令任务成功依据。

## 6. View-control 与安全边界

[`ViewControlState`](../instruction_adapter/view_control.py) 把 `need_better_view` 作为有预算的取证子任务：由 [视角控制器](../navigation/view_verification_controller.py) 调用 [视点策略](../planning/viewpoint_policy.py)，生成可见性、中心度、边界余量、面积和距离等通用视点 proposal，而非编写目标类别特例。

- 同一候选的 objective 文案变化不重置尝试历史；记录 baseline、尝试/观测质量、剩余可行 proposal、无改善轮次及预算。
- `pinned_visual_evidence` 是早期身份参照，`latest`/`best` 是观察历史；它们都不是 STOP 许可。历史图像必须匹配当前候选和任务 revision，关系边也须满足当前对象对与证据依赖；不能因重分割产生新 UID 就无条件继承旧接受结果。
- 视角质量预算耗尽只表示停止软优化；若仍有满足物理停止合同的可行接近 proposal，继续处理物理要求。无改善、无剩余 proposal、模型预算耗尽都不证明成功。
- 合法 accept 不再被独立的“相比 baseline 改善不足”规则推回补拍，但硬约束、证据有效性和任务进度规则始终保留。
- `check_again` 的当前图像可供约束和最终验证复用；旧 ray/voxel 可见性或到达回执不能独自结束带任务的 episode。

视角资源参数由 `STRIVE_VIEW_CONTROL_MAX_ATTEMPTS`、`STRIVE_VIEW_CONTROL_MAX_VERIFIER_CALLS`、`STRIVE_VIEW_CONTROL_MAX_NO_IMPROVEMENT_ROUNDS` 等控制，具体默认值以源码为准。它们不是模型能覆盖的碰撞、运动占有或距离安全许可。

## 7. 产物、调用审计与评测

不要从文档示意路径猜测本次输出根目录；以运行参数和入口实际写出位置为准：

| 内容 | 正式写出入口 |
| --- | --- |
| 初始任务合同、兼容 plan/spec、episode metrics | [Habitat runner](../objnav_benchmark_with_process_obs.py)；其中 `task_spec.json` 是安装时合同，后续修订还须查看 revision trace |
| 初始请求、结构修正、准入结果 | [`InitialInstructionCompiler`](../instruction/compiler.py)，`trace_path` 由运行入口传入 |
| 工具提案、结果、修订与失败 | [`AgentRuntime`](../harness/runtime.py) 及 [guided driver](../backends/habitat/guided_runtime.py) |
| 约束、对象对、概念匹配和进度快照 | [`ConstraintEvaluator.dump_state`](../instruction_adapter/constraints.py) |
| 模型输入、原始响应、解析失败和调用计数 | [`lvlm_call_tracker.py`](../llm_utils/lvlm_call_tracker.py) |

模型请求发送前保存真实 `messages`（包括内联图像字节）、模型名、响应 schema 和生成参数，响应通过 `request_id` 关联；不再是“只保存原始输出”。图像归档和完整请求可能包含现场信息，分享前需检查隐私和体积。`calls` 与 `cache_hits` 分开计数，batch 一次请求不能按候选数当作多次请求；传输失败还应核对请求/错误 trace，不能只看成功响应统计。

官方 evaluator 的 success/SPL 与 `instruction_success`、`final_stop_success` 是不同指标，必须并列保留。后两者是系统自判诊断，不替代 benchmark 官方评分；`accepted_distance_to_target` 与 GT `distance_to_goal` 也不必指向同一实例。输入、距离阈值和 provenance 见 [Benchmark provider 文档](benchmark_providers.md)。

ROS 实物部署见 [正式入口](real_robot_navigation_entry.md) 和 [控制合同](real_robot_controller_contract_zh.md)：bridge-off 下真实感知、规划和速度输出验证不等于底盘已经行驶、到达或任务完成，仿真结果不能替代现场闭环验收。
