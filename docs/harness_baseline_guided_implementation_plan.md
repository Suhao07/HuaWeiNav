# STRIVE：以 baseline 策略为默认能力的 Harness 修改与验证计划

日期：2026-10-25。本文是经源码核对后的实现方案，尚未修改运行代码、未启动新实验。既有 baseline 五例结果继续冻结复用。相关现象见 harness_r7_regression_analysis.md。

## 一、架构原则与明确目标

默认执行原 STRIVE 的探索、候选选择和目标接近策略；Harness 统一上下文、决策归属、工具准入、运动反馈和日志；Agent 在领域策略需要语义选择或恢复时介入。执行旧算法不要求模型逐步批准。

分工应为：

```text
观测与地图更新
  → 任务候选查询 / baseline 探索与视点策略
  → 默认可执行决策 + 有限语义分支
  → Harness 决定本轮采用默认决策还是请求 Agent
  → 唯一 MotionSupervisor → 原局部导航器
  → 终态 / 取证结果 → 地图、搜索状态、TaskProgress 各自更新
```

用 B 表示原领域策略，A 表示 Agent，z 表示本轮明确介入事件：

\[
u_t=\begin{cases}B(s_t,g),&z_t=0\\
A(C_t,\mathcal U_t),&z_t=1.
\end{cases}
\]

两条分支每轮互斥，均由同一个 Harness 向运动所有者提交。模型不回应或解析失败是可记录的错误，不采用“捕获所有异常然后悄悄退回 baseline”的机制。默认执行 baseline 是显式正常路径，不是错误兜底。

先证明能力保留，再验证 Agent 增益。不能承诺单次五例的每条路径都与旧结果完全一致，也不能据此宣称新框架普遍优于旧框架。

## 二、要复用的是具体策略，不是概念上的最近点

### 1. 探索层

复用 `planning/exploration_policy.py`：

- `find_closest_nodes`：现有距离及 prior 配置下的排序；相同距离保留原候选顺序和 argmin tie-break。
- `explore_in_room`：当前房间内可探索节点。
- `find_closest_viewpoint_in_room`：选定房间的入口/探索节点。
- `explore_after_fully_explored`：房间策略耗尽后的未访问节点路径。
- `explore_in_room_relocate`：只在原 relocate 配置启用时使用，不能混入默认 no_relocate 路径。

现有 no_relocate 的实际顺序为“目标候选 → 当前房间探索 → 房间耗尽后的 relocation → 未访问节点”。无 prior 时：

\[
n^*=\arg\min_{n\in\mathcal F(r_t)}d_{\rm nav}(x_t,n).
\]

只有 \(\mathcal F(r_t)=\varnothing\) 才进入下一层策略。首版不增加信息增益、新权重或新的探索温度参数。

注意旧 baseline 默认 `relocate=False`、`gpt_relocate=True`（未传 no_gpt_relocate），不是全程纯规则。已有房间语义选择应保持原配置和调用边界，不再额外套一层 Agent 复议。五例没有 room-policy 模型调用记录，尚不能验证跨房间语义选择能力。

`sim.geodesic_distance` 来自 Habitat NavMesh；本轮为保持对照继续使用，物理后端将来由自己的 planner 提供路径代价。当前 `mapper.get_path()` 仅返回目标节点，不是完整拓扑多段路由；不要以为复用它就增加了路径规划能力。

### 2. 目标选择和接近层

每次语义地图更新后调用现有候选查询，保留实例排除、anchor 角色、pending verified pair 和排序。复用 `backends/habitat/query.py::select_observed_candidate` 中现有绑定逻辑，避免维护 Agent 专用的第二套 matching 规则。

复用 `find_final_waypoint`、`select_check_again_viewpoint` 与 `planning/viewpoint_policy.py`。旧流程设置 need_check_again 后优先沿其复核流程执行；不能把 approach/recheck 作为无质量差别的两个点让模型随意选。

`query_viewpoints` 发布现有评分分量与推荐用途；有 need_better_view 时，将 view_objective 写入已有 view-control 状态，调用原提案更新过程。模型只能选择服务发布的物理 ID，不能自造坐标。

### 3. 底层执行

保留 `navigation/action_controller.py::next_action_to_waypoint`、现有 planner、坐标变换及 MotionSupervisor。模型不下发离散动作，不在 active goal 外另开导航循环。到达终态进入统一 dispatcher；验证成功仍经 reducer，随后才能发任务 STOP。

## 三、最小代码修改面

| 位置 | 必须改变的职责 | 保持的内容 |
|---|---|---|
| `planning/exploration_policy.py` | 必要时分离选点与临时缓冲清理；提供原策略选项/推荐 | 候选定义、距离、tie-break、配置语义 |
| `backends/habitat/agent_runtime.py` | 引入 baseline-guided 默认分支；按事件调用 Agent；恢复候选/视点闭环 | 现有任务 session 与运动所有者 |
| `backends/habitat/query.py` | 让两条入口共享候选绑定，保护同一周期生成的视点 | matcher、ledger、anchor 处理 |
| `harness/runtime.py` | 模型请求只发生在需要选择的事件；写明 decision_source 和触发原因 | typed schema、异常传播、依赖检查 |
| `harness/tools.py` | 提供导航推荐的只读查询；自动呈现最新运动终态 | ID 准入、任务版本与证据边界 |
| `backends/habitat/scene.py` | 导出决策需要的路径代价/房间信息；朝向填真实值或显式未知 | 稳定实例 ID、可序列化几何 |
| `skills/search.md`、`active_observation.md`、`recovery.md` | 写明何时采用领域默认、何时请求语义分支 | 不把数字几何判断转进提示词 |
| benchmark 入口 | 增加明确 guided 模式及配置回执 | 既有选样、身份校验、独立 evaluator |

可以增加一个 Habitat 专用小型策略绑定文件来承接旧主函数的决策部分，前提是旧入口也调用这份逻辑；不新增插件注册系统、PolicyFactory 或通用 DAG。逻辑适合放入既有文件时，直接修改既有文件。

推荐模式为 `legacy`（保留历史入口）、`baseline_guided`（新的默认交付）、`agent`（保留 r7 对照）。legacy 仅作为代码参考和可选择入口，本轮不重新运行其五例。模式在配置与结果中显式记录。

默认分支直接调用共享领域函数；暴露给 Agent 的只读工具可命名 `query_navigation_options`，它只在介入时使用。查询不访问节点、不清理点云、不改变 room 状态、不启动运动。

## 四、需要先处理的状态耦合

### 1. 点云清理不是查询职责

`explore_in_room` 与 relocation 方法会清理 process_obs_pcd/process_nav_pcd。若 Agent 查询两次就清两次，会改变后续建图。先确认旧循环清理的执行时机，再将清理保留在对应单次周期转换中；选点函数只消费同一时刻的数据。

这不是随意删除旧副作用，而是让其发生次数和时序明确。离线差分不能在同一个 live mapper 上连续调用两套有副作用策略。

### 2. selected、arrived、observed 不是同一事实

旧代码在选中目标后调用 change_state，后者将节点设为 EXPLORED 并清除 frontier 标记；r7 在到达后调用。直接照搬旧调用位置会把尚未到达的节点记作已探索。

最小处理：用现有 active goal/selected_node_id 表示本次已选目标，从可选集合中临时排除；实际到达和观测完成后更新访问/覆盖状态。blocked/cancelled 不自动宣称已完成该区域。

\[
\mathcal F_t^{\rm eligible}=\mathcal F_t^{\rm observed}-\{n_t^{\rm selected}\}.
\]

这是显式状态语义修正，需要独立测试，不能包装为字节级 baseline 等价。验证重点是正常成功路径的候选集合、下一决策和实际运动一致，以及失败路径不会永久丢失未探索区域。

核对 agent.current_node_idx 与 mapper.current_node_idx、traj、room 状态的更新位置，给每个事实明确一个写入口。第一版复用地图已有访问状态，不新增一套 room/frontier 全量镜像。

### 3. 共享候选查询会替换 snapshot

当前 `select_observed_candidate` 会重建 session.snapshot，而 `HabitatNavigationTools.refresh_scene` 另有视点保留逻辑。接回候选查询时必须统一刷新入口，否则此前修好的“补拍视点丢失”可能再次出现。查询必须消费已有快照，或由唯一快照发布入口同时维护对象 lookup 与相关视点。

### 4. 视点有效性与质量更新分开

同一实例的视点 ID 不能因机器人移动无故消失；但验证目标改变、几何变化或反馈要求新视角时，必须重新评价/生成提案。保留身份不等于永久缓存最优视点。

## 五、Agent 的介入边界

首版仅考虑已有、明确的事件：

- 当前房间耗尽，需要语义选择下一房间；复用旧 room selector，保持单次选择。
- 多目标/anchor 存在需要语义排序的分支；仍遵守 TaskSpec 顺序约束。
- 导航器报告 BLOCKED 或已记录的明确无进展，需要选择其他已观测方案。
- 原视点过程用尽仍缺证据，需要选择另一候选/观察策略。

普通“当前房间内去最近 frontier”直接执行领域策略。只有一个有效选项也无需发模型请求。

第一阶段可以只恢复原介入边界，关闭新增自由决策；这用于证明能力保留。随后逐个开放增强事件。Agent 不得直接设置完成计数、已到达、已观测或停止成功。

Skills 记录的是过程规则和介入条件，不能只写“未找到就探索”。可写成：先查询已观测目标；没有目标则执行当前房间推荐；本房间耗尽再在提供的房间集合中选择；补拍保留实例绑定并依据新的验证目标请求提案。

## 六、到达后的最小处理

先按 baseline 目的恢复调度，不增加新的智能感知系统：

```python
while not episode_finished:
    if motion.active:
        receipt = advance_original_navigator()
        if receipt.is_terminal:
            process_motion_result_once(receipt)
        continue

    if event_requires_scene_update:
        update_scene_once()
        refresh_candidates_once()

    proposal = baseline_policy.propose(context)
    if proposal.requires_semantic_choice:
        proposal = agent.choose(proposal.options, context)

    execute_through_single_owner(proposal)
```

frontier 到达按旧探索循环更新场景；对象/补拍到达按旧定向取证流程处理。终态自动放进上下文，不请求模型再调用 motion_status 才知道已到达。到达后是否验证由明确任务阶段驱动，验证结果由 reducer 推进；Task STOP 和 waypoint 到达继续区分。

这个循环仍可串行运行。先恢复行为，异步 worker 是之后的独立优化，避免把并发和策略变化混为一个问题。

## 七、验证分层与实际执行次序

### A. 静态和固定输入验证：不跑 episode

1. 固定同一候选集合和路径代价，比较原选点逻辑与提取后逻辑；检查当前房间非空、局部耗尽、全房间耗尽、距离并列、不可达等真实分支。
2. 重复查询推荐，地图、访问状态、点云缓冲与运动状态保持不变。
3. 验证正常到达更新一次；blocked/cancelled 不标记完成探索；目标丢失按原错误协议处理。
4. 候选刷新后 recheck 仍可引用；新 view_objective 能实际改变视点提案。
5. 只有一个推荐、普通局部探索时不调用 decision Agent；语义事件调用一次，错误保留。
6. 保留已有工具参数、稳定 UID、依赖版本、取消和 TaskProgress 回归。

这些是行为测试，不写字段赋值自测或泛化防御测试。旧 node_info JSON 没有保存完整 NavMesh、节点内部状态，不能声称仅凭该文件能精确恢复 geodesic 决策。可用受控候选+距离验证纯选择，用新变体的实际运行补录真实决策。

### B. 最小真 episode 诊断：只运行新变体

| 阶段 | 单独改变 | 样本 | 判别问题 |
|---|---|---|---|
| E1 | baseline 探索/候选自动查询接回，保持 r7 的目标到达流程 | toilet、plant | 首次发现是否提前、搜索路程是否下降、失败是否恢复 |
| E2 | 目标接近与定向取证恢复 baseline 过程 | TV，必要时 chair | 同样早发现目标后，能否减少接近回退和无条件扫描 |
| E3 | 合并 E1/E2，冻结 baseline_guided 版本 | 固定五例 | 整体能力是否回到旧结果水平 |
| E4 | 开放一个明确 Agent 增强事件 | 该事件适用的任务 | 增益是否来自新增机制 |

E1 和 E2 是不同阶段的行为恢复；若要把视点排序与扫描调度的收益分别量化，应再将 E2 拆成两个单因素变体。首轮目标是恢复原能力，不需要穷举所有排列组合。

原 baseline 保持单批冻结，不重复运行。新变体保留源码 SHA、配置、模型、身份和失败回执。过程中禁止按某个场景 ID 写特判，失败仍计入分母。

### C. 最终五例验收指标

以 baseline 的五例 SR=1.0、mean SPL=0.6006675182 作为本小样本的能力保留目标，而不是默认保证。逐例同时检查：

- 完整身份/初始 RGB 与原样本匹配；五例各不相同。
- Habitat success、SPL、平移路程、总步骤和终止原因。
- 首次目标证据步骤、当前房间内选择、区域切换、选中点路径代价。
- 全景次数、定向取证次数、bbox/semantic/decision/verifier 分类调用数。
- 语义介入次数、原因、原推荐与实际选择；普通探索新增 Agent 决策调用应为零。
- 所有差异先定位首个决策分叉，不以平均值掩盖单例异常。

总调用不要求简单等于 baseline：解析调用已经减少，正常感知可能有波动。并发 wall time 不能用于严格速度宣称；如需验证时延，应对新变体固定资源/调度，当前旧运行仍只作带条件参考。

五例结果只能说明本组回归恢复情况。若达不到目标，继续定位而不调低通过标准；达到也不能宣称总体优于 baseline。要检验 Agent 增益，需增加语义/恢复事件确实发生的任务切片，并明确未来新增评测范围。

## 八、实施顺序与交付

建议分为四个可审查修改单元：

1. **提取共享 baseline 策略和状态边界**：候选、选点、清理/访问时序，先固定输入比较。
2. **接入 baseline_guided 默认循环**：统一快照、自动候选查询、唯一运动提交、完整决策回执；跑 E1。
3. **恢复对象视点与到达取证闭环**：共享评分、view-control、定向取证；跑 E2，再冻结跑 E3。
4. **逐事件开放 Agent 增强**：一个机制一个对照，保留原默认选项，不增加每步模型批准。

不需要同时重写指令系统、地图、局部规划器、ROS 后端或创建新的多 Agent 层。每完成一条共享路径再删除它的旧重复实现，避免长期两套算法漂移。文档中的接口名称和伪代码属于实现建议，不代表已落地。
