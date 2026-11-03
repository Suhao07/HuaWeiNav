# STRIVE Harness r7 五例性能回退诊断

日期：2026-10-25。结论状态：**工程闭环已完成；这五例的导航性能回退；各修改对回退的独立贡献尚未经干预实验确定。**

## 1. 对照身份与结论边界

原 baseline 为 `cf8dd8e` 加公开 setter 选样修复；重构提交为 `4217991`，运行修复提交为 `14231db`。最终 Agent 五例使用同一 r7 冻结源码；提交源码与冻结功能一致，仅文档、函数说明有后续补充。模型为 doubao-seed-2-0-pro-260215，镜像 strive-hm3d:local，每例最多 500 步，Agent 决策最多 60 次。

baseline 仅运行一次有效五例并冻结，分析不重跑。两组完整身份和首帧逐例相同；组内五个独立任务。全部进程正常退出，Agent 工具无 failed 事件。toilet/plant 是 episode_limit，均未进入最终目标验证，不是模型调用预算耗尽。

数据源：

- `../outputs/agent_repair_20261025/paired5/combined-results.json`：全部十份结果。
- `../outputs/agent_repair_20261025/paired5/validation.json`：配对核验。
- `../outputs/agent_repair_20261025/paired5/diagnostic-evidence.json`：逐例首次标签、扫描时点、模型调用和逐决策距离统计。
- `../outputs/agent_repair_20261025/agent_r7/source/logs/`：冻结 Agent 代码对应的模型请求、工具回执、图像。
- `../outputs/agent_repair_20261025/paired5/baseline/source/logs/`：baseline 中间产物。

这不是“仅加 Harness、其余行为不变”的实验：高层搜索策略、候选查询触发、目标视点选择、到达后的观测调度、指令编译路径均变了。不能把全部差值归因于 Harness 抽象，也不能外推完整 HM3D 或官方 SuperNav。temperature=0 也不等于服务端生成及感知完全可复现。

## 2. 逐例定位

表内数值为 baseline → Agent；首次出现指地图快照中的目标检测标签，统一下划线/空格，不表示首次在 RGB 中可见。

| 目标 | 成功 | SPL | 步数 | 路程（米） | 首次目标标签步骤 | 完整全景次数 |
|---|---|---|---|---|---|---|
| toilet | 1 → 0 | .4709 → 0 | 128 → 500 | 9.75 → 59.75 | 121 → 未出现 | 5 → 11 |
| chair | 1 → 1 | .4415 → .1975 | 109 → 241 | 8.75 → 19.75 | 101 → 192 | 4 → 7 |
| plant | 1 → 0 | .2636 → 0 | 274 → 500 | 20.73 → 48.99 | 257 → 未出现 | 9 → 12 |
| bed | 1 → 1 | .9811 → .7091 | 87 → 118 | 7.25 → 10.00 | 70 → 87 | 3 → 4 |
| TV | 1 → 1 | .8463 → .5558 | 33 → 75 | 2.75 → 4.22 | 13 → 13 | 1 → 3 |

SR：1.0 → 0.6；平均 SPL：0.6006675182 → 0.2924670952。总步数 631 → 1434，总路程 49.23 → 142.70 米。

\[
\mathrm{SPL}=\frac1N\sum_{i=1}^N S_i\frac{L_i^*}{\max(L_i,L_i^*)}.
\]

旋转增加动作预算和耗时，通常不直接增加 SPL 的平移路程；绕路直接降低 SPL，预算耗尽令该例 SPL 为零。因此“扫描多”与“SPL 下降”之间需要区分直接路程影响和间接预算影响。

五个目标的 canonical_target 和 target_detector_prompts 在两组一致，包括 plant 的 potted_plant。当前证据不支持优先恢复旧的多层语言解析/相似词调用。

## 3. 首要机制：搜索能力从领域策略转给模型，决策信息没有一起转移

### 代码事实

legacy 主入口 `objnav_agent_with_process_obs.py::make_plan_mod_no_relocate`：

1. 全景更新后调用候选选择；
2. 当前房间内优先探索；
3. `planning/exploration_policy.py::find_closest_nodes` 按路径距离排序；
4. 当前房间耗尽后才进入 relocation。

距离实现使用 Habitat `sim.geodesic_distance`。这是仿真 NavMesh 能力；实物应使用自己的地图/规划器提供代价，不能假装机器人拥有该 oracle。

Agent 主入口绕过上述策略。`backends/habitat/scene.py` 导出全部未探索 frontier，但仅提供 UID、坐标、room_id；score 未填写。`harness/tools.py::observe` 没有提供房间语义、当前房间搜索进度、路径长度、frontier 面积或预期信息收益。`skills/search.md` 只要求没有候选时探索，没有定义房间保持和代价排序过程。

模型获得选择权，却没有获得旧策略所依赖的充分信息与默认动作。低层 ShortestPathFollower 只能尽量高效到达被选中的点，不能修复高层目标点选择导致的全局绕路。

### 实际证据

- toilet 第 13、32 步 RGB 哈希两组一致。第 32 步后 baseline 选择约 `[-0.924, 2.125]`；Agent 选择 node:3，走向另一方向，第 59 步 RGB 已不同。这是同状态附近可定位的策略分叉。
- Agent 29 次 frontier 选择中，23 次未选当前提供集合的欧氏最近点，20 次选列表首项。plant 序列为 `1,2,3,4,6,7,8,10,11,13,14,16`。
- toilet 第 59 步选择约 6.93 米外的 node:6，集合中最近点约 1.23 米；plant 第 306 步选择约 7.61 米外的 node:11，最近点约 2.03 米。理由基本是“未发现目标，继续探索”，没有说明更远选择的收益。

**解释边界：**欧氏最近不保证可达路径最短，也不保证信息增益最大；列表首项比例只能提示顺序敏感，不能直接证明模型存在列表偏置。上述结果与“缺少成本和探索记忆导致绕路”一致，尚未完成因果消融。

### 原理更新

将选择分成语义意图和可执行动作评价：

\[
f_t^*=\arg\max_{f\in\mathcal F_t^{\rm reachable}}
\{w_g\widehat{G}(f\mid b_t,g)+w_i\widehat{IG}(f)-w_c C(x_t,f)-w_r R(f,h_t)\}.
\]

这里 G 是目标相关收益，IG 是信息收益，C 是导航代价，R 是重复/低收益访问代价。**这是后续设计表达式，不是当前已实现或已经标定的模型。**首版无需实现所有项：先复用当前房间约束和已有路径代价，让模型只对语义有意义的分支提出偏好。

建议在现有 scene/planning 层增加一个薄的探索选项接口，返回可达 ID、代价、房间关联、访问状态和一个确定性推荐。领域层提供候选和事实，Agent 作为唯一高层决策者消费它们；不并行运行两个互相覆盖的控制者。

## 4. 第二机制：到达事件与全量感知绑定，验证视点选择退化

### 每个目标到达都触发全景

`backends/habitat/agent_runtime.py:145-153` 对所有成功运动调用 `run_observation_mapping_cycle`，既包括 frontier，也包括目标 approach/recheck。后者执行 12 次旋转，再做分割、对象融合、建图和 bbox 复核。

五例完整扫描 22 → 37 次，对应旋转动作 264 → 444，多 180 次。它是动作差的可核算组成部分，不是独立因果贡献：扫描地点/次数本身受到搜索策略影响。

TV 的扫描时点最清楚：baseline 只有第 13 步；Agent 为 13、47、68。两个新增目标到达全景直接消耗 24 次旋转，占该例新增 42 步的一部分。

应按事件区分观测需求：

- 新区域探索到达：更新地图，必要时全景；
- 已知实例验证到达：定向 RGB/depth、可见性、距离证据；
- 遮挡/重定位：由实际证据决定扩展观察。

不需要另建通用调度框架，先在现有 driver 根据 motion mode 区分即可。

### approach 和 recheck 失去了质量信息

`HabitatNavigationTools.query_viewpoints` 内部已经调用旧视点生成器，但对外只发布两个位置，丢弃其可见率、中心度、边界、路径成本和推荐优先级。Agent 在 chair、bed、TV 都先选择 approach。

TV：两组第 13 步都有目标。Agent 先到 approach，在第 48 步取证：距离 0.335 米，bbox 贴左边、偏心、图像很暗；验证返回 need_better_view。随后退回 recheck，第 74 步距离 0.869 米，验证通过。实际图像与返回理由一致：

- `../outputs/agent_repair_20261025/agent_r7/source/logs/harness_agent_integration/episode-4/final_verifier/current_bbox_48.jpg`
- `../outputs/agent_repair_20261025/agent_r7/source/logs/harness_agent_integration/episode-4/final_verifier/current_bbox_74.jpg`

**距离更近不是证据更好。** 应从当前可达视点中选择满足停止约束、预计能清楚看见目标且运动代价较低的点：

\[
v^*=\arg\min_{v\in\mathcal V_t}
\{C(x_t,v)+\lambda U(v)\},\quad
v\in\mathcal V_t^{\rm reachable},\;Q(v)\ge q_{\min},\;d(v,o)\le d_{\max}.
\]

U 表示观测不确定性，Q 表示质量估计。这是设计原则；可直接复用 `planning/viewpoint_policy.py` 的现有评分，先向 Agent 暴露推荐和评分分量。

### 验证反馈还没有真正驱动几何重规划

`query_viewpoints` 一旦存在缓存便直接返回。`need_better_view` 的 view_objective 返回给模型/reducer，但 Agent 绑定没有像 legacy 那样调用 view-control 更新和重新生成提案。当前依靠预生成 recheck 在三例中完成了任务；不能因此认为一般遮挡、多次补拍、关系目标已支持。

稳定 UID 和视点生命周期修复应保留；增加的是语义失效条件：验证目标或几何发生相关变化后重新评价视点。避免把“不丢失视点”误实现为“视点质量永久有效”。

## 5. 第三机制：事实很多，面向决策的上下文不足

`AgentRuntime` 只保留最近四次工具结果；observe 提供全部对象，但不提供结构化 SearchProgress。TaskProgress 记录任务完成情况，不能替代“哪个房间搜过、哪些方向收益低、为什么放弃、当前子目标是什么”。

Agent 用户消息最大长度约 52,645 个字符（plant），其中大量为全量对象、浮点 bbox 和 UID。输入长度是字符数，不是 token 数。与此同时，它没有探索代价和房间语义；这是信息选择问题，不能只通过提高上下文窗口解决。

另有协议缺口：快照 robot_pose 未填实际 orientation，默认单位四元数被作为实际朝向输出；对象 bbox 未携带原始视图时间/坐标系语境。当前模型主要按 ID/坐标选点，尚不能证明这两项导致本轮失败，但应修正为真实值或显式未知，避免以后引入错误方向推理。

建议：

- 完整 SceneSnapshot 继续作为事实源；模型输入改为与任务相关的候选、必要环境摘要、排序后的探索选项和增量事件。
- 候选查询在地图更新时增量运行/按版本缓存，不让模型反复承担“是否需要调用匹配器”的机械判断。
- 在现有运行状态中增加小型 SearchProgress：当前区域/子目标、访问结果、未完成选项、最近低收益原因。先用明确字段，不引入向量数据库或另一个记忆 Agent。
- 不恢复 objects[:32]。目标完整性必须由明确查询保证；输出压缩应按任务关联和覆盖规则设计，不能按发现顺序截断。

## 6. 调用开销分解与异步边界

| 调用类型 | baseline | Agent |
|---|---:|---:|
| instruction_parser | 5 | 0 |
| similar_objects | 5 | 0 |
| concept_match_batch | 22 | 9 |
| final_instruction_verifier | 15 | 5 |
| bbox_object_in_box | 26 | 63 |
| navigation_agent | 0 | 56 |
| 合计 | 73 | 133 |

Agent 最终验证次数更少，部分因为两个失败例从未到验证阶段，不能解释为验证效率提升。bbox 增量主要在 chair（8 → 38）和 bed（8 → 13）。输入路径不同、扫描次数不同都可能增加检测候选，尚不能仅凭次数认定缓存失效。

当前 Habitat driver 是串行的：有 active motion 就只 advance；空闲才调用模型；到达后完整感知；之后模型再决策。因此它没有实现“导航持续执行、Agent 异步认知”的收益。motion_status 工具在 chair/TV 各被模型调用一次，但此时 driver 已经推进到终态，增加了不必要的串行轮次。

推荐首先让终态回执自动进入上下文，保留底层查询 API；完成条件仍由 reducer 和独立 evaluator 保证。随后在确实需要时引入非阻塞模型 worker 与取消屏障。异步只能改善等待时间或反应性，不能保证搜索成功率、路径长度改善；不应作为第一修复项。

本批有并发资源争用，不能用逐例 wall time 做严格速度对比。记录的调用耗时可用于定位等待来源，正式时延比较需要固定资源和相同调度条件。

## 7. 模块调整：保留边界，补回能力，减少机械流程

| 模块 | 调整 | 原因/最小实现 |
|---|---|---|
| instruction / TaskSpec | 保留精简方案 | 五例语义及检测词匹配；复杂指令仍需另验 |
| SceneSnapshot / SceneQuery | 扩展有效决策事实 | 探索候选代价、房间关联、可达性、推荐视点；修正假朝向 |
| exploration_policy | 重新接入 Agent 可调用服务 | 复用房间探索与距离函数，不恢复 mapper 对任务策略所有权 |
| Agent SearchProgress | 增加少量状态 | 维护搜索阶段/访问收益；与 TaskProgress 含义分离 |
| viewpoint_policy / view-control | 恢复反馈到提案的闭环 | 暴露已有质量评分，按最新反馈更新提案 |
| perception scheduling | 修改现有 driver 分支 | 探索需要建图；对象验证优先定向采集 |
| Skills | 从泛化描述改成明确过程知识 | 何时留在当前区域、何时换房间、何时补拍；不写死任务真值 |
| Harness | 自动投递状态，按事件请求模型 | 减少显然的查询/轮询；不把几何算法搬进提示词 |
| MotionSupervisor / evidence / reducer | 保留 | 单一控制权、证据新鲜度、唯一计数仍必要，无证据指向其为此次主要损失来源 |

不建议新增多 Agent 协作、复杂插件生态、独立大模型规划层或全面替换局部导航器。首先把现有能力接完整。

可采用的最小闭环：

```python
on_scene_update(snapshot):
    candidates = query_candidates(task, snapshot, evidence)
    options = existing_exploration_policy.options(snapshot, search_progress)
    context = summarize(task, candidates, options, latest_motion_result)
    # 普通推进使用已选 skill 的明确规则；语义分歧或失败恢复再请求 Agent。
    decision = active_skill.next_action(context) or agent.choose(context)
    motion.submit(resolve_observed_id(decision))

on_motion_terminal(result):
    record_search_outcome(result)
    if result.goal_mode == GO_TO_FRONTIER:
        refresh_scene_with_required_coverage()
    elif result.goal_mode == GO_TO_OBJECT:
        evidence = capture_targeted_observation()
        judgment = verifier.verify(evidence)
        task_progress = reducer(task_progress, judgment)
        if judgment.needs_better_view:
            viewpoints = regenerate_views(judgment.view_objective)
```

伪代码表达建议边界，不代表现有代码已实现；active skill 与 Agent 必须共享同一个决策所有权协议。

## 8. 最小判别实验：复用 baseline，按机制拆开

本轮只做离线日志和源码分析，没有启动新 episode。

1. **搜索选项诊断**：固定 r7 其他行为，给 frontier 提供原房间约束和路径代价/推荐；先跑 toilet、plant。记录首次目标证据时间、搜索路程、SR、跨区域次数。若目标发现仍无改善，应转查感知覆盖与选择逻辑，不能宣称已恢复。
2. **视点推荐诊断**：先只在 TV 恢复旧视点评分的推荐顺序，扫描策略保持 r7。观察是否消除 approach→recheck 回退，记录首次可接受证据位置与路程。失败则检查视点评分/相机朝向，不直接调低验证阈值。
3. **感知调度诊断**：在 r7 的 TV/bed 上仅将对象到达全景改为定向证据，视点选择保持不变。比较旋转动作、检测框调用、验证成功和新鲜度。可能改善成本，但不必然改善路径 SPL。
4. **列表顺序离线试验**：固定一份模型上下文，原序/逆序/按代价序请求同一模型，比较实际选中的物理 frontier；只用于诊断位置偏置，不当作导航成绩。
5. **Harness 行为等价检查**：让原 legacy policy 通过新 Harness 的接口执行同一动作规则，对照既有 baseline，确认“所有权/协议变化”和“策略变化”的差别。新增的是诊断变体，旧 baseline 不重跑。

每个单因素变化得到结果后再决定组合；最终固定一版跑同样五例。五例足以验证本次故障机制，不能证明总体提升；后续才扩充独立样本和重复运行。

## 9. 当前判断

首要问题是策略迁移时丢失了搜索与视点选择所需的决策信息；其次是把所有到达事件都处理成全量感知。协议、状态所有权和证据边界已经改善，但不会自动带来更好的搜索策略。

最值得先做的是：**复用原探索能力作为带代价的选项服务，保留视点质量推荐，按运动目的调度感知。** 再把搜索记忆和模型调度收敛成薄的运行时能力。恢复这些能力应通过清晰接口完成，无需恢复原指令模块的职责混杂。
