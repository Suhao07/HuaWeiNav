# 搜索先验引导的小目标搜索排序模块

本文档描述 VLN 当前已经实现的“小目标搜索先验排序”机制。该机制参考
CogNav 中“小目标通常依赖大物体、支撑面、房间或功能区域定位”的设计思想，
但当前仓库落地的是 `InstructionPlan.search_priors` 到
`PriorMapQueryService` 的排序承载层，而不是一套独立的知识库状态机。

核心边界如下：

```text
Search priors can suggest where to search.
Search priors cannot declare what has been found.
Search priors cannot create MotionGoal.
Search priors cannot trigger STOP.
```

也就是说，`remote -> couch / tv / table / living room` 这样的常识只会提升相关
房间、支撑物或 frontier 的搜索优先级，不能把 `couch`、`tv` 或 `table`
当成最终目标。真正的成功仍由在线感知、目标实例匹配、物理停靠合同和 final
verifier 决定。

## 1. 已实现的数据接口

自然语言指令或 benchmark metadata 会被编译为 `InstructionPlan`。当前已经实现的
搜索先验接口位于 `instruction_adapter/contracts.py`：

```text
InstructionPlan
  targets          # terminal targets
  constraints      # relation / attribute / count / room constraints
  search_priors
    room_hints
    support_objects
    affordances
```

形式化地，指令 \(I\) 被表示为：

\[
\mathcal{P}(I)=
\left(
  \mathcal{T},
  \mathcal{C},
  \mathcal{S}
\right)
\]

其中：

- \(\mathcal{T}\)：terminal target 集合，只有这些目标可以触发任务成功。
- \(\mathcal{C}\)：约束集合，例如属性、空间关系、数量、房间限制等。
- \(\mathcal{S}\)：搜索先验集合，只用于调整探索顺序。

当前 `search_priors` 的三类字段为：

\[
\mathcal{S} =
\left(
  \mathcal{R},
  \mathcal{A},
  \mathcal{F}
\right)
\]

其中：

- \(\mathcal{R}\)：`room_hints`，高概率房间，例如 `kitchen`、`living room`。
- \(\mathcal{A}\)：`support_objects`，支撑物、容器或语义锚点，例如 `table`、`couch`、`cabinet`。
- \(\mathcal{F}\)：`affordances`，功能语义，例如 `drink`、`watch`、`read`。

该接口的关键约束是：

\[
\forall s \in \mathcal{S},\quad s \notin \mathcal{T}
\]

搜索先验 \(s\) 可以影响排序，但不能成为最终成功目标。

## 2. 解析来源

当前仓库已经支持两类写入 `search_priors` 的路径：

```text
instruction_adapter/parser_llm.py
  从 LLM 解析结果写入 room_hints / support_objects / affordances。

instruction_adapter/parser_metadata.py
  从 episode metadata 或 instruction contract 写入 room_hints、
  support_context_concepts 和 support_policy。
```

其中 `support_context_concepts` 会进入 `support_objects`，`support_policy`
会进入 `affordances`。这使得小目标搜索可以显式表达为：

```text
terminal target: cup
room_hints: kitchen, dining room
support_objects: countertop, cabinet, dining_table
affordances: drink_container
```

## 3. 排序服务

当前搜索先验由 `prior_map/query.py` 中的 `PriorMapQueryService` 消费。该服务读取：

```text
InstructionPlan.search_priors
PriorMapMemory.current_map()
runtime_context objects / rooms / frontiers
alignment diagnostics
```

并输出：

```text
SearchPriorResult
  room_rankings
  object_rankings
  support_regions
  frontier_biases
  prompt_context
  diagnostics
```

对任意候选 \(x\)，排序分数可以抽象为：

\[
S(x|I,M_t)=
S_{base}(x,M_t)
+
\Delta S_{prior}(x,\mathcal{S})
+
\Delta S_{memory}(x,H_t)
\]

其中：

- \(M_t\)：当前先验地图和在线语义地图形成的只读视图。
- \(H_t\)：运行记忆，包括 observed、visited、verified、rejected 等状态。
- \(S_{base}\)：候选本身的基础分数，例如已有策略分数、距离或可达性。
- \(\Delta S_{prior}\)：由 `room_hints / support_objects / affordances` 引入的先验增量。
- \(\Delta S_{memory}\)：由运行时正负反馈引入的动态调整。

概念匹配采用轻量文本归一化和最大匹配思想：

\[
M(x,\mathcal{Y})=
\max_{y \in \mathcal{Y}}
\operatorname{sim}
\left(
  \operatorname{norm}(label(x)),
  \operatorname{norm}(y)
\right)
\]

其中 \(\mathcal{Y}\) 可以是 room hints、support objects 或 affordance terms。

## 4. 房间排序

对候选房间 \(r\)，当前排序逻辑可以概括为：

\[
S_{room}(r|I)=
\alpha M(r,\mathcal{R})
+\beta G(r,\mathcal{T})
+\gamma L(r)
+\eta U(r)
-\lambda E(r)
\]

其中：

- \(M(r,\mathcal{R})\)：房间标签与 `room_hints` 的匹配分数。
- \(G(r,\mathcal{T})\)：该房间是否包含目标对象的几何先验。
- \(L(r)\)：在线观测是否支持该房间。
- \(U(r)\)：未访问房间奖励。
- \(E(r)\)：已访问、已耗尽或无进展惩罚。

因此，如果指令目标是 `cup`，而 `room_hints` 包含 `kitchen`，厨房类房间会在
room ranking 中获得更高分。

## 5. 对象与支撑区域排序

对先验对象或支撑物 \(o\)，排序分数可以写为：

\[
S_{obj}(o|I)=
\alpha M(o,\mathcal{T})
+\beta M(o,\mathcal{A}\cup\mathcal{F})
+\gamma S_{room}(parent(o))
+\delta L(o)
-\lambda R(o)
\]

其中：

- \(M(o,\mathcal{T})\)：对象是否匹配 terminal target。
- \(M(o,\mathcal{A}\cup\mathcal{F})\)：对象是否匹配支撑物、容器、锚点或功能语义。
- \(S_{room}(parent(o))\)：对象所在房间的先验分数。
- \(L(o)\)：实时观测是否支持该对象。
- \(R(o)\)：对象是否被 verifier 或运行记录拒绝。

支撑区域排序由 `support_objects` 和 `affordances` 共同驱动。例如 `cup` 的
support objects 如果包含 `countertop` 和 `cabinet`，则这些对象会形成
`support_regions`，用于提示 planner 优先搜索附近区域。

## 6. Frontier 排序

当 prior map alignment 可用，并且 target/support object 有可用几何位置时，
frontier \(f\) 会获得额外几何先验：

\[
\Delta S_{frontier}(f)=
\rho
\left(
  w_d \frac{1}{1+d(f,p^*)}
  + w_r S_{room}(r_f)
  + w_o M(o^*,\mathcal{T}\cup\mathcal{A})
  - w_e E(r_f)
\right)
\]

其中：

- \(\rho\)：alignment confidence。
- \(p^*\)：top target 或 support object prior 的位置。
- \(r_f\)：frontier 所属或最近的 prior room。
- \(d(f,p^*)\)：frontier 到目标/支撑物先验位置的距离。
- \(E(r_f)\)：该区域已访问、已耗尽或无进展的惩罚。

若 alignment 不可用，几何 frontier bias 必须降级，不得伪造可执行目标。

## 7. 运行记忆反馈

`prior_map/memory.py` 已经记录运行时状态，包括：

```text
update_from_mapper(...)
update_from_snapshot(...)
mark_room_visited(...)
mark_object_verified(...)
mark_prior_rejected(...)
current_map()
```

运行记忆对候选排序的影响可表示为：

\[
\Delta S_{memory}(x,H_t)=
\lambda_o O_t(x)
+\lambda_v V_t(x)
-\lambda_r R_t(x)
-\lambda_e E_t(x)
\]

其中：

- \(O_t(x)\)：近期在线观测是否支持候选。
- \(V_t(x)\)：候选是否被验证。
- \(R_t(x)\)：候选是否被拒绝。
- \(E_t(x)\)：候选所在区域是否已访问、耗尽或无进展。

因此，搜索排序不是静态常识查表，而是会随探索过程更新。例如某个 `table`
附近已经被搜索且未发现 `remote`，相关区域会被降权；如果某个房间内持续观测到
与目标相关的支撑物，则该房间和支撑区域会被升权。

## 8. 运行边界

当前已实现链路为：

```text
Instruction / metadata
  -> InstructionPlan.search_priors
  -> PriorMapQueryService.query(...)
  -> SearchPriorResult
  -> planning policy adapter / debug context
  -> existing planner
  -> live perception
  -> final verifier
```

边界规则：

1. `search_priors` 只能改变排序，不能生成 `MotionGoal`。
2. `SearchPriorResult` 不能直接发布 waypoint。
3. `support_objects` 和 `affordances` 不能触发 STOP。
4. live perception 与 prior 冲突时，live evidence 优先。
5. final verifier 仍是最终语义成功裁决模块。

## 9. 示例

### 9.1 Remote

```text
Instruction: find the remote
terminal target: remote
room_hints: living room
support_objects: couch, tv, table
affordances: watch_media
search behavior: rank living-room frontiers and couch/tv/table nearby areas first
stop authority: only a detected and verified remote can stop
```

### 9.2 Cup

```text
Instruction: find a cup
terminal target: cup
room_hints: kitchen, dining room
support_objects: countertop, cabinet, dining_table
affordances: drink_container
search behavior: rank kitchen/dining-room frontiers and support surfaces first
stop authority: only a detected and verified cup can stop
```

## 10. 模块对应关系

```text
instruction_adapter/contracts.py
  InstructionPlan / SearchPriors / TargetQuery / Constraint

instruction_adapter/parser_llm.py
  LLM parsed search priors -> InstructionPlan.search_priors

instruction_adapter/parser_metadata.py
  metadata room_hints / support_context_concepts / support_policy
  -> InstructionPlan.search_priors

prior_map/query.py
  room / object / support region / frontier ranking

prior_map/memory.py
  runtime observation / visited / verified / rejected state

prior_map/policy_adapter.py
  apply SearchPriorResult to ranking paths
```

## 11. 参考来源

- CogNav local knowledge base: `/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data/lkb_anchor_prior.json`
- CogNav object affordance rules: `/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data/object_affordance_rules.json`
- CogNav object concepts: `/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data/object_concepts.json`
- Goal-Oriented Semantic Exploration / SemExp: https://arxiv.org/abs/2007.00643
- PONI: Potential Functions for ObjectGoal Navigation with Interaction-free Learning: https://arxiv.org/abs/2201.10029
- osmAG-LLM: Zero-Shot Open-Vocabulary Object Navigation via Semantic Maps and Large Language Models Reasoning: https://arxiv.org/abs/2507.12753
- FloorPlan-VLN: A New Paradigm for Floor Plan Guided Vision-Language Navigation: https://arxiv.org/abs/2603.17437
