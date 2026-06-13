# STRIVE-CogNav Object Navigation 技术白皮书

项目 pipeline 示意图：

![HuaWeiVLN Navigation Pipeline](/home/ubuntu/Downloads/HuaWeiVLN_Nav-Pipeline.png)

## 1. 系统定位

本项目在 Habitat HM3D / HM3D-OVON benchmark 中实现
open-vocabulary object navigation，并扩展了面向复杂自然语言任务的
instruction adapter。核心目标不是把 VLM 放到每一步动作决策里，而是将其
限制在更合适的语义层级：

```text
传感器观测
  -> 目标检测与分割
  -> 三层结构化场景图
  -> 指令解析与概念 grounding
  -> 房间级/对象级语义推理
  -> 局部路径与视角控制
  -> final verifier 判断是否 STOP
```

STRIVE 的关键贡献在于用 `Room / Viewpoint / Object` 三层结构压缩历史
观测，使 VLM 可以在结构化上下文上进行 room-level reasoning，而不是直接
面对长轨迹图像序列。SysNav 进一步强调系统分层：高层语义推理、中层导航
规划、低层运动控制分别承担不同职责，避免将空间一致性和运动安全交给 VLM。

本项目当前实现遵守同一原则：

- VLM/LLM 负责语义编译、概念匹配、关系验证和最终停止判断。
- 几何建图、前沿探索、路径规划和动作执行由确定性模块完成。
- 动态语义边按需计算，避免为所有物体对建立稠密关系图。
- benchmark object-goal 与自然语言 instruction mode 共享 final verifier
  和 view-control 闭环，但 benchmark 的结构化目标不会被复杂自然语言语义污染。

## 2. 任务形式化

ObjectNav 中，智能体在未知室内环境中寻找目标物体实例。时间步 `t` 的观测为：

$$
O_t = \{I_t, D_t, P_t\}, \quad P_t = \langle p_t, R_t \rangle
$$

其中 `I_t` 为 RGB 图像，`D_t` 为深度图或由点云投影得到的深度，`P_t`
为相机位姿。动作空间在仿真中为离散动作：

$$
a_t \in \{\text{move\_forward}, \text{turn\_left}, \text{turn\_right}, \text{stop}\}
$$

真实机器人中，动作输出被替换为中层 waypoint 或 navigation intent，再由
底层控制器转成连续速度：

$$
u_t = (v_x, v_y, \omega)
$$

传统 object-goal benchmark 的成功条件为：

$$
\text{Success} =
\mathbb{1}\left[
a_t = \text{stop}
\land d(p_t, \mathcal{G}) \le d_s
\land t \le T
\right]
$$

其中 `\mathcal{G}` 是目标实例集合，`d_s` 是 benchmark 指定的停止距离。
本项目在该条件外增加 instruction-level final verifier：

$$
\text{Accept} =
\text{semantic\_satisfied}
\land \text{constraints\_satisfied}
\land \text{view\_sufficient\_for\_stop}
$$

对于普通 benchmark，原始指令被编译为结构化目标，例如 `Find the <tv>.`；
对于自然语言任务，原始指令可能包含属性、房间、数量、顺序或空间关系。

## 3. 总体框架

### 3.1 分层架构

```text
+---------------------------------------------------------------+
| User / Benchmark Task                                         |
|   object_category=tv | instruction="I want to watch movie"     |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
| Instruction Adapter                                            |
|   parse -> concept grounding -> execution policy -> constraints|
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
| Perception and Mapping                                         |
|   RGB-D / point cloud -> detection -> masks -> objects         |
|   viewpoint graph -> room graph -> semantic edges              |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
| Planning and Navigation                                        |
|   target selection -> room/frontier exploration -> path         |
|   final verifier -> view-control -> STOP or continue           |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
| Runtime Adapter                                                |
|   Habitat discrete action | Real robot waypoint/cmd bridge     |
+---------------------------------------------------------------+
```

## 4. 指令解析模块

### 4.1 设计原则

指令解析模块是任务编译层，不是导航策略层。它只回答五个问题：

1. 用户要找什么。
2. 哪些约束必须满足。
3. 哪些目标可以终止任务。
4. 哪些对象只是上下文或 anchor。
5. 哪些约束需要运行时验证。

它不编码“TV 常在客厅”“杯子常在桌上”这类常识表。常识可以由 LLM 在
prompt 中生成 search priors，或由 room selection prompt 在运行时使用，但
不能静默写入 parser 规则。

### 4.2 Canonical schema

指令被编译为 `InstructionPlan`：

```python
InstructionPlan:
    raw_instruction: str
    dataset_target: str
    task_type: str
    eval_mode: str
    targets: list[TargetQuery]
    constraints: list[Constraint]
    search_priors: SearchPriors
    execution: ExecutionPolicy
    concept_queries: list[ConceptQuery]
    valid: bool
    diagnostics: dict
```

目标、anchor 和支持物统一表达为 `ConceptQuery`：

```python
ConceptQuery:
    id: str
    name: str
    role: primary | anchor | support | secondary
    detector_terms: list[str]
    aliases: list[str]
    description: str
    negative_terms: list[str]
    terminal: bool
```

只有 `terminal=True` 的 concept 可以触发最终成功。anchor 和 support 只能用于
搜索、关系验证或局部扫描，不能让 episode stop。

约束为声明式结构：

```python
Constraint:
    type: room | spatial | sequence | count | attribute | area | co_occurrence
    subject: str
    relation: str
    object: str
    hardness: hard | soft
    verifier: planner | geometry | vlm | metadata
```

parser 只声明约束，不验证约束。运行时由 `ConstraintEvaluator` 根据
`verifier` 选择几何、VLM、metadata 或执行状态机处理。

### 4.3 编译过程

令自然语言指令为 `x`，可用检测类别集合为 `C_det`，episode metadata 为 `M`。
指令编译可写为：

$$
\mathcal{P} =
\begin{cases}
f_{\text{meta}}(M), & M \text{ contains structured task fields} \\
f_{\text{LLM}}(x, C_{\text{det}}), & \text{otherwise}
\end{cases}
$$

其中 `\mathcal{P}` 即 `InstructionPlan`。随后进行 detector grounding：

$$
g(q, C_{\text{det}}) \rightarrow
\{\text{detector\_terms}, \text{aliases}, \text{negative\_terms}, \text{description}\}
$$

`q` 是 `ConceptQuery`。grounding 结果写入 `plan.json`，不再通过代码中隐藏的
同义词表实现。

### 4.4 Runtime concept matching

编译期 concept 不是运行时 object label。mapper 发现对象实例 `o_i` 后，需要
判断它是否满足 concept `q_j`：

$$
m(q_j, o_i, I, r) \rightarrow
(\text{match}, \text{confidence}, \text{role}, \text{reason})
$$

其中 `I` 是原始指令，`r` 是 concept role。匹配结果按
`(instruction_hash, concept_id, object_uid)` 缓存，避免同一对象反复调用 VLM。

重要区分：

```text
terminal match:
  可作为 final verifier 候选

anchor/support match:
  只能作为搜索参考或 relation verifier 的 anchor
```

### 4.5 支持的任务子集

当前 schema 与执行状态支持：

- 单目标：`find a chair`
- 隐式功能目标：`I want to watch movie`
- 属性目标：`red couch`
- 房间约束：`chair in bedroom`
- 多目标任一成功：`find a chair or sofa`
- 顺序任务：`first find the bed, then the towel`
- 数量任务：`find all chairs` 或 `find three chairs`
- 空间关系：`book on shelf`, `cup near table`, `object inside cabinet`
- anchor-first relation search：先找 anchor 区域，再局部搜索 terminal object

### 4.6 伪代码：指令编译

```text
Algorithm CompileInstruction
Input:
  raw_instruction x
  episode_metadata M
  detector_classes C_det
Output:
  InstructionPlan P

1: if M contains structured task fields then
2:     P <- parse_metadata(M)
3: else
4:     P <- LLM_parse(x, C_det)
5: end if
6: for each target or constraint concept q in P do
7:     q' <- concept_grounding(q, C_det)
8:     write q' back to P
9: end for
10: P.execution <- choose_execution_strategy(P)
11: validate terminal/anchor roles
12: persist P as plan.json
13: return P
```

### 4.7 Prompt 样例：指令解析

```text
You are a semantic compiler for indoor object navigation instructions.

Return strict JSON. Do not solve navigation and do not invent scene-specific
facts. Extract only what the user asks for:
- targets: object concepts mentioned or implied by the instruction.
- terminal=true only for objects that may satisfy the final goal.
- anchors/support objects must be terminal=false.
- constraints: room, spatial, sequence, count, attribute, co_occurrence.
- Do not encode common-sense priors such as "TV is in living room" unless
  the room or context is explicitly stated by the instruction.

Instruction:
  "find a book on a shelf"

Expected structure:
  terminal target: book
  anchor/support concept: shelf
  hard spatial constraint: book on shelf, verifier=vlm
```

### 4.8 Prompt 样例：概念 grounding

```text
You map an instruction target concept to detector vocabulary.

Concept:
  name: shelf
  role: anchor
  instruction: find a book on a shelf

Available detector classes:
  [book, cabinet, bookshelf, table, chair, wall, ...]

Return JSON:
{
  "detector_terms": ["bookshelf", "cabinet", "shelf"],
  "aliases": ["bookcase", "shelving"],
  "description": "storage furniture or shelf-like structure that can hold books",
  "negative_terms": ["book", "table surface"],
  "terminal": false
}
```

这里的 `bookshelf/cabinet` 不是硬编码同义词，而是本次指令、本次 detector
vocabulary 和 role 共同作用下的显式 grounding 产物。

## 5. 建图模块

### 5.1 输入与坐标系

仿真输入为 posed RGB-D：

```python
Observation:
    rgb: H x W x 3
    depth: H x W x 1
    pose: SE3
```

真实机器人计划输入为 RGB/panorama、LiDAR point cloud、SLAM pose 和相机外参。
内部统一为局部地图坐标：

$$
X_c = D(u,v) K^{-1}
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix},
\qquad
X_w = R_t X_c + p_t
$$

其中 `K` 是相机内参，`X_c` 是相机坐标下点，`X_w` 是地图坐标下点。

### 5.2 视觉检测与分割

当前 benchmark pipeline 使用 GroundingDINO + SAM：

```text
RGB image + text prompts
  -> GroundingDINO boxes
  -> SAM masks
  -> mask + depth back-projection
  -> object point clouds
```

对第 `i` 个 mask：

$$
\mathcal{P}_i =
\left\{
T_t \left(
D(u,v)K^{-1}[u,v,1]^T
\right)
\mid (u,v) \in M_i
\right\}
$$

对象节点属性为：

$$
A(v_i^o) =
\{c_i, conf_i, \mathcal{P}_i, bbox_i, I_i, \phi_i\}
$$

其中 `\phi_i` 表示按需推理得到的颜色、材质、功能等属性。

### 5.3 三层场景图

场景表示为：

$$
\mathcal{R} = (\mathcal{V}, \mathcal{E})
$$

节点分为三类：

$$
\mathcal{V}
= \mathcal{V}^{room}
\cup \mathcal{V}^{vp}
\cup \mathcal{V}^{obj}
$$

边分为：

$$
\mathcal{E}
= \mathcal{E}^{r-r}
\cup \mathcal{E}^{r-v}
\cup \mathcal{E}^{r-o}
\cup \mathcal{E}^{v-v}
\cup \mathcal{E}^{v-o}
\cup \mathcal{E}^{o-o}
$$

各层职责如下：

- `Room node`：高层语义和跨房间规划单位。
- `Viewpoint node`：离散化探索位置，连接前沿、对象和房间。
- `Object node`：物体实例、类别、点云、bbox、代表图像和属性。

### 5.4 Viewpoint construction

每个 viewpoint 控制半径为 `\zeta_{cover}` 的区域：

$$
C(v_i^{vp}) =
\{x \in \mathbb{R}^2 \mid \|x - p_i\|_2 \le \zeta_{cover}\}
$$

累计覆盖区域：

$$
C_{\text{prev}} = \bigcup_{v_i^{vp} \in \mathcal{R}} C(v_i^{vp})
$$

若当前观测带来足够新覆盖：

$$
|C_t \setminus C_{\text{prev}}| > \epsilon
$$

则添加新的 viewpoint node。对于有 frontiers 的区域，先聚类 frontier edge
segments，再以 clique 或聚类中心生成候选 viewpoint；对于无 frontier 的区域，
以连通区域中心作为候选。

### 5.5 Object association

新对象点云 `\mathcal{P}_j` 与已有对象 `\mathcal{P}_i` 是否合并，依据类别、
空间距离、点云重叠和历史置信度综合判断。抽象形式为：

$$
\text{merge}(i,j) =
\mathbb{1}\left[
s_c(c_i,c_j)
\lambda_o s_o(\mathcal{P}_i,\mathcal{P}_j)
\lambda_d e^{-\|p_i-p_j\|_2}
> \tau_m
\right]
$$

其中 `s_c` 为类别/概念一致性，`s_o` 为空间重叠或邻近度，`\tau_m` 为合并阈值。
instruction mode 中还会利用 instance ledger 避免被明确拒绝的实例反复成为候选。

### 5.6 Room segmentation

房间分割遵循 STRIVE/SysNav 的结构化思想：墙体或障碍区域将可通行空间分割为
连通区域，每个区域形成 `room node`。实际实现中由
`mapping/room_segmenter.py` 承接主要逻辑，并在失败时回退为单房间模式，避免
room segmentation 异常破坏导航主循环。

房间节点属性：

$$
A(v_i^r) = \{m_i^r, c_i^r, I_i^r, \mathcal{O}_i, \mathcal{V}_i^{vp}\}
$$

其中 `m_i^r` 是 top-down room mask，`c_i^r` 是可选 room category，
`I_i^r` 是代表视图，`\mathcal{O}_i` 是房间内对象集合。

### 5.7 动态语义边

SysNav 的关键思想是 object-object 关系边按需计算，而不是对所有对象对建立稠密图。
对于关系约束：

```text
book on shelf
```

设候选 book 为 `v_i^o`，anchor shelf 为 `v_j^o`。共同观察过两者的 viewpoint
集合为：

$$
\mathcal{V}_{i,j} =
\{v_k^v \mid e_{k,i}^{v-o} \in \mathcal{R}
\land e_{k,j}^{v-o} \in \mathcal{R}\}
$$

系统从这些 viewpoint 中取代表图像或 check_again 图像，让 VLM 判断关系 `\varphi`
是否成立：

$$
h_{\text{VLM}}(I_k, v_i^o, v_j^o, \varphi)
\rightarrow
(\text{verified}, conf, reason)
$$

若成立，则添加动态边：

$$
e_{i,j}^{o-o} \in \mathcal{E}^{o-o},
\qquad
A(e_{i,j}^{o-o}) = \{\varphi, conf, evidence\}
$$

关系失败只拒绝该对象对，不拒绝对象类别：

```text
reject(book_uid, shelf_uid, on)
  != reject(book)
  != reject(shelf)
```

这保证了“找红色椅子看到蓝色椅子”或“book 不在当前 shelf 上”时，系统不会屏蔽
整个 chair/book 类别，也不会无限回到同一个错误实例。

### 5.8 伪代码：地图更新

```text
Algorithm UpdateStructuredMap
Input:
  observation O_t = {I_t, D_t, P_t}
  detector prompts Q
  current map R
Output:
  updated map R

1: boxes, labels, scores <- Detector(I_t, Q)
2: masks <- Segmenter(I_t, boxes)
3: for each detection i do
4:     P_i <- BackProject(mask_i, D_t, P_t)
5:     o_i <- BuildObjectNode(P_i, label_i, score_i, bbox_i)
6:     MergeOrInsertObject(R, o_i)
7: end for
8: UpdateNavigablePointCloud(R, D_t, P_t)
9: UpdateViewpointNodes(R, current_pose, frontier_clusters)
10: UpdateTopologyEdges(R)
11: SegmentOrUpdateRooms(R)
12: UpdateRoomObjectViewpointEdges(R)
13: Persist debug artifacts if enabled
14: return R
```

## 6. 导航策略模块

### 6.1 两阶段策略

STRIVE 的导航策略可以概括为两阶段：

1. 高层：VLM 在 room-level 选择下一探索区域。
2. 低层：传统 frontier/viewpoint 方法在房间内部覆盖探索。

本项目在此基础上增加 instruction-aware target selection、anchor-first relation
search、final verifier 与 view-control。

### 6.2 目标选择

对象候选来自 mapper 的 object nodes。候选过滤顺序为：

```text
object nodes
  -> detector/alias/concept matching
  -> instance rejection ledger
  -> room/attribute/relation/count/sequence constraints
  -> terminal target candidate
```

对于 benchmark object-goal，目标被转换为窄指令：

```text
object_category=tv
  -> InstructionPlan(raw_instruction="Find the <tv>.")
```

这使 benchmark 和自然语言任务共用同一 final verifier 与 ledger 机制。

### 6.3 Room-level planning

在当前房间探索结束但未找到目标时，系统需要选择下一房间。抽象 room utility：

$$
U(r_i \mid g, \mathcal{R})
=
\alpha S_{\text{sem}}(r_i, g)
+ \beta S_{\text{frontier}}(r_i)
- \gamma \tilde{d}(p_t, r_i)
- \eta B(r_i)
$$

其中：

- `S_sem`：房间对象、caption、功能语义与目标的相关性。
- `S_frontier`：房间剩余探索价值。
- `\tilde{d}`：考虑已走步数和回溯惩罚的路径代价。
- `B(r_i)`：已失败或低价值房间的惩罚项。

论文中的 STRIVE 让 VLM 综合语义相关性与 travel cost；SysNav 进一步强调将
VLM 限制在 room-level，以避免其进行细粒度 3D 路径判断。

### 6.4 In-room exploration

房间内部探索主要依赖 frontier 和 viewpoint graph。候选 viewpoint 的价值可写为：

$$
S(v_i^{vp})
=
\lambda_f F(v_i^{vp})
+ \lambda_c C_{\text{novel}}(v_i^{vp})
+ \lambda_o O_{\text{relevance}}(v_i^{vp})
- \lambda_d d(p_t, v_i^{vp})
$$

其中 `F` 表示 frontier 价值，`C_novel` 表示新覆盖区域，`O_relevance`
表示可见对象与任务的相关性。

### 6.5 Anchor-first relation search

对小目标或关系任务，直接等待 detector 发现 terminal object 往往效率低。例如：

```text
find a book on a shelf
```

理想执行过程：

```text
terminal concept: book
anchor concept: shelf
relation: on

1. concept matcher 找到 shelf-like anchor。
2. agent 导航到 anchor 附近。
3. 局部扫描 terminal concept book。
4. 对 book-anchor pair 做 relation verifier。
5. final verifier 用原始指令确认。
```

anchor 不参与 stop。若某个 anchor 附近搜索失败，只记录该 anchor 实例已搜索，
然后继续探索其他 anchor 或房间。

### 6.6 Final verifier 与 view-control

最终停止不是单一 detector 判断，而是统一的 instruction satisfaction verifier：

```text
candidate object
  + original instruction
  + InstructionPlan
  + geometry facts
  + relation edges
  + view-control history
  -> VerificationResult
```

输出字段：

```python
VerificationResult:
    satisfied: bool
    decision: accept | reject_candidate | need_better_view | need_relation_check | uncertain
    semantic_satisfied: bool
    view_sufficient_for_stop: bool
    hard_constraints: dict
    confidence: float
    satisfied_constraints: list[str]
    failed_constraints: list[str]
    view_feedback: str
    preferred_view_goal: str
    view_objective: dict
    reason: str
```

停止条件：

$$
\text{STOP} =
\mathbb{1}[
\text{decision}=\text{accept}
\land \text{satisfied}
\land \text{semantic\_satisfied}
\land \text{view\_sufficient\_for\_stop}
\land \text{hard\_constraints.satisfied}
]
$$

其中 `hard_constraints` 是通用机器人停止合同，例如
`within_final_stop_distance`。它不是对象类别规则；若该约束未满足，除非 verifier
明确给出不可执行或不适用的结构化解释，否则 `accept` 会被转为 `need_better_view`。

`need_better_view` 不是语义失败。它表示目标或关系已经可信，但当前视角不足以
形成可靠停止证据。系统进入 view-control 子任务：

```text
SEARCH
  -> candidate detected
  -> final verifier

SEMANTIC_CONFIRMED
  -> decision=need_better_view
  -> pin candidate_uid / relation edge / candidate_record
  -> pin stable visual reference for target identity

IMPROVE_VIEW
  -> generate class-agnostic viewpoint proposals
  -> navigate to proposal
  -> collect check_again evidence
  -> final verifier re-evaluates view sufficiency
  -> update best_visual_evidence and bounded attempt ledger
  -> if proposals/verifier/candidate budget is exhausted
     pass budget_exhausted and best evidence back to final verifier

ACCEPT
  -> stop
```

view-control 的几何候选不写对象类别规则。候选由可见性、居中、bbox 面积、
边界余量、路径可达性与 final verifier 给出的 view objective 共同影响。最终接受
仍由 verifier 判断，而不是由几何阈值直接决定。若候选视角、verifier 调用或同一
candidate 的 final verifier 调用达到预算上限，控制层把 `budget_exhausted`、
`best_visual_evidence`、remaining proposal 摘要和 attempt history 回传给 final verifier。
连续无改善只表示 `progress_stalled`，用于切换 proposal 或重采样，不表示“没有可行更好视角”。

为了避免点云实例在近距离观察时发生语义漂移，系统将首次
`semantic_satisfied=true` 的图像证据保存为 `pinned_visual_evidence`。该证据只作为
目标身份参照；后续 `latest_visual_evidence` 与当前 stop evidence 仍需由 final verifier
重新判断。对于 benchmark object-goal，`STOP` 也必须由该 verifier 的 `accept` 触发；
legacy 几何可见性检查不能单独结束 episode。

### 6.7 伪代码：导航主循环

```text
Algorithm ObjectNavigationLoop
Input:
  environment Env
  instruction or object_goal G
Output:
  metrics and artifacts

1: P <- CompileInstructionOrObjectGoal(G)
2: Initialize mapper R and execution state E
3: while t < T do
4:     O_t <- Env.observe()
5:     R <- UpdateStructuredMap(O_t, P.detector_terms)
6:     candidate <- SelectTargetCandidate(R, P, E)
7:     if candidate exists then
8:         evidence <- CollectCandidateEvidence(candidate)
9:         result <- FinalVerifier(P.raw_instruction, candidate, evidence)
10:        if result.decision == accept then
11:            if E marks full task complete then
12:                return STOP
13:            else
14:                continue with next subgoal
15:        else if result.decision == need_better_view then
16:            PinConfirmedTarget(candidate, evidence, result)
17:            waypoint <- SelectBetterViewpoint(candidate, result.view_objective)
18:        else if result.decision == reject_candidate then
19:            MarkRejected(candidate)
20:            waypoint <- ExploreOrRelocate(R, P, E)
21:        end if
22:    else
23:        waypoint <- ExploreOrRelocate(R, P, E)
24:    end if
25:    action <- LowLevelPlanner(waypoint)
26:    Env.step(action)
27: end while
28: return timeout
```

### 6.8 Prompt 样例：final verifier

```text
You are the final instruction-satisfaction verifier for an indoor navigation robot.

Original instruction:
  "I want to watch movie"

Candidate:
  uid: tv:c75ccae926db
  label: tv
  role: terminal

Evidence:
  - RGB image with candidate bbox
  - object crop
  - geometry facts: distance_to_object, bbox_area_ratio, center_offset
  - relation edges if any
  - view-control history if any

Return JSON:
{
  "satisfied": false,
  "semantic_satisfied": true,
  "view_sufficient_for_stop": false,
  "decision": "need_better_view",
  "view_feedback": "The candidate is a valid TV, but the final view is weak.",
  "preferred_view_goal": "Keep the TV visible and obtain a clearer centered view.",
  "reason": "The object satisfies the watching-movie instruction, but the stop evidence is not yet strong."
}
```

### 6.9 Prompt 样例：relation verifier

```text
You verify one spatial relation for an indoor navigation robot.

Instruction:
  "find a book on a shelf"

Subject candidate:
  uid: book:56e406d1ac42
  label: book

Anchor candidate:
  uid: cabinet:39f2a38841f1
  label: cabinet
  role: anchor

Relation:
  on

Evidence:
  - shared viewpoint images
  - check_again image
  - geometry hints

Return JSON:
{
  "verified": true,
  "confidence": 0.86,
  "need_better_view": false,
  "reason": "The book-like object is visibly resting on the shelf/cabinet surface."
}
```

## 7. Prompt 与调用管理

Prompt 文本集中在 `prompting/templates.py`，prompt 身份在
`prompting/registry.py` 中维护。每类调用都有稳定 `prompt_id`、`trace_label`
和 schema name，便于日志统计和后续 A/B 测试。

主要 prompt：

| Prompt | trace label | 用途 |
| --- | --- | --- |
| `instruction.parse.v1` | `instruction_parser` | 自然语言任务编译 |
| `concept.grounding.v1` | `concept_grounding` | 指令概念到 detector vocabulary |
| `execution.strategy.v1` | `execution_strategy` | 判断 any-success、sequence、anchor-first 等执行模式 |
| `concept.match.batch.v1` | `concept_match_batch` | 运行时对象实例是否满足 concept |
| `relation.verify.v1` | `relation_verifier` | 动态 object-object 关系验证 |
| `final.verify.v2` | `final_instruction_verifier` | 最终停止判定与 view feedback |
| `bbox.object_label.v1` | `bbox_object_in_box` | bbox 内物体标签复核 |

调用日志保存在：

```text
logs/<save_dir>/episode-0/lvlm_calls/*.json
```

每条记录应包含 prompt id、调用类型、模型、输入摘要、输出 JSON 和解析结果。
工程上应避免在高频循环中逐对象单独调用 VLM；concept matcher 应优先批量调用，
并通过 object uid、crop hash、bbox hash 和 instruction hash 做缓存。

## 8. 实物部署模块设计

实物模式尚未作为完整 runtime 合入主流程。当前设计目标是将真实传感器与底层
规划控制栈适配为 STRIVE 可消费的统一接口，而不是让高层语义模块直接控制底盘。

### 8.1 分层原则

参考 SysNav，实物部署分三层：

```text
High-level Semantic Reasoning
  structured scene representation
  instruction planning
  room/object/relation reasoning

Mid-level Navigation Planning
  room-based exploration
  waypoint generation
  local/global path coordination

Low-level Base Autonomy
  SLAM
  local obstacle avoidance
  path following
  cmd_vel / chassis control
```

高层不直接发布 `/cmd_vel`。它输出 `NavigationIntent` 或 waypoint；底层负责安全、
避障、速度控制和平台差异。

### 8.2 硬件配置

STRIVE 论文中的实物配置：

```text
Base:
  Mecanum wheel platform

RGB sensor:
  Ricoh Theta Z1 360-degree panoramic camera

Spatial sensor:
  Livox Mid-360 LiDAR

Compatibility:
  LiDAR point clouds can be converted to depth maps when simulation-like
  RGB-D input is required.
```

本项目计划支持 RealSense 作为可选相机：

```text
Ricoh Theta Z1:
  适合房间级全景上下文、快速语义观察和 room-level reasoning。

Intel RealSense:
  适合近距离 RGB-D、局部目标确认、小物体几何重建。
  FOV 较窄时需要通过机器人旋转或多视角拼接弥补上下文不足。
```

上层不应判断具体品牌，而应依赖相机模型：

```python
CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray | None
    camera_model: "panorama" | "pinhole"
    intrinsics: dict
    extrinsics: SE3
    fov: dict
    timestamp: float
```

### 8.3 ROS 接口

推荐输入 topic：

```text
/camera/image
  sensor_msgs/Image
  Theta panorama RGB or RealSense RGB

/registered_scan
  sensor_msgs/PointCloud2
  Livox Mid-360 registered point cloud

/camera/aligned_depth_to_color/image_raw
  sensor_msgs/Image
  Optional RealSense aligned depth

/state_estimation
  nav_msgs/Odometry
  SLAM pose in map frame

/tf
  camera, lidar, base_link, map transforms
```

统一观测 contract：

```python
RealObservation:
    rgb: np.ndarray
    pointcloud: np.ndarray
    pose: SE3
    timestamp: float
    camera_model: str
    intrinsics: dict
    extrinsics: SE3
    rgb_pano: np.ndarray | None
    depth_pano: np.ndarray | None
    depth: np.ndarray | None
    depth_valid_mask: np.ndarray | None
    frame_ids: dict[str, str]
```

输出 contract：

```python
NavigationIntent:
    intent_type: "explore" | "approach_object" | "improve_view" | "stop" | "recover"
    target_pose: SE2 | None
    target_object_uid: str
    target_room_id: str
    constraints: dict
    reason: str
    safety_policy: dict
```

### 8.4 LiDAR 与相机融合

LiDAR 点 `X_l` 到相机像素的投影：

$$
X_c = T_{c \leftarrow l} X_l
$$

对于 pinhole 相机：

$$
u = f_x \frac{X_c}{Z_c} + c_x,\qquad
v = f_y \frac{Y_c}{Z_c} + c_y
$$

对于 panorama 相机，可用球面投影：

$$
\theta = \operatorname{atan2}(Y_c, X_c),
\qquad
\phi = \arctan2(Z_c, \sqrt{X_c^2+Y_c^2})
$$

$$
u = W\frac{\theta + \pi}{2\pi},
\qquad
v = H\frac{\frac{\pi}{2}-\phi}{\pi}
$$

对每个 pixel 保留最近深度：

$$
D(u,v) = \min_{X_c \mapsto (u,v)} Z_c
$$

稀疏深度不能当作 Habitat dense depth 的完全替代。小物体附近缺失的深度应保留为
unknown，而不是插值为虚假表面。

### 8.5 实物闭环

```text
ROS sensors
  -> SensorAdapter
  -> RealObservation
  -> PerceptionAdapter
  -> DetectionFrame
  -> RealMapperRuntime
  -> InstructionPlan / SceneGraph
  -> NavigationIntent
  -> ROSNavigationBridge
  -> local planner / path follower
  -> robot base
```

实物模式需要额外处理：

- 时间同步和 TF 外参漂移。
- 稀疏点云导致的小物体深度缺失。
- 动态障碍和人类活动。
- 底层规划失败或路径不可达。
- 相机视角不足时的主动转身与重新观测。
- STOP 前安全距离、底盘 footprint 与可达姿态检查。

## 9. 关键产物与指标

### 9.1 Benchmark metrics

Habitat 原始指标：

```text
success
spl
distance_to_goal
softspl
num_steps
```

instruction-aware 指标：

```text
instruction_success
instruction_decision
instruction_accept_step
accepted_candidate_uid
accepted_relation_edge
accepted_distance_to_target
accepted_distance_source
final_stop_success
final_stop_decision
final_stop_accept_step
final_stop_mode
lvml_call_count_by_type
```

需要注意：复杂 instruction mode 中，Habitat `success=0` 不一定表示语义任务失败；
它只表示 benchmark 原始目标距离指标不满足。应同时查看 instruction-level metrics
和 `final_verifier/result_*.json`。

### 9.2 调试产物

```text
instruction_adapter/plan.json
  编译后的任务计划。

instruction_adapter/runtime_state_*.json
  count/sequence/relation/view-control 状态。

final_verifier/result_*.json
  final verifier 判断、reason、view feedback 和失败约束。

detection/step_*/comb_img_*.jpg
  检测与 bbox 复核图。

lvlm_calls/*.json
  每次 LLM/VLM 调用记录。
```

## 10. 设计边界与后续工作

当前项目已经将 benchmark object-goal 与自然语言 instruction mode 收敛到同一
final verifier/view-control 语义闭环，但仍有可继续优化的方向：

1. RoomEvidence 需要进一步增强。当前 room-level semantic planning 仍依赖对象列表、
   frontier 和部分 prompt，上层语义房间选择还不够系统。
2. 小物体搜索需要更强的 support-region policy。应由 LLM/VLM 生成 search priors，
   不应把 `cup -> table` 等关系写成规则。
3. Dynamic semantic edges 已具备基本机制，但关系证据选择和多视角缓存还可继续强化。
4. 实物模式需要独立 runtime，不应直接复用 Habitat action loop。
5. LVLM 调用仍需持续压缩，尤其是 concept matcher、bbox verifier 和 final verifier
   的重复调用。

总体原则保持不变：

```text
语义由 prompt-first 模块声明和验证；
几何由 mapper / planner / controller 保证；
状态由 ledger 和 execution state 单调维护；
VLM 不承担底层路径规划；
benchmark 与 instruction 共享 verifier 闭环，但保持任务语义隔离。
```
