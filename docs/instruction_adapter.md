# STRIVE 指令解析适配器

## 设计目标

`instruction_adapter` 是 STRIVE 前面的任务编译层，职责是把自然语言或 CogNav episode metadata 编译成可执行的 `InstructionPlan`：

```text
instruction / episode.info
  -> InstructionPlan
  -> detector grounding
  -> STRIVE legacy spec
  -> mapper / agent
```

解析模块只回答五个问题：

```text
用户要找什么？
哪些约束必须满足？
哪些目标可以终止 episode？
哪些信息只是搜索上下文？
哪些约束需要运行时验证？
```

它不保存“TV 常在客厅”这类目标常识表，也不决定房间策略、视点策略或路径规划。常识来源要么是 CogNav 数据集 metadata，要么是 LLM prompt 输出，要么在运行时由 VLM/几何验证。

## 核心 Schema

主合同是 `InstructionPlan`：

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
    valid: bool
    diagnostics: dict
```

`TargetQuery` 表达目标概念：

```python
TargetQuery:
    id: str
    name: str
    detector_terms: list[str]
    aliases: list[str]
    attributes: dict
    min_count: int
    terminal: bool
```

只有 `terminal=True` 的目标能触发成功。支持物、锚点、房间、关系对象只能进入 `constraints` 或 `search_priors`。

`Constraint` 是声明式约束：

```python
Constraint:
    type: room | spatial | sequence | count | area | attribute | co_occurrence
    subject: str
    relation: str
    object: str
    value: Any
    hardness: hard | soft
    verifier: planner | geometry | vlm | metadata
```

解析器不立即验证约束。运行时根据 `verifier` 决定由 planner、几何结构、dataset metadata 或 VLM 处理。

### ConceptQuery

新版本把 terminal target 和 relation anchor 都统一成 `ConceptQuery`：

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

`ConceptQuery` 是指令概念，不是运行时 detector label。编译期 LLM
负责给出 detector queries、语义描述和 negative concepts；运行时
`RuntimeConceptMatcher` 再判断某个 mapper object 是否满足这个概念。

这样 `book/books`、`shelf/bookshelf/cabinet` 这类名称泛化不会写成代码
规则，而是变成可审计的 prompt-first grounding 结果：

```text
instruction concept + available detector classes
  -> ConceptQuery
runtime mapped object + ConceptQuery + instruction role
  -> ConceptMatchRecord
```

所有 concept grounding 结果都会写入 `plan.json` 的 `concept_queries`、
`targets[*].concept` 和 `constraints[*].object_concept`。

## 数据源优先级

### 1. CogNav episode.info

如果 episode 中已有结构化字段，优先复用：

```text
instruction_targets
candidate_targets
target_sequence
room_constraints
min_instance_counts
complex_constraints
task_type
eval_mode
```

这样可以避免用手写规则重新猜 benchmark 的真实语义。

### 2. LLM Prompt Parse

没有 metadata 时，使用 CogNav 风格的 LLM client 做结构化解析。prompt 的原则是：

```text
只抽取指令显式要求的目标和约束；
不要把常识位置硬塞进 plan；
support/context 不能变成 terminal target；
空间关系只声明，不在解析阶段判断成立。
```

例如：

```text
find a cup on the table in the kitchen
```

会被编译为：

```text
target: cup, terminal=True
constraint: cup on table, verifier=vlm
constraint: cup in kitchen, verifier=planner
anchor target: table, terminal=False
```

### 3. Grounding Layer

`tv_monitor -> tv`、`tv -> tv_monitor` 这类问题属于 detector vocabulary grounding，不属于指令解析。

grounding 只把目标概念映射到可检测类别：

```text
TargetQuery.name / aliases / dataset_target
  + available_classes
  -> detector_terms
```

它不会把支持物扩展进终止链。例如“watch a movie”即使 LLM 给出 sofa/couch 作为上下文，sofa/couch 也不能让任务成功。

Grounding 现在分为两层：

```text
Compile-time grounding:
  TargetQuery / Constraint.object
  -> ConceptQuery

Runtime grounding:
  mapper ObjectNode
  -> RuntimeConceptMatcher.match_object(...)
  -> ConceptMatchRecord
```

`Constraint.object` 会像 `TargetQuery` 一样被 grounded。例如 `book on
shelf` 中的 `shelf` 会得到一个 non-terminal anchor ConceptQuery。运行时
不会再用 `label == "shelf"` 判断 anchor，而是问：

```text
Does this observed object satisfy the shelf anchor concept for this instruction?
```

这和 SysNav 的思想一致：语义解释按需发生在对象节点和指令概念之间，
而不是提前写死同义词表。

## STRIVE 接入

当前 STRIVE 主链仍消费 legacy `StriveInstructionSpec`，它由 `InstructionPlan.to_legacy_spec()` 生成：

```text
objnav_benchmark_with_process_obs.py
  -> parser.parse_plan(...)
  -> mapper.instruction_plan
  -> mapper.instruction_spec
  -> mapper.target_list = plan.target_detector_prompts
  -> mapper.target_aliases = plan.target_match_terms
  -> agent.instruct_goal = render_instruction_context(plan)
```

每个 episode 会保存：

```text
logs/<save_dir>/episode-*/instruction_adapter/plan.json
logs/<save_dir>/episode-*/instruction_adapter/spec.json
```

`plan.json` 是新的 canonical 输出；`spec.json` 是兼容旧 STRIVE 代码的视图。

## 支持的可解子集

当前 schema 和编译链支持以下任务类型：

```text
single target: 找到一个明确目标
implicit/function target: 找能满足功能的目标，由 LLM parse + grounding 完成
room constraint: 目标需要在指定房间或区域
multi target any-success: 多个候选目标任一成功
sequential target: 多目标顺序约束，当前 legacy STRIVE 只执行 active target
min count: 至少找到 N 个实例，作为 count constraint 声明
```

这些能力不依赖完整 SysNav object-object graph。运行时由
`ConstraintEvaluator` 和 `InstructionExecutionState` 执行：

```text
InstructionExecutionState
  -> active_target_index
  -> accepted_candidate_uids
  -> rejected_candidate_uids
  -> constraint_status
```

执行语义：

```text
attribute/color/material:
  由 final verifier 基于原始指令、bbox crop、全图上下文判断。

room:
  当前作为显式约束证据交给 final verifier；未来接入 room caption 后可以 hard reject。

min count:
  每个 TargetQuery 维护 accepted_candidate_uids，数量达到 min_count 才终止。

any-success:
  任一 terminal target 通过 final verifier 即完成。

sequence:
  只暴露 active target 给 mapper；当前子目标完成后推进 active_target_index。
```

运行状态会保存到：

```text
logs/<save_dir>/episode-*/instruction_adapter/runtime_state_<step>.json
```

`runtime_state_<step>.json` 现在也包含运行时可观测性字段：

```text
concept_matcher_stats:
  single_llm_calls        # 单对象概念匹配真实 LLM 调用次数
  batch_llm_calls         # 批量概念匹配真实 LLM 调用次数
  batch_items_requested   # 批量调用中实际送入模型的候选对象数量
  cache_hits              # concept/object 级缓存命中次数
  exact_matches           # 无需 LLM 的精确概念匹配次数
lvlm_call_counts:
  calls                   # 按调用类型统计的真实 LLM/LVLM 请求
  cache_hits              # 按调用类型统计的缓存命中
  total_calls
  total_cache_hits
```

这里的 `batch_llm_calls` 是真实请求次数，不是被匹配 object 的数量；因此它能区分
“一次批量问 8 个对象”和“8 次单独问模型”。

## 动态语义边

`instruction_adapter.semantic_edges` 提供 SysNav 风格的按需关系验证接口：

```text
candidate object pair + relation + shared views
  -> geometry hard filter
  -> optional VLM verifier
  -> cached SemanticEdge
```

数学形式：

\[
\mathcal{V}_{i,j}=\{v_k^v|e_{k,i}^{v-o}\in\mathcal{R}\land e_{k,j}^{v-o}\in\mathcal{R}\}
\]

STRIVE 运行时只应对候选对象对调用关系验证器，而不是对所有物体对预计算。验证结果缓存为：

```python
SemanticEdge:
    subject_id: str
    relation: str
    object_id: str
    confidence: float
    verified: bool
    source: geometry | geometry_prefilter | vlm
    evidence_view_ids: list[str]
```

已支持的关系族包括：

```text
on / near / next to / with / inside / under
```

这里的关系归一化是操作符语义，不是目标常识。几何层只负责排除明显不可能的候选；最终语义成立仍由 VLM callback 基于共视角图像判断。

Phase 3 的运行模块：

```text
InstructionSpatialGraph:
  保存 ObjectNodeRecord / ViewNode / object-view 共视索引。

DynamicRelationService:
  对候选对象对执行 geometry prefilter + VLM relation verifier。

ConstraintEvaluator:
  只在 plan.constraints 需要 relation 时调用 DynamicRelationService。
```

### Anchor-First Relation Search

对于 `find a book on a shelf` 这类“terminal target + relation anchor”的任务，
显式空间关系本身就是结构约束；只要 plan 中存在 non-terminal relation
anchor，就会进入：

```text
anchor_first_relation_search
```

该模式的运行逻辑：

```text
1. 先寻找 terminal target 的 ConceptQuery。
2. 如果 terminal target 未出现，寻找未搜索过的 anchor ConceptQuery。
3. anchor 只作为导航参考点，不允许触发 final success。
4. agent 到达 anchor 附近后进行现有旋转/复核/局部观察链。
5. 若没有 terminal target 被接受，写入 AnchorSearchLedger。
6. 后续不再回到同一个 anchor uid，但可以搜索其它 anchor 实例。
7. 一旦发现 terminal candidate，必须先有未屏蔽 anchor 证据；否则不追踪孤立 terminal。
8. relation verifier 对 terminal-anchor 对生成动态语义边。
9. relation verified 后才进入原始指令 final verifier。
```

运行时状态保存在 `runtime_state_<step>.json`：

```json
{
  "concept_matches": {
    "instruction|concept_id|object_uid": {
      "matches_concept": true,
      "terminal_eligible": false,
      "source": "llm"
    }
  },
  "anchor_search_ledger": {
    "instruction|concept_id|anchor_uid": {
      "status": "searched_no_terminal_found"
    }
  }
}
```

`RuntimeConceptMatcher` 会把 object label、几何摘要、对象全图/crop 图一起交给 LVLM；
非精确匹配不会依赖代码里的同义词表。

这个 ledger 只屏蔽一个 anchor 实例，不屏蔽整个 anchor 概念。例如某个
bookshelf 搜索失败后，不会禁止另一个 bookshelf/cabinet-like anchor 被选中。

关系约束的拒绝粒度是对象对，不是类别：

```text
cup on table 失败:
  拒绝 (cup_uid, on, table_uid) 这条关系边。
  不拒绝所有 cup，也不拒绝所有 table。

red chair 失败:
  拒绝这个 chair_uid，因为属性绑定在目标实例上。
```

这和 SysNav 的原则一致：几何是硬过滤，VLM 是软判别，关系边按需生成并缓存。

## 边界

本模块禁止引入：

```text
目标常识硬编码表
固定 room prior 表
完整导航状态机
路径规划策略
视点选择策略
检测模型生命周期
```

这些都属于 STRIVE mapper/agent 或运行时 verifier，不属于 instruction parser。

## Final Instruction Verifier

STRIVE 原本已有两类确认：

```text
check_again: 从更好视角复核 bbox 内是否是目标类别
final_check: 用几何 ray/voxel 判断目标是否可见
```

普通 ObjectNav benchmark 仍完整保留这两步。指令模式下，
`check_again` 会变成 evidence-only：它只保存更清晰的 bbox 图和几何事实，
不再单独调用 `check_again_object_in_bbox()` 做语义结论。新增的
`FinalInstructionVerifier` 统一回答：

```text
当前候选实例和当前视角是否满足原始自然语言指令？
```

例如 `I want to watch movie` 的最终确认不再只是 `is this tv`，而是把原始 instruction、`InstructionPlan`、候选实例、bbox 图、crop 图、附近对象和几何事实一起交给 VLM 判断。

### 运行链路

```text
mapper.object_found_no_gpt()
  -> 选择未被 hard-rejected 的候选实例
  -> agent.check_again() 采集更清晰 bbox evidence
  -> instruction mode 下立即复用该 evidence 做约束/原始指令/视角质量验证
  -> 如未提前完成，再进入 agent.final_check()
  -> FinalInstructionVerifier.verify(raw_instruction, plan, candidate, evidence)
  -> VerificationLedger 记录结果
  -> accept / retry view / reject instance / continue exploration
```

`check_again` 的图像证据会直接传给 `ConstraintEvaluator`。对于
`book on shelf` 这类关系任务，关系 verifier 会优先用这张清晰 bbox 图
判断目标实例和 anchor 实例之间是否存在动态语义边，而不是等下一次 stop
时重新构造证据。

### 实例级屏蔽

候选目标按实例管理，而不是按 detector 类别管理：

```python
CandidateInstance:
    uid: str
    detector_label: str
    canonical_label: str
    centroid: list[float]
    bbox_2d: list[float]
    confidence: float
    step: int
```

当前 STRIVE `ObjectNode` 没有永久 object id，因此 `uid` 由类别、点云中心和点云尺寸的量化签名生成。这样可以解决：

```text
指令：find the red chair
检测器：只能检测 chair
现象：先找到 blue chair
处理：只 hard-reject blue chair 这个实例，不 reject chair 类别
```

`VerificationLedger` 的 key 是：

```text
instruction_hash + candidate_uid
```

状态包括：

```text
accepted: 可以终止
rejected_hard: 明确不是目标实例，后续跳过该实例
rejected_soft: 证据不足或暂时不满足，允许后续新证据再评估
needs_better_view: 触发 STRIVE 现有视角优化复核
```

mapper 在候选选择阶段只跳过 `rejected_hard` 实例，不屏蔽同类其它实例。

### Evidence

final verifier 的证据包由 agent 构建：

```text
current_rgb_with_bbox_path: 当前视角 bbox 图
object_crop_path: bbox crop 图
centered_view_path: 当前实现复用 bbox 图，后续可替换为全景居中图
geometry:
  bbox_xyxy
  bbox_center_norm
  bbox_area_ratio
  visible_projected_points
  distance_to_object
view_quality_facts:
  bbox_center_norm
  bbox_area_ratio
  visible_projected_points
  distance_to_object
  center_offset_norm
  border_margin_norm
  target_position_hint
nearby_objects: 2m 内附近对象摘要
room_context: 预留房间上下文
relation_evidence_paths: 预留空间关系证据图
```

输出保存在：

```text
logs/<save_dir>/episode-*/final_verifier/evidence_<step>.json
logs/<save_dir>/episode-*/final_verifier/result_<step>.json
logs/<save_dir>/episode-*/final_verifier/current_bbox_<step>.jpg
logs/<save_dir>/episode-*/final_verifier/object_crop_<step>.jpg
```

### Prompt-First 输出

代码只提供候选实例和事实证据，不写目标常识规则。VLM 输出严格 JSON：

```python
VerificationResult:
    satisfied: bool
    semantic_satisfied: bool
    view_sufficient_for_stop: bool
    decision: accept | reject_candidate | need_better_view | need_relation_check | uncertain
    confidence: float
    satisfied_constraints: list[str]
    failed_constraints: list[str]
    view_feedback: str
    preferred_view_goal: str
    reason: str
```

最终 stop 必须同时满足：

```text
semantic_satisfied == true
view_sufficient_for_stop == true
decision == accept
```

如果语义已经满足但当前视角不足，VLM 应返回：

```text
semantic_satisfied=true
view_sufficient_for_stop=false
decision=need_better_view
```

agent 会进入通用 `ViewControlState`，而不是只做一次 retry。final verifier
会输出或补全 `view_objective`：

```python
view_objective:
    keep_visible_roles: list[str]
    improve_goals: list[str]
    minimum_expected_improvement: str
    accept_if_no_better_view: bool
    reason: str
```

`whether_to_check_again()` 不理解具体物体类别，只对候选路径上的相机姿态生成多个通用 proposal：

```text
visibility
centerability
border margin
projected area
distance to mapped instance
```

`ViewControlState` 会记录：

```text
baseline_quality
attempted proposals
observed_quality
improvement_over_baseline
remaining_feasible_proposals
```

如果 verifier 在 better-view 子目标中想 `accept`，但本次证据相对 baseline
没有足够改善且还有未尝试 proposal，agent 会把结果转回 `need_better_view`。
这一步是通用执行闭环，不是语义判断；`book/shelf/chair` 等对象语义仍由
InstructionPlan、ConceptMatcher、relation verifier 和 final verifier 决定。final verifier
在下一次调用时会看到完整 `view_control` 历史，并可在 proposal exhausted 时决定是否 limited accept。

如果 final verifier 第一次就 `accept`，agent 仍会做一次通用 initial-accept
deferral：几何层预测仍存在明显更好的可达证据视角时，先进入 `ViewControlState`
采集更好 evidence；如果没有明显更好视角，才保留 verifier 的 accept。这个机制只比较
通用 view quality，不写目标类别规则。

进入 better-view 后，`ViewControlState` 会 pin 已经验证过的动态语义边：

```text
candidate_uid + relation + anchor_uid
```

后续换视角时，如果 mapper 把同一语义区域切成新的 object uid，约束层会优先复用
`pinned_relation_context`，把当前证据视为“同一已接受语义区域的新视角”，只重新判断
view sufficiency。这样不会因为靠近后实例重分割而丢掉 51 步已经确认的
`book on shelf`。

另外，`check_again` 图像是强视觉证据；当它存在时，object-object 几何预筛失败会降级为
`geometry_inconclusive` 并允许 VLM relation verifier 覆盖，而不是直接
`reject_relation`。旧的 geometry failure cache 在这种情况下也会被绕过。

为了防止 VLM 把“语义正确但目标贴边/太小/太偏”的证据直接 accept，final verifier
后还有一个可配置的通用 view guard。它只读取 `view_quality_facts`，默认要求最终
stop 证据不要投影失败、不要过度偏离中心、不要过度贴边、不要过小。相关环境变量：

```bash
export STRIVE_FINAL_VIEW_GUARD=1
export STRIVE_FINAL_VIEW_MAX_CENTER_OFFSET=0.35
export STRIVE_FINAL_VIEW_MIN_BORDER_MARGIN=0.08
export STRIVE_FINAL_VIEW_MIN_BBOX_AREA=0.003
export STRIVE_VIEW_CONTROL_MIN_IMPROVEMENT=0.08
```

这些阈值不是目标规则；它们是相机停止证据的几何硬约束，和 SysNav 中“几何硬约束 +
VLM 软推理”的分层原则一致。

如果 `decision=reject_candidate`，ledger 只 hard-reject 当前 `candidate_uid`。

空间关系失败不会 hard-reject 目标实例。运行时会写入
`RelationPairLedger`，只拒绝当前 `(subject_uid, relation, anchor_uid)`：

```text
book_uid X on cabinet_uid Y -> rejected_relation
```

这避免“一个 cabinet 上没有目标”导致所有 cabinet 被屏蔽，也避免“一个
book 不满足关系”导致所有 book 被屏蔽。

### 开关

```bash
export STRIVE_FINAL_VERIFIER=auto  # 默认：仅在存在 InstructionPlan/InstructionSpec 时介入
export STRIVE_FINAL_VERIFIER=1     # 强制开启，要求调用侧传入 instruction plan
export STRIVE_FINAL_VERIFIER=0     # 关闭后保持旧 STRIVE 行为
```

普通 HM3D ObjectNav benchmark 不传 `--enable_instruction_adapter` / `--custom_instruction` 时，mapper 不会设置 `InstructionPlan`，agent 会直接旁路 final verifier，不额外调用 VLM，也不改变原始 STRIVE 停止逻辑。

当 `LLM_OFFLINE=1` 或 LLM client 不可用时，final verifier 会 fallback accept，以免离线 smoke test 被外部服务阻断。

## 可观测性和 Metrics

### Raw LVLM Response

每个 episode 会初始化独立的 LVLM trace 目录：

```text
logs/<save_dir>/episode-*/lvlm_calls/
```

真实 LLM/LVLM 请求会保存为：

```text
0001_instruction_parser.json
0002_concept_grounding.json
0003_concept_match_batch.json
0004_relation_verifier.json
0005_final_instruction_verifier.json
...
```

每个文件只保存：

```text
kind
time
metadata
raw_response
```

prompt 中的大图 base64 不会写入 trace 文件；视觉证据图仍由
`final_verifier/`、`detection/` 等模块按原路径保存。这样可以审计模型原始
JSON/JSON-like 输出是否稳定，同时避免日志体积过大。

### BBox 复核缓存

`ask_gpt_object_in_box()` 现在按 crop 图像 hash、bbox 和 detector class 列表缓存结果：

```text
logs/<save_dir>/episode-*/detection/object_box_cache.json
```

同一 step、同一 crop、同一 bbox 再次触发时会直接复用结果，并在
`lvlm_call_counts.cache_hits.bbox_object_in_box` 中累计命中数。这个缓存只减少重复
类别复核调用，不改变普通 benchmark 或 instruction mode 的决策边界。

### metrics.csv

Habitat 原始 `success` 仍保留，表示 ObjectNav 距离目标的官方指标。指令模式额外写入：

```text
instruction_success       # final verifier + 约束执行链是否接受原始指令
instruction_decision      # 最后一次 final verifier 决策
instruction_accept_step   # 指令级成功发生的 step
accepted_candidate_uid    # 被接受的 terminal instance uid
accepted_relation_edge    # 被接受的动态语义边，JSON 字符串
lvlm_call_count_by_type   # calls/cache_hits/total_calls/total_cache_hits，JSON 字符串
lvml_call_count_by_type   # 兼容旧拼写的别名，内容同上
```

因此复杂指令实验要优先看 `instruction_success`。`success=0` 可能只是 Habitat
原始目标类别指标未命中，并不代表自然语言指令验证失败。
