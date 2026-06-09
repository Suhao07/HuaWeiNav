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

这两步仍然保留。新增的 `FinalInstructionVerifier` 回答第三个问题：

```text
当前候选实例和当前视角是否满足原始自然语言指令？
```

例如 `I want to watch movie` 的最终确认不再只是 `is this tv`，而是把原始 instruction、`InstructionPlan`、候选实例、bbox 图、crop 图、附近对象和几何事实一起交给 VLM 判断。

### 运行链路

```text
mapper.object_found_no_gpt()
  -> 选择未被 hard-rejected 的候选实例
  -> agent.check_again()
  -> agent.final_check()
  -> FinalInstructionVerifier.verify(raw_instruction, plan, candidate, evidence)
  -> VerificationLedger 记录结果
  -> accept / retry view / reject instance / continue exploration
```

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
    decision: accept | reject_candidate | need_better_view | need_relation_check | uncertain
    confidence: float
    satisfied_constraints: list[str]
    failed_constraints: list[str]
    reason: str
```

如果 `decision=reject_candidate`，ledger 只 hard-reject 当前 `candidate_uid`。如果 `decision=need_better_view`，agent 会复用 STRIVE 的 `whether_to_check_again()` / `check_again()` 进行一次视角优化复核。

### 开关

```bash
export STRIVE_FINAL_VERIFIER=auto  # 默认：仅在存在 InstructionPlan/InstructionSpec 时介入
export STRIVE_FINAL_VERIFIER=1     # 强制开启，要求调用侧传入 instruction plan
export STRIVE_FINAL_VERIFIER=0     # 关闭后保持旧 STRIVE 行为
```

普通 HM3D ObjectNav benchmark 不传 `--enable_instruction_adapter` / `--custom_instruction` 时，mapper 不会设置 `InstructionPlan`，agent 会直接旁路 final verifier，不额外调用 VLM，也不改变原始 STRIVE 停止逻辑。

当 `LLM_OFFLINE=1` 或 LLM client 不可用时，final verifier 会 fallback accept，以免离线 smoke test 被外部服务阻断。
