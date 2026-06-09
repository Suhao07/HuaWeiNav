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

这些能力不依赖完整 SysNav object-object graph。复杂约束先进入 `InstructionPlan.constraints`，由运行时逐步接入。

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
