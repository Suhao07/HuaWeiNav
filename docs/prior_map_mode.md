# STRIVE 先验地图模式技术文档

本文档定义 STRIVE 的先验地图模式。目标是在 Habitat/HM3D 仿真和真实机器人模式中
复用同一套先验地图 contract，让预先构建的房间、拓扑、物体和区域信息作为搜索先验
进入 STRIVE，而不是创建第二套导航系统。

核心原则：

```text
Prior map provides context and ranking evidence.
Runtime semantic map provides observed facts.
Planner/controller owns physical navigation.
Final verifier owns task success.
```

也就是说，先验地图只能影响候选排序、prompt context、搜索区域优先级和调试解释；
不能直接写 `NavigationIntent.goal_pose`，不能直接触发 `STOP`，也不能绕过
`InstructionPlan`、`ConstraintEvaluator` 或 final verifier。

## 1. 设计目标

先验地图模式需要覆盖两个运行目标：

1. 仿真模式：Habitat/HM3D 中读取已有 floorplan、OSM、HM3D top-down map 或离线重建地图，
   用于 room/frontier/object 搜索排序。
2. 实物模式：真实机器人中读取预先构建地图、人工标注地图或 VLM 重建地图，并和
   `SemanticMapSnapshot` 的在线观测对齐。

两种模式共享同一中间层：

```text
PriorMapData
  -> PriorMapMemory
  -> PriorMapQueryService
  -> SearchPriorResult
```

仿真和实物的差异只在 runtime observation 来源：

```text
Habitat mapper state
  -> PriorMapMemory.update_from_mapper(...)

Real robot SemanticMapSnapshot
  -> PriorMapMemory.update_from_snapshot(...)
```

## 2. 外部参考

`Zero-shot-VLN` 中的 `prior_map/` 是主要参考：

```text
prior_map/prior_map_data.py
  PriorMapData / PriorRoom / PriorObject / ObservationRecord / PriorMapMemory

prior_map/map_describer.py
  OSM-like XML 和自然语言地图摘要

prior_map/som_visualizer.py
  Set-of-Marks 多尺度可视化

prior_map/vlm_reconstructor/
  从户型图图片重建 PriorMapData
```

这些模块的思想可复用，但不应直接照搬旧仓库中的 Episode mixin。STRIVE 当前已经有
`instruction_adapter`、`planning`、`real_robot/contracts.py` 和 final verifier 边界，
先验地图必须按这些边界重新封装。

## 3. 总体数据流

### 3.1 仿真模式

```text
Instruction / benchmark target
  -> InstructionPlan
  -> Habitat observation + mapper state
  -> PriorMapMemory.update_from_mapper(...)
  -> PriorMapQueryService.query(plan, runtime_context, memory)
  -> SearchPriorResult
  -> room_policy / exploration_policy / target_selection_policy ranking
  -> Habitat controller executes waypoint/discrete action
  -> FinalInstructionVerifier decides STOP
```

### 3.2 实物模式

```text
Instruction
  -> InstructionPlan
  -> SysNav object/room topics
  -> SemanticMapSnapshot
  -> PriorMapMemory.update_from_snapshot(snapshot)
  -> PriorMapQueryService.query(plan, snapshot, memory)
  -> SemanticMapSnapshotPolicyContext(..., prior_context)
  -> existing planning policy / intent adapter
  -> NavigationIntent
  -> MotionGoal
  -> /way_point
  -> NavigationStatus / ViewEvidence
  -> FinalInstructionVerifier decides STOP
```

关键边界：

```text
SearchPriorResult is ranking evidence, not a motion command.
PriorMapMemory is runtime memory, not planner state.
PriorMapData is map input, not observed truth.
```

## 4. 建议模块结构

新增目录建议如下：

```text
prior_map/
  contracts.py
  loaders.py
  memory.py
  alignment.py
  query.py
  snapshot_adapter.py
  prompt_context.py
  visualizer.py
```

职责划分：

| 模块 | 职责 | 不能做什么 |
| --- | --- | --- |
| `contracts.py` | 定义 `PriorMapData`、`PriorRoom`、`PriorObject`、`SearchPriorResult` | 不 import Habitat、ROS、OpenCV |
| `loaders.py` | 读取 JSON、OSM、FloorPlan-VLN、VLM reconstruction 输出 | 不做 runtime 观测融合 |
| `alignment.py` | prior frame 和 runtime frame 的坐标变换 | 不猜测语义房间或目标 |
| `memory.py` | 维护 visit、verified、rejected、confidence 和 online update | 不直接发布导航目标 |
| `query.py` | 根据 `InstructionPlan` 和 runtime state 生成搜索先验 | 不触发 STOP |
| `snapshot_adapter.py` | 从 mapper 或 `SemanticMapSnapshot` 构造统一 runtime context | 不修改 mapper/SysNav 状态 |
| `prompt_context.py` | 渲染 LLM/VLM 可读 prior map context | 不调用 live planner |
| `visualizer.py` | 生成 SoM/top-down debug 视图 | 不参与运行时决策 |

## 5. 核心数据结构

### 5.1 PriorMapData

```python
@dataclass(frozen=True)
class PriorMapData:
    scene_id: str
    rooms: tuple[PriorRoom, ...] = ()
    objects: tuple[PriorObject, ...] = ()
    topology_edges: tuple[PriorTopologyEdge, ...] = ()
    source_format: str = "unknown"
    frame_id: str = "prior_map"
    world_min: tuple[float, float] | None = None
    world_max: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

`PriorMapData` 是不可变输入视图。运行中新增或更新的信息进入
`PriorMapMemory`，不要直接改原始 `PriorMapData`。

### 5.2 PriorRoom

```python
@dataclass(frozen=True)
class PriorRoom:
    uid: str
    label: str = "unknown"
    boundary_xy: tuple[tuple[float, float], ...] = ()
    centroid_xy: tuple[float, float] | None = None
    neighbors: tuple[str, ...] = ()
    level: int = 0
    confidence: float = 0.5
    source: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

`label` 是先验语义，不是 runtime 事实。实物中 SysNav room node 默认没有自然语言
房间名，因此 prior room label 只能作为搜索提示。

### 5.3 PriorObject

```python
@dataclass(frozen=True)
class PriorObject:
    uid: str
    label: str
    position_xyz: tuple[float, float, float] | None = None
    parent_room_uid: str | None = None
    exact: bool = False
    confidence: float = 0.5
    source: str = ""
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
```

`exact=False` 表示对象只是房间或区域中的语义先验，例如“kitchen 可能有 microwave”。
这类对象不能被当作已观测目标，只能影响搜索优先级。

### 5.4 PriorMapMemory

```python
@dataclass
class PriorMapMemory:
    base_map: PriorMapData
    alignment: PriorMapAlignment
    room_states: dict[str, PriorRoomRuntimeState]
    object_states: dict[str, PriorObjectRuntimeState]
    observations: list[PriorObservationRecord]

    def update_from_mapper(self, mapper: Any, step: int) -> None: ...
    def update_from_snapshot(self, snapshot: SemanticMapSnapshot) -> None: ...
    def mark_room_visited(self, room_uid: str, step: int) -> None: ...
    def mark_object_verified(self, prior_uid: str, runtime_uid: str, step: int) -> None: ...
    def mark_prior_rejected(self, prior_uid: str, reason: str, step: int) -> None: ...
    def current_map(self) -> PriorMapData: ...
```

runtime state 需要单调记录：

```text
visited rooms
verified prior objects
runtime observed objects matched to prior objects
rejected prior hypotheses
frontier / room exhaustion evidence
alignment confidence
```

### 5.5 SearchPriorResult

```python
@dataclass(frozen=True)
class SearchPriorResult:
    room_rankings: tuple[RoomPrior, ...] = ()
    object_rankings: tuple[ObjectPrior, ...] = ()
    frontier_biases: tuple[FrontierPrior, ...] = ()
    support_regions: tuple[SupportRegionPrior, ...] = ()
    prompt_context: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

这是先验地图对 STRIVE 策略层的唯一输出。任何 policy 只能读取它做排序或解释，
不能把其中某个 prior point 直接当成最终 goal。

## 6. Query 语义

`PriorMapQueryService` 输入：

```text
InstructionPlan
Current runtime pose
Observed objects / rooms / frontiers
PriorMapMemory
Execution state
Rejected / verified ledger
```

输出 `SearchPriorResult`：

```text
room_rankings:
  room uid, label, score, reason, visit state, reachable_hint

object_rankings:
  prior object uid, label, parent room, score, exactness, matched runtime uid

frontier_biases:
  runtime frontier uid, prior room uid, score delta, reason

support_regions:
  support object / room / region hints for contextual search
```

推荐的排序形式：

```text
score =
  w_task   * concept_relevance
+ w_room   * room_relevance
+ w_topo   * topology_reachability_hint
+ w_visit  * unvisited_or_underexplored_bonus
+ w_live   * live_evidence_agreement
- w_reject * rejection_penalty
```

这些分数是 soft ranking，不是 hard filter。除非未来某个 benchmark 明确定义先验地图
是 ground truth，否则先验地图不能 hard reject 当前 live observation。

## 7. 与现有 STRIVE 的接入点

### 7.1 instruction_adapter

`InstructionPlan.search_priors` 已经有：

```text
room_hints
support_objects
affordances
```

先验地图 query 应读取这些字段，但不能修改 parser 语义。例如 `cup on table in kitchen`
可以让 query 提高 kitchen、table 附近 frontier 的优先级，但最终仍由
`ConstraintEvaluator` 和 final verifier 判断 cup/table/kitchen 关系是否满足。

### 7.2 planning

建议新增轻量 policy adapter：

```python
class PriorMapPolicyAdapter:
    def rank_rooms(self, rooms, prior_result: SearchPriorResult) -> list
    def rank_frontiers(self, frontiers, prior_result: SearchPriorResult) -> list
    def annotate_target_candidates(self, candidates, prior_result: SearchPriorResult) -> list
```

它只能改变排序和 debug annotation，不负责目标确认。

### 7.3 real_robot

实物模式中先验地图读取 `SemanticMapSnapshot`：

```python
memory.update_from_snapshot(snapshot)
prior_result = query_service.query(plan, snapshot, memory)
intent = policy.decide(snapshot, instruction, prior_context=prior_result.prompt_context)
```

如果 prior map 和 live map 冲突，live evidence 优先。prior map 降级为 hypothesis，
并写入 diagnostics。

## 8. 对齐与坐标

先验地图必须显式声明坐标系：

```text
prior_map frame
Habitat world frame
ROS map frame
camera/lidar frame
```

`PriorMapAlignment` 负责转换：

```python
class PriorMapAlignment:
    def prior_to_runtime(self, point_xy: tuple[float, float]) -> tuple[float, float, float]: ...
    def runtime_to_prior(self, point_xyz: tuple[float, float, float]) -> tuple[float, float]: ...
    def confidence(self) -> float: ...
```

第一版可以支持 `identity` alignment。OSM、FloorPlan 图片或 VLM reconstruction
需要额外提供 scale、rotation、translation 和 diagnostics。

## 9. 日志与可观测性

每个 episode/run 建议保存：

```text
prior_map/base_map.json
prior_map/alignment.json
prior_map/runtime_state_*.json
prior_map/query_*.json
prior_map/search_prior_result_*.json
prior_map/som_global_*.png
prior_map/som_room_<room_uid>_*.png
```

`query_*.json` 至少包含：

```text
instruction plan hash
runtime pose
top room/object/frontier rankings
score components
live/prior conflict diagnostics
rejected or exhausted prior hypotheses
```

## 10. 测试策略

先验地图的测试应从纯逻辑开始：

```text
tests/test_prior_map_contracts.py
  dataclass validation, JSON roundtrip, immutable base map

tests/test_prior_map_loaders.py
  floorplan JSON, FloorPlan-VLN, OSM XML minimal fixtures

tests/test_prior_map_alignment.py
  identity / affine transform / inverse consistency

tests/test_prior_map_memory.py
  room visit, object verification, rejection, live update

tests/test_prior_map_query.py
  room ranking, object ranking, rejected prior penalty, live evidence agreement

tests/test_prior_map_policy_adapter.py
  ranking-only behavior; no direct goal publication
```

集成测试再覆盖：

```text
sim smoke:
  enable prior map, verify planner still routes through existing frontier/path logic

real snapshot replay:
  feed fake SemanticMapSnapshot, verify NavigationIntent comes from policy,
  not from PriorMapData directly
```

## 11. 分阶段实施

推荐阶段：

1. `PriorMapData`、loader、alignment 和 roundtrip 测试。
2. `PriorMapMemory`，接入 mapper/snapshot 的只读 update。
3. `PriorMapQueryService`，生成 `SearchPriorResult`。
4. 仿真接入 room/frontier/object ranking。
5. 实物接入 `SemanticMapSnapshotPolicyContext` / 现有 planning policy 的 prior ranking context。
6. SoM/debug artifact。
7. 仿真和 bag replay smoke。

阶段 1-3 完成前，不应把先验地图接入主导航循环。阶段 4-5 接入时必须保留开关，
默认关闭，方便和无先验地图模式做 A/B 回归。
