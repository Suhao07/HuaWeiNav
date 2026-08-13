# STRIVE 先验地图模式技术文档

本文档定义 STRIVE 的先验地图模式。目标是在 Habitat/HM3D 仿真和真实机器人模式中
复用同一套语义先验地图 contract，让预先构建的房间、拓扑和区域信息作为搜索先验
进入 STRIVE，而不是把 Habitat 的 NavMesh 真值带入真实机器人流程。

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

### 1.1 HM3D 与 FloorPlan-VLN 数据边界

FloorPlan-VLN 的原始房间布局来自 Matterport3D `*.house` 房屋分割数据和
Matterport connectivity graph；HM3D v0.2 发布的是 `basis.glb`、`semantic.glb`、
`semantic.txt`、`basis.navmesh`，不包含同构的 `.house` 文件。因此 STRIVE 不应
伪造 HM3D `.house`，而是从 HM3D NavMesh 与语义场景生成同构的 room-only
`floorplan.json`。

```text
HM3D semantic scene + NavMesh
  -> prior_map.hm3d_layout.HM3DLayoutBuilder
  -> levels / regions / boundaries / connectivity
  -> FloorPlan-compatible floorplan.json
  -> PriorMapLoader(source_format="floorplan_vln_json")
```

该格式的坐标契约是：

```text
metric source frame: habitat_world, world axes=(x, y, z)
floorplan metric plane: (x, -z)
floorplan.json frame_id: floorplan_metric
rendered PNG: image pixels only, never a navigation pose
```

layout-only prior 的 authority 分层如下：

```text
构建阶段 NavMesh         -> 仅用于恢复房间几何，不作为输出先验
semantic prior JSON       -> 房间边界、标签与拓扑
semantic prior BEV        -> LVLM 和人工使用的地图上下文
online ObjectNode         -> 运行时物体身份与实测证据
real-time local planner   -> 当前可达性、避障和 waypoint 执行
final verifier            -> 任务满足与停止
```

命令：

```bash
COGNAV_ROOT=/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav \
IMAGE_TAG=strive-hm3d:local \
bash docker/run_hm3d_floorplan_layout.sh \
  /workspace/data/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35 \
  --scene_id wcojb4TFT35 \
  --output logs/prior_maps/wcojb4TFT35_floorplan.json \
  --quality_output logs/prior_maps/wcojb4TFT35_floorplan_quality.json \
  --topdown_resolution 0.25
```

该脚本的输入场景路径是容器内路径，宿主数据目录通过
`STRIVE_DATA_ROOT -> HM3D_DATA_ROOT -> configs/strive_weights.yaml:data_root`
的顺序选择，最后才回退到 `STRIVE_ROOT/data`。它只要求 `habitat_sim` 和 HM3D
basis/semantic/NavMesh 文件。
不要使用 `docker/run_hm3d_baseline.sh` 生成布局。该 benchmark wrapper 还会检查
SAM、GroundingDINO 和 episode 标注；HM3D 文件存在并不代表宿主 Python 已安装
Habitat-Sim，实际构建应在 `IMAGE_TAG` 指定的容器中执行。

该命令写出完整的语义先验地图 bundle：

```text
<floorplan.json 同目录>/floorplan.json
<floorplan.json 同目录>/prior_map_bev.png
<floorplan.json 同目录>/prior_map_bev.svg
<floorplan.json 同目录>/prior_map_bev_markers.json
<floorplan.json 同目录>/prior_map_manifest.json
```

其中 `prior_map_bev.png` 是与 `floorplan.json` 同源生成的语义平面图，供 LVLM
读取地图上下文和人工检查；`prior_map_bev.svg` 保留可缩放版本；markers 文件记录
图像坐标与地图元素的对应关系。`prior_map_manifest.json` 声明坐标系、来源和完整
artifact 列表。

正式 bundle 只包含上述语义文件。Habitat `PathFinder` 只在构建阶段用于恢复房间的
几何范围，不能作为仿真先验输入，也不能进入 LVLM prompt 或实物导航接口。

STRIVE 生成的布局被 `PriorMapLoader` 读取时会识别
`frame_id=floorplan_metric` 与 `floorplan_axes=["x", "-z"]`，将第二个平面轴反射回
canonical Habitat `(x,z)`；未带该显式标记的第三方 FloorPlan 文件不会被静默改写。

当 Habitat-Sim 暴露的 semantic region AABB 为无效值时，builder 会从 semantic mesh
中的实例几何按 parent region 聚合保守包围盒，并在质量报告的
`region_geometry_sources` 中记录来源；若仍没有任何有效 room，CLI 会返回失败，不会
写出可被误用的空布局。

NavMesh 采样可能在同一语义区域返回多个高度层。BEV 先验在每个 semantic region 内
先将 `(x, z, y)` 采样投影为二维 `(x, z)` 网格，再提取四连通组件；因此不同高度的
重复采样不会生成重复房间。输出中的 `component_N` 表示几何连通组件，不是 HM3D
提供的自然语言房间名称。

## 本地检测权重

完整 ObjectNav 的 Docker 入口读取 `configs/strive_weights.yaml`。该文件只保存本机
checkpoint 路径，不把权重提交到仓库；启动时会以只读方式挂载为：

```text
/weights/sam_vit_h_4b8939.pth
/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

环境变量 `SAM_CHECKPOINT` 和 `GROUNDING_DINO_CHECKPOINT` 的优先级高于 YAML，适合
迁移到另一台机器。GroundingDINO 的配置固定使用镜像内的 Swin-L 配置；不能用
GroundingDINO Swin-T 权重静默替代。

批量构建仍可在已准备好的 Habitat Docker 容器中直接调用 CLI：

```bash
bash docker/run_hm3d_baseline.sh bash -lc '
  cd /workspace/STRIVE
  python scripts/build_hm3d_floorplan_layouts.py \
    /workspace/data/scene_datasets/hm3d_v0.2/val \
    --output_root logs/prior_maps/hm3d_val_floorplans \
    --topdown_resolution 0.25
'
```

批量 manifest 记录每个场景的源目录、布局路径、质量元数据以及语义 BEV 路径，便于
后续做启用/关闭先验地图的成对实验。

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

仿真中可以直接消费的先验地图是 canonical `PriorMapData` JSON，而不是 Habitat
dataset 原始 episode 文件。正式 smoke 优先使用 HM3D scene semantic annotation
生成场景级语义库存先验，因为它来自 scene asset，不依赖当前 ObjectNav episode 的
goal object 坐标，不会把评测目标位置泄漏给 policy。

HM3D `*.semantic.txt` 路径只提供 instance label 和 region id，不提供可校准的
object position 或 room boundary。因此第一版生成的 alignment 默认为
`unavailable`，先验地图会降级为 prompt/context/ranking evidence，不参与几何
frontier 生成，也不能直接给出 motion goal：

```text
HM3D scene semantic txt
  -> scripts/build_hm3d_semantic_prior_map.py
  -> prior_map.json            # source_format=hm3d_semantic_txt
  -> alignment.json            # unavailable, prompt_context_only
  -> objnav_benchmark_with_process_obs.py --enable_prior_map
```

示例：

```bash
mkdir -p logs/prior_maps

python3 scripts/build_hm3d_semantic_prior_map.py \
  /home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35/wcojb4TFT35.semantic.txt \
  --scene_id wcojb4TFT35 \
  --output logs/prior_maps/wcojb4TFT35_hm3d_semantic_prior_map.json \
  --alignment_output logs/prior_maps/wcojb4TFT35_hm3d_semantic_alignment.json \
  --alignment_mode unavailable
```

加载验证：

```bash
LLM_OFFLINE=1 STRIVE_LLM_FALLBACK=1 bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_ovon \
  --benchmark_split val_seen_complex_balanced_2k \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --episode_rank 0 \
  --save_dir hm3d_semantic_prior_on \
  --vlm cognav \
  --max_steps 1 \
  --enable_prior_map \
  --prior_map_path /workspace/STRIVE/logs/prior_maps/wcojb4TFT35_hm3d_semantic_prior_map.json \
  --prior_map_source canonical_json \
  --prior_map_alignment /workspace/STRIVE/logs/prior_maps/wcojb4TFT35_hm3d_semantic_alignment.json
```

A/B 对照关闭 prior map：

```bash
LLM_OFFLINE=1 STRIVE_LLM_FALLBACK=1 bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_ovon \
  --benchmark_split val_seen_complex_balanced_2k \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --episode_rank 0 \
  --save_dir hm3d_semantic_prior_off \
  --vlm cognav \
  --max_steps 1
```

Habitat/ObjectNav dataset 也可以生成 canonical prior map，但它读取 episode
`goals_by_category`。这条路径适合调试 loader、artifact 和 A/B 回归，不适合作为
正式“真实先验地图”评测，因为 goal object 坐标可能泄漏当前任务答案：

```text
Habitat ObjectNav val.json.gz
  -> scripts/build_habitat_prior_map.py
  -> prior_map.json            # source_format=habitat_objectnav_json
  -> alignment.json optional   # default unavailable, unless coordinates are calibrated
  -> objnav_benchmark_with_process_obs.py --enable_prior_map
```

示例，宿主侧生成 ObjectNav 调试 prior，Docker 内按 `/workspace/STRIVE/...` 路径消费：
脚本可以直接读取 `content/<scene>.json.gz`，也可以读取 split 根文件并自动扫描
同目录 `content/` 下的 scene episode 文件。

```bash
mkdir -p logs/prior_maps

python scripts/build_habitat_prior_map.py \
  /home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data/datasets/objectnav/hm3d_ovon/v1/val_seen_complex_balanced_2k/val_seen_complex_balanced_2k.json.gz \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --episode_rank 0 \
  --output logs/prior_maps/wcojb4TFT35_tv_prior_map.json \
  --alignment_output logs/prior_maps/wcojb4TFT35_tv_alignment.json \
  --alignment_mode unavailable
```

加载验证：

```bash
LLM_OFFLINE=1 STRIVE_LLM_FALLBACK=1 bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_ovon \
  --benchmark_split val_seen_complex_balanced_2k \
  --eval_episodes 1 \
  --start_episode 0 \
  --save_dir hm3d_prior_smoke \
  --vlm cognav \
  --max_steps 1 \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --episode_rank 0 \
  --enable_prior_map \
  --prior_map_path /workspace/STRIVE/logs/prior_maps/wcojb4TFT35_tv_prior_map.json \
  --prior_map_source canonical_json \
  --prior_map_alignment /workspace/STRIVE/logs/prior_maps/wcojb4TFT35_tv_alignment.json
```

A/B 对照关闭 prior map：

```bash
LLM_OFFLINE=1 STRIVE_LLM_FALLBACK=1 bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_ovon \
  --benchmark_split val_seen_complex_balanced_2k \
  --eval_episodes 1 \
  --start_episode 0 \
  --save_dir hm3d_prior_ab_off \
  --vlm cognav \
  --max_steps 1 \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --episode_rank 0
```

验证重点：

```text
prior map on:
  logs/<save_dir>/episode-*/prior_map/base_map.json exists
  query_*.json / search_prior_result_*.json / runtime_state_*.json exists
  metrics/debug records prior_map_enabled=true

prior map off:
  no prior_map runtime artifacts are required
  planner still uses current mainline behavior

both:
  STOP authority remains final verifier / existing success logic
  SearchPriorResult never contains motion_goal or navigation_intent
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

### 3.3 实物部署接入点

实物模式通过 runtime 参数接入先验地图，入口保持在高层 policy context，不进入
ROS adapter 或底层控制器：

```text
STRIVE_PRIOR_MAP_PATH=/abs/path/to/prior_map.json
  -> docker_en.sh 只读挂载该文件所在目录
  -> strive_instruction_runtime prior_map_path 参数
  -> SemanticMapSnapshotPolicyContext / prior ranking context
```

示例：

```bash
START_STRIVE_RUNTIME=1 \
STRIVE_INSTRUCTION="find a book near the table" \
STRIVE_DATASET_TARGET=book \
STRIVE_POLICY_MODE=semantic_snapshot \
STRIVE_INSTRUCTION_PLAN_BACKEND=rules \
STRIVE_PRIOR_MAP_PATH=/home/orin26/maps/lab_prior_map.json \
STRIVE_DRY_RUN=true \
SUDO_STDIN_PASSWORD=1 ./docker_en.sh start
```

边界：

```text
prior_map_path 只能提供搜索排序、房间/区域提示、debug context。
prior map 不能直接生成 /way_point。
prior map 不能绕过 SemanticMapSnapshot 的 live observation。
prior map 不能触发 STOP；STOP 仍只来自 final verifier accept。
```

若 prior map 和 live SysNav snapshot 冲突，live snapshot 优先；prior map 降级为
hypothesis，并应写入 `RuntimeDecision.metadata` 便于复盘。

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
prior_map/som_global.png
prior_map/som_global_markers.json
prior_map/som_room_<room_uid>.png
prior_map/som_room_<room_uid>_markers.json
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

tests/test_habitat_objectnav_prior_map.py
  Habitat ObjectNav dataset -> canonical prior map generator,
  loader roundtrip, alignment file, CLI smoke

tests/test_hm3d_semantic_prior_map.py
  HM3D scene semantic txt -> canonical prior map generator,
  structural label filtering, loader roundtrip, unavailable alignment, CLI smoke
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

## 12. BEV 与房间语义推理

先验地图的多模态部分采用 SysNav 的分层边界，但不复制 SysNav 的固定
`room_types` 枚举：

```text
SysNav RoomNode geometry/mask + current RGB evidence
  -> RoomEvidence
  -> RoomSemanticClassifier (open-ended LVLM label)
  -> RoomSemanticResult + evidence-version cache

static room prior + live pose/trajectory/frontiers/live objects
  -> dynamic_prior_map_bev_<step>.png
  -> PriorMapMultimodalContext
  -> PriorMapHighLevelSelector
  -> existing room/frontier candidate UID
```

房间语义分类需要可读的 RGB crop 或 frame。SysNav 的 `RoomNode.room_mask`
只是空间分割证据，ROS URI 不会被错误地当成 RGB 输入；没有 RGB 时结果为
`unknown`，不产生 LVLM 请求。缓存键包含场景、房间 UID、图像/几何证据哈希、
模型和 prompt 版本，观测版本变化后才重新分类。

动态 BEV 包含：

- 静态房间边界与 room-room topology；
- 当前机器人位置、轨迹和在线 frontier；
- live object 与 prior object 的对齐标记；
- 当前房间、候选房间和已选房间 UID。

BEV 只送入高层 room/frontier selector，用于探索顺序；它不送入 concept
matcher、relation verifier 或 final verifier，也不能直接发布 waypoint 或
授予 STOP。模型返回的 UID 必须存在于 planner 提供的候选集合，否则退回已有
几何策略的第一候选。

高层 selector 的输入候选统一为 `HighLevelCandidate`，候选类型至少包括
`room` 和 `frontier`。候选 UID、运行时可达性和实际位姿由 mapper/SysNav
snapshot 提供；LVLM 只能在这组 UID 中排序。selector 的原始响应、解析结果、
BEV SHA-256、候选 UID、模型、prompt 版本和请求耗时分别写入
`high_level_selection_<step>.json` 与全局 LVLM trace，便于复核模型是否真正消费了
BEV 和候选 JSON。

房间语义产物为 `room_semantics_<step>.json`。每条结果包含 RGB/mask 证据哈希、
开放标签、描述、不确定性、原始响应文本、解析结果和请求元数据。RGB 或 mask 内容
变化会使证据版本失效；没有可读 RGB 时返回 `unknown`，不发送请求。

当先验图与运行时坐标尚未完成标定时，BEV 只输出静态房间/拓扑语义和文本状态，
不绘制机器人、轨迹或未经变换的 frontier 位置。这样可以保留 LVLM 的语义上下文，
同时避免把运行时坐标误当作先验图坐标。

默认保持现有行为。仿真可增加：

```bash
--enable_prior_map --prior_map_path <prior_map.json> \
--enable_prior_map_vlm --prior_map_vlm_interval 10
```

实物 ROS2 节点对应参数为 `enable_prior_map_vlm`、
`prior_map_vlm_interval`、`room_semantic_interval` 与
`enable_room_semantics`。实物要真正提供 SysNav 风格的 RGB+mask 房间证据，
还需启用 `persist_observation_images:=true`；否则 ROS URI 不能直接送入 LVLM，
房间语义保守保持 `unknown`。运行产物位于
`<run_directory>/prior_map/dynamic_prior_map_bev_*.png`，同时在
`prompt_context_*.json` 与 runtime diagnostics 中保存图像哈希、证据版本、
候选 UID 和原始结构化响应。
