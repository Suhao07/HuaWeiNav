# STRIVE 重构后代码框架

本文记录 2026-06-10 开始的 Phase0-6 重构边界。目标不是一次性重写
STRIVE，而是在保持 HM3D benchmark 和 instruction mode 行为兼容的前提下，
把可复用的数学工具、策略接口和运行时状态逐步拆出来。

## Phase0：安全护栏

新增本地检查脚本：

```bash
bash scripts/check_refactor.sh
```

检查内容：

```text
核心入口 py_compile
cv_utils / llm_utils / mapping_utils py_compile
instruction_adapter py_compile
navigation_core / planning py_compile
artifact_utils / mapping / navigation py_compile
lightweight pytest regression tests
```

这个脚本是后续重构的最低门槛。任何继续拆分 agent / mapper 的提交，都应先
通过它，再考虑 Docker smoke 或完整 benchmark。

当前纳入脚本的轻量回归测试：

```text
tests/test_frontier_extractor.py
  frontier 自适应局部半径

tests/test_panoramic_detection.py
  panorama circular triplet index
  stitched center-panel bbox filter

tests/test_artifact_paths.py
  episode/subdir/detection step artifact 路径约定
```

这些测试只覆盖纯逻辑和路径约定，不依赖 Habitat、GPU、Open3D 或线上 LLM。

## Phase1：核心数学工具层

新增目录：

```text
navigation_core/
  view_geometry.py
```

职责：

```text
interpolate_polyline()
truncate_at_stop_radius()
rotation_towards()
score_projection()
habitat_path_to_strive_points()
```

输入：

```text
Habitat path points
mapper initial position
object point cloud positions
2D projection points
success distance / stop criterion
```

输出：

```text
ProjectionQuality:
  score
  visible_ratio
  area_ratio
  center_score
  border_score
  distance_score
  predicted_quality
```

设计原则：

```text
只处理几何和视角质量；
不判断 book/shelf/chair 等语义；
不调用 LLM/VLM；
不读写日志或运行时状态。
```

核心评分仍是通用几何证据质量：

```text
view_score =
  (0.35 * visible_ratio
 + 0.25 * center_score
 + 0.15 * border_score
 + 0.15 * area_score
 + 0.10 * distance_score)
 * (0.75 + 0.25 * aspect_stability)
```

其中：

```text
visible_ratio = visible_projected_points / all_object_points
center_score  = 1 - normalized distance from image center
border_score  = normalized distance from image border
area_score    = sqrt(projected_area / original_area) clipped
distance_score= closeness to preferred stop distance
```

## Phase2：Planning 策略薄层

新增目录：

```text
planning/
  exploration_policy.py
  mode_policy.py
  object_search_policy.py
  room_policy.py
  target_selection_policy.py
  viewpoint_policy.py
```

### viewpoint_policy.py

`build_check_again_viewpoints()` 负责从一段 Habitat shortest path 生成更好视角候选：

```text
object point cloud + camera intrinsic + path samples
  -> projection from each candidate viewpoint
  -> class-agnostic quality score
  -> sorted/consumable viewpoint proposals
```

`HM3D_Objnav_Agent.whether_to_check_again()` 现在只负责：

```text
调用 Habitat pathfinder；
调用 viewpoint policy 生成 proposals；
把 proposals 交给 ViewControlState；
设置 check_again_position。
```

### object_search_policy.py

`InstructionObjectSearchPolicy` 负责 instruction mode 的对象选择：

```text
mapper objects + InstructionPlan
  -> RuntimeConceptMatcher.match_many()
  -> terminal candidate 或 anchor reference
  -> ObjectSearchResult
```

角色边界：

```text
terminal target:
  可以进入 final verifier，最终可能触发 stop。

anchor reference:
  只作为局部搜索参考物；
  不允许触发任务成功；
  搜索失败只屏蔽该 anchor 实例。
```

这让 `mapper_with_process_obs.py` 不再直接维护 terminal/anchor 的批量
grounding 细节，只保存运行日志和 benchmark 兼容入口。

### mode_policy.py

集中处理 execution mode：

```text
execution_mode(plan)
is_anchor_first_relation_search(plan)
is_ordered_execution(plan)
```

后续新增 mode 时，应优先扩展这里，而不是在 agent/mapper 中散落字符串比较。

### room_policy.py

`select_nearest_frontier_room()` 是关闭 LLM room selection 时的确定性
relocation policy：

```text
frontier nodes + current pose
  -> Habitat geodesic distance
  -> nearest frontier room
```

它不读取目标类别，也不解释自然语言。这样 `--no_gpt_relocate` 的行为仍然
保留，但不再需要独立维护一整套 `*_no_gpt` agent 主流程。

### exploration_policy.py

`exploration_policy.py` 承接 mapper 中的 room/frontier node 选择：

```text
mapper node graph + room nodes + Habitat geodesic distance
  -> closest frontier viewpoint
  -> relocation frontier viewpoint
  -> exhausted room state update
```

职责边界：

```text
可以读取 node / room_node / frontier state；
可以更新 room_node.state 和 node.has_frontier；
不做 perception；
不做 room segmentation；
不做 target/final verifier 判断。
```

核心原则是把“选下一个探索 node”的策略从 `mapper_with_process_obs.py`
剥离出来，mapper 只保留兼容 wrapper：

```text
mapper.explore_in_room(room)
  -> planning.exploration_policy.explore_in_room(mapper, room)
```

### target_selection_policy.py

`target_selection_policy.py` 统一普通 benchmark 与 instruction mode 的候选选择入口：

```text
mapper objects + optional InstructionPlan
  -> TargetSelectionResult(found, obj, answer, debug)
```

行为边界：

```text
benchmark mode:
  只做 target alias / detector tag 匹配；
  不调用 LLM/VLM；
  不启用 relation/final verifier。

instruction mode:
  注册 mapper objects 到 instruction spatial graph；
  更新 sequence active target；
  委托 InstructionObjectSearchPolicy 做 terminal/anchor 选择。
```

mapper 仍负责把 `TargetSelectionResult` 写入原有 `no_gpt_obj/answer_*.txt`，
planning 层不做文件 IO。

## Phase3：Mapping 服务层

新增目录：

```text
mapping/
  frontier_clusterer.py
  frontier_extractor.py
  map_serializer.py
  node_candidate_builder.py
  room_segmenter.py
  topology_graph.py
```

### frontier_extractor.py

`frontier_extractor.py` 先承接 `get_nodes()` 中无副作用或弱副作用的 frontier 子步骤：

```text
frontier clusters
  -> debug concatenated frontier points
  -> adaptive local intersection radius
  -> redundant visible center pruning
  -> traversable/reachable candidate append
```

该模块当前不是完整 frontier pipeline。`get_nodes()` 仍然保留 DBSCAN、Open3D
point-cloud slicing、debug PLY 写入和 node 创建，因为这些步骤同时依赖 mapper
运行时缓存、GPU point cloud、日志路径和 topology graph。继续拆分时，应优先把
这些函数扩展为：

```text
FrontierExtractionResult:
  frontier_clusters
  frontier_centers
  candidate_centers
  candidate_map_indices
  debug_artifacts
```

核心注释：

```text
frontier_extractor 不读取目标类别；
不调用 LLM/VLM；
只处理 frontier 几何候选，不决定任务成功。
```

### frontier_clusterer.py

`frontier_clusterer.py` 承接 `get_nodes()` 中 DBSCAN cluster 到 candidate center
的转换：

```text
pcd_removed + DBSCAN labels
  -> match local clusters with global frontier clusters
  -> merge visible frontier fragments
  -> prune redundant visible centers
  -> fallback centers from large uncovered navigable clusters
```

该模块仍使用 mapper 的 `merge_frontier_with_visibility_1()` 和 `is_visible()`，
因为这些几何函数依赖 mapper runtime state。新的边界是：

```text
frontier_clusterer 可以调用 mapper geometry helpers；
不能 add_node；
不能 update_edges；
不能 segment_room；
不能读取目标类别或 instruction plan。
```

核心中文注释集中在两个地方：

```text
多视角可见 frontier 合并：避免一个门口生成多个重复 viewpoint；
无 frontier 大 cluster fallback：补充可能漏掉的通路节点，但不携带 frontier index。
```

### node_candidate_builder.py

`node_candidate_builder.py` 承接 traversable candidate 到 topology node 的转换：

```text
valid_centers
  -> candidate bearing
  -> 80-degree local point-cloud sector
  -> mapper.add_node()
  -> mapper.add_edge(current_node, new_node)
```

它是几何候选与 topology mutation 的边界。candidate 是否存在、是否可达，在前置
模块中决定；graph 去重和 frontier 合并仍由 `mapping.topology_graph.add_node()`
负责。

核心约束：

```text
node_candidate_builder 不选择探索策略；
不判断目标；
不做 room segmentation；
只把已经可达的 candidate materialize 成 node。
```

### room_segmenter.py

`room_segmenter.segment_room()` 承接原 `mapper.segment_room()` 的房间分割算法：

```text
scene point cloud + navigable point cloud + navigation nodes
  -> 2D obstacle histogram
  -> wall/free-space masks
  -> connected room regions
  -> Room_node list + node.room_idx assignment
  -> frontier map labels
```

该模块仍通过 mapper runtime 写回 `room_nodes / node.room_idx / grid_map`，这是
为了保持现有 benchmark 行为不变；但图像形态学、region marker 和 fallback
room 逻辑已经离开 mapper 主文件。后续可以把输入输出进一步收敛为纯
`RoomSegmentationResult`。

核心约束：

```text
room segmentation 不使用目标类别；
不调用 LLM/VLM；
不触发导航动作；
早期地图无法稳定切房间时，fallback_single_room 保证 relocation 不崩溃。
```

### topology_graph.py

`topology_graph.py` 承接 mapper 的 viewpoint graph / object-node graph 操作：

```text
add_node / visit_node / add_edge / remove_edge
update_obj
find_closest_node / find_closest_unexplored_node
find_the_closest_path / check_connected
```

当前仍以 mapper runtime 作为状态容器，原因是 node、object、frontier map、
Habitat geodesic distance 还共享同一运行时上下文。新的边界是：

```text
mapper:
  持有 state 和兼容方法名

mapping.topology_graph:
  执行图结构更新和 Dijkstra 查询
```

### map_serializer.py

`map_serializer.py` 承接 mapper 的 prompt/debug JSON 构造：

```text
to_json()
to_json_wo_some_class()
to_json_save_node_info()
```

它只负责 presentation format，不做目标判断、不修改导航策略。唯一保留的状态修正是：

```text
已探索 node 在序列化前清空 frontier 标记；
避免 room prompt 重复选择已访问 viewpoint。
```

## Phase4：运行时边界

当前运行时边界如下：

```text
objnav_benchmark_with_process_obs.py
  -> episode/config/metrics orchestration

mapper_with_process_obs.py
  -> mapping runtime state
  -> compatibility wrappers for mapping/planning services
  -> frontier extraction and visibility-edge update that have not yet been extracted
  -> instruction policy invocation and debug artifact writing

objnav_agent_with_process_obs.py
  -> Habitat action loop
  -> final verifier orchestration
  -> compatibility wrapper for view verification controller

instruction_adapter/
  -> prompt-first instruction plan
  -> concept grounding
  -> relation / final verifier
  -> ledgers and dynamic semantic edges

navigation_core/
  -> pure math and geometry helpers

mapping/
  -> room segmentation / topology graph / map serialization services
  -> frontier cluster analysis / node candidate materialization

navigation/
  -> action controller / observation pipeline / view verification controller
  -> panoramic detection / object merge / bbox refinement / path progress

artifact_utils/
  -> shared episode/step path construction
  -> shared point-cloud/line-set writer helpers

planning/
  -> target/object/search/room/exploration/viewpoint policy wrappers
```

### navigation/view_verification_controller.py

`select_check_again_viewpoint()` 承接 agent 中的 check-again 视角选择：

```text
found candidate + Habitat shortest path
  -> build_check_again_viewpoints()
  -> ViewControlState proposals
  -> check_again_position / need_check_again
```

该 controller 不调用 final verifier，不判断目标类别，只负责选择下一帧更好视角。
这让 `objnav_agent_with_process_obs.py.whether_to_check_again()` 退化为兼容 wrapper。

### navigation/observation_pipeline.py

`observation_pipeline.py` 承接 agent 中与全景观测和观测点云相关的可复用工具：

```text
reset panoramic buffers
  -> collect 12-view RGB/depth/pose/point-cloud sweep
  -> temporary panoramic point clouds
  -> merged current observation point cloud
  -> debug obs_*.ply artifact
```

它不更新 mapper graph、不选择 action、不判断目标。当前 `rotate_segmentation()`
仍留在 agent 中，因为它同时涉及 detector、SAM、object association、LLM bbox
refine 和大量 debug IO。后续继续拆时，应让该模块输出：

```text
ObservationBatch:
  rgb frames
  depth frames
  mapper-local poses
  rotations
  temporary point clouds
  detected object entities
```

核心原则：

```text
observation_pipeline 可以移动传感器和整理观测缓存；
不能选择 frontier/node；
不能调用 final verifier；
不能决定 stop。
```

### navigation/planner_loop.py

`planner_loop.py` 承接每个 planning cycle 重复出现的前半段：

```text
rotate panoramic observation
  -> merge temporary point clouds
  -> save observation debug PLY
  -> mapper.get_nodes()
  -> mapper.update_obj()
```

它返回 `PlanningCycleResult(step, episode_over)`。目标发现、room exploration、
relocation 和 stop 判断仍保留在 agent 主流程中。

这个边界的目的不是把状态机藏起来，而是把高重复、低策略含义的
observation->mapping 前缀先稳定下来。后续拆 `navigation/planner_loop.py`
时可以继续往下移动：

```text
target candidate check
current-room exploration
relocation fallback
path materialization
```

但每一步都应保持 benchmark/instruction 两种模式的行为一致。

### navigation/panoramic_detection.py

`panoramic_detection.py` 承接 `rotate_segmentation()` 中 detector 输入与输出整理：

```text
12-view RGB/depth buffers
  -> previous/current/next triplet stitching
  -> detector raw B/C outputs
  -> center-panel filtering
  -> pose/depth triplet alignment
```

关键边界：

```text
panoramic_detection 不创建 mapper ObjectNode；
不保存 debug 文件；
不调用 LLM/VLM；
不判断目标是否成功。
```

### navigation/detection_artifacts.py

`detection_artifacts.py` 集中检测阶段 debug 产物写入：

```text
comb_img / comb_depth
B_dino_result / C_dino_result
C_dino_pcd / real_C_objs
real_C_obj_image / real_C_obj_image_bbox
```

该模块是纯 IO adapter。它只维护路径和文件写入，不读取 instruction plan，不修改
mapper state。

### navigation/object_cluster_merger.py

`object_cluster_merger.py` 承接 C 类检测结果的跨视角实例合并：

```text
candidate C object groups
  -> symmetric point-cloud overlap
  -> asymmetric containment guard
  -> merged physical object instances
```

核心原则：

```text
这是 instance-level geometry merge；
不是语义 verifier；
不能因为 tag 相同就合并对象；
不能把合并结果直接当作 target success。
```

### navigation/object_view_projector.py

`object_view_projector.py` 将 3D object point cloud 投影回最合适的全景 RGB 证据：

```text
object pcd + robot orientation
  -> 15-degree panorama bin
  -> raw frame or stitched half-view
  -> projected bbox
  -> ObjectViewEvidence
```

它为 bbox tag refinement 提供视觉证据，但不调用 VLM，不更新对象类别。

### navigation/bbox_refinement.py

`bbox_refinement.py` 包装 legacy bbox VLM tag refinement：

```text
ObjectViewEvidence
  -> ask_gpt_object_in_box()
  -> refine_tag_with_target_obj_list()
  -> update object tag/confidence
```

这里的语义边界非常重要：

```text
bbox_refinement 只修正 mapper object label；
final instruction satisfaction 仍由 instruction_adapter.verifier 负责；
不能在这里 stop。
```

### navigation/path_progress.py

`path_progress.py` 承接 `step_mod()` 中 path/path_index/waypoint 推进：

```text
current waypoint
  -> low-level action
  -> path index advance
  -> path exhausted replan
```

它允许调用原 agent planning 入口来保持 legacy 行为，但不运行 visual check、
final verifier 或 object rejection。

### navigation/goal_approach_controller.py

`goal_approach_controller.py` 承接 found-goal 后的几何控制：

```text
distance to selected object
rotate toward object before check_again
recover action after final-instruction rejection
```

该模块只处理几何和 action，不承担语义确认。final verifier 仍在 agent 主流程中显式调用。

### navigation/action_controller.py

`action_controller.py` 统一 mapper-local waypoint 到 Habitat simulator 坐标的转换：

```text
mapper local waypoint
  -> Habitat waypoint [x, simulator_y, z]
  -> geodesic distance
  -> low-level planner action
```

坐标转换集中后，`step_mod()` 不再到处手写：

```python
pid_waypoint = waypoint + initial_position
pid_waypoint = [x, sim_y, z]
```

核心原则：

```text
action_controller 只做运动控制坐标转换；
不做目标判断；
不做 verifier；
不更新 mapper state。
```

### artifact_utils/

`artifact_utils` 是 Phase5 的基础 IO 层，目前包含：

```text
path_builder.py
  episode_dir()
  episode_subdir()
  detection_step_dir()

pointcloud_writer.py
  write_line_set()
  write_point_cloud()
```

它解决的是工程管理问题：路径规则和目录创建不应该散落在 mapping/navigation
核心算法里。当前已接入：

```text
navigation.detection_artifacts
navigation.observation_pipeline
mapping.frontier_clusterer frontier_line writer
```

核心边界：

```text
artifact_utils 只负责 IO 路径和写文件；
不构造几何；
不读取 instruction；
不修改 mapper/agent runtime state。
```

## Phase5：Prompting 层

新增目录：

```text
prompting/
  templates.py
  schemas.py
  registry.py
```

职责：

```text
templates.py
  只维护 prompt 文本和 legacy prompt 兼容入口；
  不读取 mapper/agent 状态，不调用 LLM/VLM。

schemas.py
  集中维护 Pydantic response schema；
  instruction parser、concept grounding、relation verifier、final verifier
  和 bbox VLM 复核共用这里的 schema。

registry.py
  维护 prompt_id、trace_label、schema_name 和模板版本；
  调用点不再散落裸 trace_label 字符串。
```

本阶段保持行为不变：

```text
消息结构不变；
trace label 的字符串值不变；
LLM/VLM client 入口不变；
日志路径不变；
prompt 文本只迁移，不重新调参。
```

已迁移入口：

```text
instruction_adapter/parser_llm.py
instruction_adapter/grounding.py
instruction_adapter/concept_matcher.py
instruction_adapter/relation_verifier.py
instruction_adapter/verifier.py
cv_utils/gpt_utils.py
mapper_with_process_obs.py legacy room/object/relocate prompt import
```

这次重构没有改变普通 benchmark 的开关条件：

```text
没有 InstructionPlan / InstructionSpec:
  final verifier auto-bypass
  object search 走原 STRIVE target tag 分支
  check_again 仍保留原 benchmark 复核路径

有 InstructionPlan:
  target/anchor 走 prompt-first concept matcher
  final verifier 处理原始自然语言、关系和视角充足性
```

### Verified Pair View-Control 闭环

复杂指令中的 `need_better_view` 不是失败状态。若 final verifier 返回：

```text
semantic_satisfied = true
decision = need_better_view
verified_relation_context exists
```

运行时会进入：

```text
InstructionExecutionState.mode = better_view_for_verified_pair
```

并保存：

```text
pending_verified_pair:
  candidate_uid
  relation_context
  dynamic semantic edge
  view_objective
  baseline view quality
```

之后 `InstructionObjectSearchPolicy` 优先返回该 verified pair 对应的目标实例或同一
局部几何簇，不再先做全图 terminal/anchor grounding。这样 step N 已确认
`book on shelf` 但视角不足时，后续只围绕这条已验证语义边做视角优化，而不是重新
搜索新的 book / shelf。

该闭环只固定运行时证据，不写对象常识规则。若 mapper 近距离更新导致 uid 漂移，
policy 使用同标签和局部几何距离做实例关联；语义是否满足仍由 final verifier 和
dynamic semantic edge 负责。

### Dynamic Edge 单调性

`SemanticEdgeCache` 和 `RelationPairLedger` 遵守单调规则：

```text
VLM verified edge 不会被后续 geometry reject 覆盖
accepted_relation 不会被 rejected_relation 降级
```

几何仍作为新 pair 的硬预筛，避免无证据地调用 VLM；但一旦 check-again / final
verifier 提供了强视觉证据，后续点云高度、投影或 bbox 抖动只能作为弱负证据，不能
静默删除已验证语义边。

## Phase6：删除废代码

本阶段只做删除和收口，不引入 cup/support-region 新能力：

```text
删除 make_plan_mod_no_relocate_no_gpt()
删除 get_candidate_room_fully_explored_no_gpt()
删除 check_again 中已确认无效的旧注释块
删除 mapper / agent 中大块历史注释代码
```

保留行为：

```text
--no_gpt_relocate
  -> make_plan_mod_no_relocate(use_gpt_relocate=False)
  -> mapper.get_candidate_room_fully_explored_by_distance()
  -> planning.room_policy.select_nearest_frontier_room()
```

这让 no-LLM relocate 从“重复主流程分支”变成“可替换 room policy”，后续
support-region search 可以接入 planning 层，而不是继续塞进
`object_found_no_gpt()`。

清理结果：

```text
mapper_with_process_obs.py:
  4084 lines -> 2574 lines
  frontier cluster analysis / node materialization 已拆到 mapping 层

objnav_agent_with_process_obs.py:
  2191 lines -> 1773 lines
  panoramic detection / object merge / path progress 已拆到 navigation 层

quality convergence:
  删除确认废弃的历史注释块
  增加轻量回归测试
  artifact path builder 纳入 scripts/check_refactor.sh
```

## 后续可继续拆分

建议下一阶段按以下顺序继续：

```text
Phase5:
  把 final_instruction_check() 拆成 evidence builder、constraint runner、
  final decision applier 三个类。

Phase6:
  把 mapper 的 room/node/frontier 更新从 Instruct_Mapper 中拆出 service，
  先保留原数据结构，避免一次性迁移所有字段。

Phase7:
  用显式 Result/Event 对象替代 found_goal/need_check_again 等跨函数布尔状态，
  让 benchmark metrics 与 instruction metrics 的写入更清楚。
```
