# STRIVE 先验地图模式待做 Checklist

本文档跟踪先验地图模式的实现工作。先验地图只能提供 context、ranking 和 evidence
summary，不能直接发布导航目标或绕过 final verifier。

## 0. 当前审计结论

- [x] 当前 1-11 阶段完成的是 prior-map contract、loader、memory、query、artifact 和闭环 smoke。
- [x] 当前 HM3D `*.semantic.txt` builder 只能生成 scene semantic inventory：包含 object label、instance id、region id，但不包含 object position、room boundary、room centroid 或 room-room topology。
- [x] 当前 `wcojb4TFT35` A/B 完整测试验证了 prior-on 不破坏主线：两边均 `success=1.0`、`Episode Steps=283`、`SPL=0.2402441547`。
- [x] 当前 `wcojb4TFT35` A/B 同时证明 `.semantic.txt` prior 没有改变路径：`travel distance`、`accepted_candidate_uid`、`final_stop_accept_step` 完全一致。
- [x] 当前 `.semantic.txt` prior 的 `alignment_type=unavailable`、`enabled_for_ranking=false`，因此 `frontier_biases=[]`，不能作为可导航先验地图。
- [x] 下一阶段目标改为 HM3D ground-truth geometric prior：用 Habitat semantic scene + navmesh/topdown 生成 FloorPlan-VLN 风格的 room/object/topology prior。

## 1. Contract Layer

- [x] 新增 `prior_map/contracts.py`。
- [x] 定义 `PriorMapData`，包含 scene id、rooms、objects、topology、source format、frame id 和 metadata。
- [x] 定义 `PriorRoom`，包含 uid、label、boundary、centroid、neighbors、confidence、source。
- [x] 定义 `PriorObject`，包含 uid、label、position、parent room、exactness、confidence、aliases。
- [x] 定义 `PriorTopologyEdge`，表达 room-room、room-object 或区域连接关系。
- [x] 定义 `PriorObservationRecord`，记录 runtime pose、observed objects、room hypothesis 和 source。
- [x] 定义 `SearchPriorResult`、`RoomPrior`、`ObjectPrior`、`FrontierPrior`。
- [x] 保证 contract 层不 import ROS、Habitat、OpenCV、detector 或 live LLM client。
- [x] 增加 JSON roundtrip 单元测试。

## 2. Loaders

- [x] 新增 `prior_map/loaders.py`。
- [x] 实现 `PriorMapLoader.load(path, source_format="auto")`。
- [x] 支持标准 floorplan JSON。
- [x] 支持 FloorPlan-VLN JSON。
- [x] 支持 OSM/XML 简化格式。
- [x] 支持 HM3D top-down / generated prior map JSON。
- [x] 支持 HM3D scene `*.semantic.txt` 到 canonical `PriorMapData` 的语义清单构建；该来源不是可导航几何先验。
- [x] 为 VLM reconstruction 输出预留 loader。
- [x] 增加最小 fixture 测试，覆盖 room、object、topology 和 metadata。
- [x] 明确 loader 只解析文件，不进行 runtime 坐标对齐和导航决策。

## 3. Alignment

- [x] 新增 `prior_map/alignment.py`。
- [x] 定义 `PriorMapAlignment`，包含 `prior_to_runtime()`、`runtime_to_prior()` 和 `confidence()`。
- [x] 实现 identity alignment。
- [x] 实现 affine 2D alignment：scale、rotation、translation。
- [x] 支持保存和加载 `alignment.json`。
- [x] 增加 inverse consistency 测试。
- [x] 在 diagnostics 中记录 frame id、source points、target points、误差统计。
- [x] 明确坐标对齐失败时 prior map 降级为 prompt context，不参与 frontier/object 排序。

## 4. Memory

- [x] 新增 `prior_map/memory.py`。
- [x] 实现 `PriorMapMemory(base_map, alignment)`。
- [x] 实现 `update_from_mapper(mapper, step)`，用于仿真模式。
- [x] 实现 `update_from_snapshot(snapshot)`，用于实物模式。
- [x] 实现 `mark_room_visited(room_uid, step)`。
- [x] 实现 `mark_object_verified(prior_uid, runtime_uid, step)`。
- [x] 实现 `mark_prior_rejected(prior_uid, reason, step)`。
- [x] 实现 `current_map()`，返回合并 runtime state 后的只读地图视图。
- [x] 增加 visit count、verified/rejected、confidence moving average 测试。
- [x] 确保 memory 不修改原始 `PriorMapData`。

## 5. Query Service

- [x] 新增 `prior_map/query.py`。
- [x] 实现 `PriorMapQueryService.query(plan, runtime_context, memory)`。
- [x] 根据 `InstructionPlan.target_detector_prompts` 和 concept aliases 计算 object relevance。
- [x] 根据 `InstructionPlan.search_priors.room_hints` 计算 room relevance。
- [x] 根据 support objects 和 affordances 生成 support region ranking。
- [x] 根据 visited/exhausted/rejected state 调整 soft score。
- [x] 根据 live observed objects 提高 matching prior 的 confidence。
- [x] 输出 `SearchPriorResult`，包含 score components 和 reason。
- [x] 增加测试：prior 只能改变排序，不能生成 motion goal。

## 6. Policy Adapter

- [x] 新增 `prior_map/policy_adapter.py` 或并入 `prior_map/query.py` 的 adapter 层。
- [x] 实现 `rank_rooms(rooms, prior_result)`。
- [x] 实现 `rank_frontiers(frontiers, prior_result)`。
- [x] 实现 `annotate_target_candidates(candidates, prior_result)`。
- [x] 提供可接入 `planning/room_policy.py` 或 `planning/exploration_policy.py` 排序路径的稳定 API。
- [x] 提供可接入 `planning/target_selection_policy.py` 的 debug annotation，不改变 final verifier 逻辑。
- [x] 增加 A/B 测试：关闭 prior map 时保持现有行为。

## 7. Prompt Context And SoM

- [x] 新增 `prior_map/prompt_context.py`。
- [x] 实现 prior map natural-language summary。
- [x] 实现 OSM-like compact XML summary。
- [x] 实现 `SearchPriorResult` 的 prompt-friendly summary。
- [x] 新增或迁移 SoM visualizer 到 `prior_map/visualizer.py`。
- [x] 输出 global view、room view、markers 和 legend。
- [x] 限制 prompt context 长度，避免把完整地图无节制塞给 LLM。
- [x] 增加测试：marker id 稳定、room/object labels 可追踪。

## 8. Simulation Integration

- [x] 增加 CLI 参数：`--enable_prior_map`、`--prior_map_path`、`--prior_map_source`、`--prior_map_alignment`。
- [x] 在 episode/run 初始化时加载 prior map。
- [x] 在每个 planning cycle 后调用 `PriorMapMemory.update_from_mapper()`。
- [x] 在 room/frontier/object selection 前调用 `PriorMapQueryService.query()`。
- [x] 将 `SearchPriorResult` 作为排序 context 传给 planning policy。
- [x] 保存 `prior_map/query_*.json` 和 `search_prior_result_*.json`。
- [x] 新增 Habitat/ObjectNav dataset 到 canonical `PriorMapData` 的生成脚本。
- [x] 新增 HM3D semantic txt 到 canonical `PriorMapData` 的生成脚本，避免 ObjectNav goal prior 泄漏目标位置；该脚本只用于 semantic inventory smoke。
- [x] 做 1 episode offline smoke，确认无 live API 依赖。

## 9. Real-Robot Integration

- [x] 在实物高层 node 中增加 prior map 参数。
- [x] 用 `SemanticMapSnapshot` 调用 `PriorMapMemory.update_from_snapshot()`。
- [x] 将 `SearchPriorResult` 作为 ranking/context evidence 传给 `SemanticMapSnapshotPolicyContext` / 现有 planning policy adapter。
- [x] prior map 和 live map 冲突时，live evidence 优先，并写入 diagnostics。
- [x] 确认 prior map 不直接生成 `/way_point`。
- [x] 增加 fake snapshot replay 测试。

## 10. Artifacts And Evaluation

- [x] 保存 `prior_map/base_map.json`。
- [x] 保存 `prior_map/alignment.json`。
- [x] 保存 `prior_map/runtime_state_*.json`。
- [x] 保存 `prior_map/query_*.json`。
- [x] 保存 SoM PNG 和 marker metadata。
- [x] 在 metrics/debug 中记录 prior map enabled、top room/object prior、alignment confidence。
- [x] 增加 failure mode 统计：wrong prior、alignment mismatch、live conflict、prior exhausted。

## 11. Acceptance

- [x] Contract/loaders/alignment/memory/query 单元测试通过。
- [x] 仿真 smoke：启用 prior map 后可以完成至少一个 episode，不改变 stop authority。
- [x] 仿真 A/B：关闭 prior map 后行为与当前主线一致。
- [x] 真实 HM3D semantic inventory A/B：同一 episode 下 prior-on 生成 `SearchPriorResult`，prior-off 不生成 prior runtime artifact，但 prior-on 没有改变路径。
- [x] 实物 replay smoke：fake `SemanticMapSnapshot` 能生成 prior-aware `NavigationIntent`。
- [x] 文档同步：`docs/prior_map_mode.md` 与本 checklist 保持一致。

## 12. HM3D Ground-Truth Geometric Prior Builder

- [x] 新增 `prior_map/hm3d_groundtruth.py`。
- [x] 新增 `scripts/build_hm3d_groundtruth_prior_map.py`。
- [x] 定义 `HM3DGroundTruthPriorMapBuilder`，只负责从仿真 ground truth 生成 `PriorMapData`，不依赖 runtime mapper 的已探索点云。
- [x] 定义 `HM3DGroundTruthBuildConfig`：topdown resolution、floor height tolerance、min room area、mask dilation radius、是否拆分 disconnected components、是否保留 structural labels。
- [x] 支持从已有 Habitat simulator 构建：`build_from_sim(sim, scene_id, config)`。
- [x] 支持从 scene directory 构建：读取 `*.basis.glb`、`*.semantic.glb`、`*.basis.navmesh`，初始化 Habitat-Sim 后调用 `build_from_sim()`。
- [x] 从 `sim.semantic_scene.objects` 提取 object uid、label/category、AABB center、AABB size、semantic object id、parent region id。
- [x] 生成 `PriorObject.position_xyz`，并在 metadata 中记录 AABB、semantic id、region id、navmesh snap point、distance-to-navmesh。
- [x] 从 `sim.semantic_scene.regions` 提取 region uid、label、AABB、floor/level id。
- [x] 从 `sim.pathfinder` 或 topdown map 采样可通行 mask，坐标保持 Habitat world frame：2D 平面使用 `(x, z)`，3D 使用 `(x, y, z)`。
- [x] 用 region AABB 与 navmesh traversable mask 相交生成 room mask。
- [x] 对同一 semantic region 内的 disconnected navigable components 做拆分或记录，避免把不可连通区域当成同一个 room。
- [x] 从 room mask 生成 `PriorRoom.boundary_xy`、`centroid_xy`、area、nav sample count、floor id。
- [x] 实现 room-object containment：优先 semantic parent region，其次 object center 是否落入 room polygon/mask，最后使用 nav-snap 最近 room。
- [x] 实现 room-room topology：基于 mask dilation/contact points 和 pathfinder geodesic distance 生成 `room-room connected` edges。
- [x] 实现 room-object topology：为每个 prior object 生成 `room-object contains` edge。
- [x] 写出 canonical `PriorMapData` JSON，`source_format="hm3d_groundtruth_semantic_scene"`，`frame_id="habitat_world"`。
- [x] 写出 `alignment.json`：仿真默认 `identity`、`base_confidence=1.0`、`enabled_for_ranking=true`。
- [x] 明确 builder 不读取 ObjectNav goal positions，不从 episode target 泄漏答案。
- [x] 增加 fake semantic scene + fake navmesh 单元测试：object position、room boundary、centroid、neighbors、topology 全部生成。
- [x] 增加真实 scene smoke：`wcojb4TFT35` 导出的 TV `position_xyz != None`，至少一个 room 有非空 `boundary_xy`，至少一个 `room-room` edge。
  - 容器内 smoke 已导出 `logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json`：`object_count=779`、`room_count=90`、`topology_edge_count=1039`。
  - TV `prior_object:wcojb4TFT35:tv_349` 使用 `semantic_glb_texture_bounds` 几何来源，`position_xyz=[-0.18259349465370178, -0.9289894700050354, 1.8373835682868958]`。

## 13. Geometry-Aware Query And Ranking

- [x] 扩展 `PriorMapQueryService`，在 prior object 有 `position_xyz` 且 alignment 可用时，按目标对象位置和 parent room 生成几何 search prior。
- [x] 将 target object prior 的 parent room 排名提前到 room ranking，而不是只给所有 unvisited room 相同 `0.2` 分。
- [x] 为 runtime frontier 建立稳定 uid 和 world position，供 `PriorMapQueryService._rank_frontiers()` 消费。
- [x] 实现 runtime frontier/node 坐标到 `habitat_world` 的转换，避免 mapper local frame 和 prior frame 混用。
- [x] 基于 geodesic distance 或 euclidean fallback，计算 frontier 到 top prior room/object 的 score delta。
- [x] `frontier_biases` 必须包含 score components：target object relevance、target room relevance、distance score、alignment confidence、visited/exhausted penalty。
- [x] 当 alignment 不可用或 prior object 无坐标时，显式降级为 prompt context，并在 diagnostics 写明 `geometry_disabled_reason`。
- [x] 增加测试：有几何 prior 时 `frontier_biases` 非空；无几何 prior 时保持现有行为。
- [x] 增加测试：prior 只能改变 frontier/room 排序，不能生成 motion goal、不能绕过 final verifier。

## 14. Active Planner Integration

- [x] 在 active frontier/room selection 前刷新 geometry-aware prior query。
- [x] 将 `frontier_biases` 接入实际使用的 frontier/waypoint selection 路径，而不是只写 artifact。
- [x] 在 `planning/room_policy.py` 中保留 deterministic nearest-frontier baseline，并记录 prior-adjusted distance。
- [x] 在 active non-relocate planning 路径中记录 prior-on 选中的 frontier uid、raw distance、prior score、adjusted score。
- [x] 增加 debug artifact：`prior_map/chosen_frontier_*.json`，包含候选排序前后对比。
- [x] 增加 A/B guard：关闭 prior map 时排序结果和当前主线一致。
- [x] 确认 prior map 不直接设置 `found_goal`，目标发现仍来自 live perception + existing target selection。
- [x] 确认 final stop authority 仍由 physical stop contract + final verifier 决定。

## 15. Ground-Truth Prior Acceptance

- [x] 构建真实 `wcojb4TFT35` HM3D ground-truth prior map。
- [x] 检查 `base_map.json`：TV `position_xyz` 非空、parent room 非空、room boundary/centroid 非空、room-room topology 非空。
- [x] 检查 `alignment.json`：`identity`、`confidence=1.0`、`enabled_for_ranking=true`。
- [ ] 运行同一 episode prior-on/off A/B，确认 prior-on 产生 `frontier_biases`。
- [ ] 运行同一 episode prior-on/off A/B，确认 chosen frontier 或 waypoint 至少在一次 planning cycle 中不同。
- [ ] 记录 first target seen step、final accept step、travel distance、SPL、LLM call counts。
- [ ] 若单 episode 未改善，先确认 prior signal 是否生效；不要只用 success 判断 prior 是否有效。
- [ ] 跑多 episode 统计：至少包含 mean/median first target seen step、success、SPL、travel distance。
- [ ] 增加 failure mode：ground-truth prior unavailable、geometry missing、alignment disabled、frontier bias empty、prior did not affect action。

## 16. 文档同步

- [ ] 更新 `docs/prior_map_mode.md`，区分 semantic inventory prior、ObjectNav goal oracle prior、ground-truth geometric prior。
- [ ] 补充 HM3D ground-truth prior 构建命令和输入文件要求。
- [ ] 补充 A/B 评测标准：必须检查 `frontier_biases`、chosen frontier 差异和 first target seen step。
- [ ] 补充安全边界：prior map 只做 ranking/context，不能生成 STOP 或绕过 final verifier。
