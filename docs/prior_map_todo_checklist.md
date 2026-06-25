# STRIVE 先验地图模式待做 Checklist

本文档跟踪先验地图模式的实现工作。先验地图只能提供 context、ranking 和 evidence
summary，不能直接发布导航目标或绕过 final verifier。

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

- [ ] 新增 `prior_map/prompt_context.py`。
- [ ] 实现 prior map natural-language summary。
- [ ] 实现 OSM-like compact XML summary。
- [ ] 实现 `SearchPriorResult` 的 prompt-friendly summary。
- [ ] 新增或迁移 SoM visualizer 到 `prior_map/visualizer.py`。
- [ ] 输出 global view、room view、markers 和 legend。
- [ ] 限制 prompt context 长度，避免把完整地图无节制塞给 LLM。
- [ ] 增加测试：marker id 稳定、room/object labels 可追踪。

## 8. Simulation Integration

- [ ] 增加 CLI 参数：`--enable_prior_map`、`--prior_map_path`、`--prior_map_source`、`--prior_map_alignment`。
- [ ] 在 episode/run 初始化时加载 prior map。
- [ ] 在每个 planning cycle 后调用 `PriorMapMemory.update_from_mapper()`。
- [ ] 在 room/frontier/object selection 前调用 `PriorMapQueryService.query()`。
- [ ] 将 `SearchPriorResult` 作为排序 context 传给 planning policy。
- [ ] 保存 `prior_map/query_*.json` 和 `search_prior_result_*.json`。
- [ ] 做 1 episode offline smoke，确认无 live API 依赖。

## 9. Real-Robot Integration

- [ ] 在实物高层 node 中增加 prior map 参数。
- [ ] 用 `SemanticMapSnapshot` 调用 `PriorMapMemory.update_from_snapshot()`。
- [ ] 将 `SearchPriorResult` 作为 ranking/context evidence 传给 `SemanticMapSnapshotPolicyContext` / 现有 planning policy adapter。
- [ ] prior map 和 live map 冲突时，live evidence 优先，并写入 diagnostics。
- [ ] 确认 prior map 不直接生成 `/way_point`。
- [ ] 增加 fake snapshot replay 测试。

## 10. Artifacts And Evaluation

- [ ] 保存 `prior_map/base_map.json`。
- [ ] 保存 `prior_map/alignment.json`。
- [ ] 保存 `prior_map/runtime_state_*.json`。
- [ ] 保存 `prior_map/query_*.json`。
- [ ] 保存 SoM PNG 和 marker metadata。
- [ ] 在 metrics/debug 中记录 prior map enabled、top room/object prior、alignment confidence。
- [ ] 增加 failure mode 统计：wrong prior、alignment mismatch、live conflict、prior exhausted。

## 11. Acceptance

- [ ] Contract/loaders/alignment/memory/query 单元测试通过。
- [ ] 仿真 smoke：启用 prior map 后可以完成至少一个 episode，不改变 stop authority。
- [ ] 仿真 A/B：关闭 prior map 后行为与当前主线一致。
- [ ] 实物 replay smoke：fake `SemanticMapSnapshot` 能生成 prior-aware `NavigationIntent`。
- [ ] 文档同步：`docs/prior_map_mode.md` 与本 checklist 保持一致。
