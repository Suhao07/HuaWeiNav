# Instruction Adapter Changelog

## 2026-06-13

### Added

- 新增 `real_robot/contracts.py`，作为 STRIVE 实物模式的第一层平台无关接口：
  - `RealObservation` / `CameraFrame` 描述真实 RGB、深度引用、点云引用和位姿；
  - `DetectionFrame` 统一 detector bbox、label、confidence、track id 和 mask 引用；
  - `ObjectNodeSnapshot`、`RoomSnapshot`、`SemanticMapSnapshot` 提供高层 planner 的只读地图视图；
  - `NavigationIntent`、`MotionGoal`、`ViewpointGoal`、`NavigationStatus` 明确 STRIVE 高层与下层运动控制的异步边界；
  - `ViewEvidence` 和 `RuntimeDecision` 为 final verifier、relation verifier 和实物运行复盘提供结构化记录。
- 新增 `real_robot/__init__.py`，集中导出实物 contract，后续 ROS/SysNav/bag replay adapter 都应依赖该包。
- 更新 `docs/real_robot_deployment.md`：
  - 明确 contract 层只保存 JSON-friendly metadata、image/path reference 和几何状态；
  - 明确 VLM/verifier、mapper/planner、motion controller 和 runtime 的职责划分；
  - 将初始 runtime skeleton 更新为异步 `MotionGoal -> NavigationStatus -> ViewEvidence` 形式。

### Tests

- 新增 `tests/test_real_robot_contracts.py`：
  - 约束 contract 层不引入 ROS、Habitat、numpy 等平台依赖；
  - 验证 detection frame 并行字段和 bbox 合法性；
  - 验证 `NavigationIntent -> MotionGoal`、`ViewpointGoal -> MotionGoal` 转换；
  - 验证 map snapshot lookup、view evidence verifier payload 和 navigation status 终止语义。

## 2026-06-12

### Changed

- final-stop physical authority 收敛：
  - VLM 只判断 `semantic_satisfied`、关系满足度和 `view_sufficient_for_stop`；
  - `distance_satisfied`、可达性、碰撞和“是否没有更近可执行视角”只由
    planner/geometry evidence 决定；
  - VLM 输出的 `infeasible_or_not_applicable=true` 只作为 report 记录，不能覆盖
    `within_final_stop_distance`；
  - 只有 evidence 中存在 `planner_infeasibility_proof.infeasible_by_geometry=true`
    时，best-available stop 才能绕过未满足的物理距离合同。
- final-stop hard constraints 结构化：
  - `distance_to_object` 与 benchmark `success_distance` 生成
    `hard_stop_constraints.within_final_stop_distance`；
  - final verifier schema 新增 `hard_constraints`，用于回传 VLM 对 hard contract 的
    report，但不授予 VLM 物理 override 权限；
  - 如果 VLM 返回 `accept` 但 hard constraints 未满足且没有 planner-owned 不可达证明，
    verifier 后处理会降级为 `need_better_view`；
  - 这是一条通用 stop contract，不写任何目标类别规则。
- 修正 view-control budget 语义：
  - `budget_exhausted=true` 只表示 proposal 用完，或软视角优化预算在物理合同已满足/无剩余
    物理 proposal 时耗尽；
  - 如果 `within_final_stop_distance` 仍未满足且仍有
    `remaining_physical_contract_proposals`，attempt/verifier 预算不会被解释为可停止；
  - `no_improvement_rounds` 只产生 `progress_stalled=true`，用于提示切换 proposal 或重采样，
    不再等价于“没有可行更好视角”；
  - view-control context 新增 `remaining_proposals` 和
    `closest_remaining_proposal_distance`，让 VLM 判断是否真的没有更近且可见的候选视角。
- 修正 better-view 路径端点：
  - 当 verifier 的 `view_objective` 要求靠近目标时，check-again path 优先朝当前目标实例的
    可接近点生成，而不是复用可能偏远的发现 waypoint；
  - 如果该目标点不可达，再退回 confirmed target waypoint。
- 统一 final-stop authority：
  - 当 benchmark object-goal 或自然语言 `InstructionPlan` 已安装时，Habitat `STOP`
    只能由 `FinalInstructionVerifier` 的 `decision=accept` 触发；
  - legacy `final_check()` 的 ray/voxel 可见性不再能单独结束 episode，只保留给无 plan
    的原始 STRIVE 路径；
  - 若 verifier 返回 `need_better_view` 或其它非 accept 决策，agent 会阻断 stop action，
    继续执行 view-control 或采集新的当前视角证据。
- view-control 增加 stable visual reference：
  - `semantic_satisfied=true` 后，`ViewControlState` pin 首次语义确认的
    `pinned_visual_evidence`，并单独记录后续 `latest_visual_evidence`；
  - final verifier 会同时看到当前 stop evidence 与 pinned visual reference，用于判断
    当前 bbox/crop 是否仍对准同一视觉目标；
  - 该机制防止近距离时 3D object cluster 投影漂移到支撑物、墙面或背景后误导
    view-control，但不写任何目标类别规则。
- final verifier prompt 更新：
  - 明确 `pinned_visual_evidence` 是目标身份参照，不是自动 stop 许可；
  - 若当前 evidence 漂移到 support/background，而真实目标被裁切或只弱可见，应返回
    `need_better_view`，最终是否 limited accept 仍由 VLM 根据 evidence 和 view-control
    历史判断。
- view-control 增加 best-available 收敛：
  - `need_better_view` 后如果仍有可执行 viewpoint proposal，agent 继续改善视角；
  - `ViewControlState` 现在显式维护 `max_attempts`、`max_verifier_calls` 和
    `max_no_improvement_rounds`，并记录 `best_visual_evidence`；
  - 如果 viewpoint proposal 或 verifier 调用达到预算上限，agent 只把
    `budget_exhausted`、pinned/best evidence、remaining proposal 摘要和 attempt history
    传回 final verifier；
  - 连续无改善轮数只产生 `progress_stalled=true`，不再宣称 proposal 已耗尽；
  - Python 控制层不再把 `need_better_view` 硬改成 `accept`，best available stop 由
    prompt 根据原始任务、可见证据和预算上下文决定。
- final verifier prompt 增加 bounded view-control 语义：
  - VLM 会看到 `budget_exhausted`、`best_visual_evidence`、attempt history 和剩余 proposal；
  - prompt 明确要求尽量靠近，但必须保持 target 和 relation anchor 清楚可见；
  - 当预算已耗尽时，prompt 应判断当前/最佳/固定参照证据是否足够作为 best available
    stop，而不是无限返回 `need_better_view`。
- 删除控制层冲突规则：
  - 移除 initial-accept deferral，agent 不再把 VLM 的 `accept` 改写成
    `need_better_view`；
  - final view geometry 不再通过 `STRIVE_FINAL_VIEW_*` 阈值改写 VLM 决策；
  - best available 收敛不再由 Python 合成 accept；
  - 新增 `STRIVE_FINAL_VERIFIER_MAX_PER_CANDIDATE`，即使某条路径重置了
    `ViewControlState`，同一 candidate 也不会无限进入 final verifier。
- metrics 增加 `accepted_distance_to_target` 和 `accepted_distance_source`，用于区分
  instruction-level terminal instance 距离与 Habitat 原始 `distance_to_goal`。
- 实验产物增加 run 隔离信息：
  - `objnav_benchmark_with_process_obs.py` 新增 `--run_id` 和 `--clean_save_dir`；
  - `run_manifest.json`、`metrics.csv`、final verifier evidence/result 中写入 `run_id`；
  - 复用同名 `--save_dir` 时建议加 `--clean_save_dir`，避免旧 metrics 和新 verifier 产物混读。

### Tests

- `tests/test_view_control_state.py` 新增 stable visual reference、bounded budget 和 best visual evidence 回归测试，确保后续几何上更大的漂移帧不会覆盖首次语义确认帧，并且同一目标不会无限 view retry。
- `tests/test_final_stop_verifier.py` 更新为 prompt-first guidance 回归测试，确保距离进入
  hard stop constraint，投影失败和预算状态作为 VLM prompt facts。
- 新增 hard stop constraint 和 no-improvement budget 语义测试，防止再次把
  `distance_to_object > success_distance` 降成普通提示，或把 `progress_stalled` 误当作
  `budget_exhausted`。

## 2026-06-11

### Added

- final-stop confirmed target 闭环修复：
  - `semantic_satisfied=true + need_better_view` 后，单目标 object-goal 也会进入
    `InstructionExecutionState.mode=better_view_for_verified_pair`；
  - pending state 保存 `candidate_record`，mapper uid 漂移时可用 centroid/position
    继续关联同一局部目标簇；
  - `ViewControlState` 对同一 candidate 合并 VLM view objective，不再因为 prompt
    wording 变化清空 attempted proposals；
  - final-stop proposal 排序先满足 `required_stop_distance`，再比较可见性、居中和尺度；
  - `after_check_again()` 和 reject recovery 在 pending better-view 时不再 reset
    `view_control_state`、不再把 `object_final.tag` 改成 `nothing`。
- benchmark ObjectNav 默认复用 instruction target pipeline：
  - 普通 benchmark 自动安装 object-goal `InstructionPlan`，例如 `Find the <tv>.`；
  - 该 plan 复用 mapper 搜索、ledger、check_again evidence、final verifier 和
    view-control，不再走 benchmark-only final verifier 分支；
  - 显式 instruction mode 继续使用原始自然语言 prompt 和完整 InstructionPlan；
  - benchmark object-goal plan 不读取 episode metadata 的复杂自然语言，因此不引入
    relation、sequence、room、count 等复杂指令语义；
  - `metrics.csv` 新增 `final_stop_success`、`final_stop_decision`、
    `final_stop_accept_step` 和 `final_stop_mode`。
- final-stop evidence 增加 stop-distance facts：`distance_to_object` 和 benchmark
  provider 的 `success_distance` 会写入 verifier prompt，用于提示 VLM 在可行时继续靠近。
- 新增 `benchmark/` provider 抽象：
  - `BenchmarkSpec` 统一记录 benchmark、split、dataset path、success distance
    和单 episode materialization provenance；
  - `hm3d_objectnav`、`hm3d_ovon`、`gibson_objectnav`、`gibson_custom`
    provider 初步拆分；
  - HM3D-OVON 支持显式 `--benchmark hm3d_ovon --benchmark_split ...`，
    `auto` 仅保留旧单场景 smoke test 兼容路径。
- 新增 `docs/benchmark_providers.md`：
  - 说明 provider 边界、HM3D-OVON split 选择、success distance 对齐和
    Gibson custom wrapper 的迁移边界。
- 新增 `docs/real_robot_deployment.md`：
  - 整理 STRIVE 实物部署接口设计；
  - 对照 SysNav ROS2 传感器、语义地图、VLM、exploration planner、
    local planner 和 path follower；
  - 明确 RealObservation、DetectionFrame、SemanticMapSnapshot、
    NavigationIntent 等建议 contract；
  - 补充 Theta Z1 与 RealSense 的可插拔 CameraAdapter 设计；
  - 给出离线 bag replay 到真车测试的分阶段落地路线。
- 新增轻量回归测试：
  - `tests/test_frontier_extractor.py` 覆盖 frontier 自适应半径；
  - `tests/test_panoramic_detection.py` 覆盖 panorama triplet index 与 center-panel bbox filter；
  - `tests/test_artifact_paths.py` 覆盖 artifact path builder。
- 新增 `navigation/planner_loop.py`：
  - 抽离每个 planning cycle 的 observation->mapping 前缀；
  - 统一执行 panoramic observation、point-cloud merge、obs debug 保存、
    `mapper.get_nodes()` 和 `mapper.update_obj()`；
  - 返回 `PlanningCycleResult`，不处理目标确认、room exploration 或 stop。
- 新增 `navigation/panoramic_detection.py`：
  - 抽离 stitched triplet 构造、中心面板 bbox 过滤和 pose/depth triplet 对齐；
  - 只做 detector 输入输出整理，不创建 mapper object。
- 新增 `navigation/detection_artifacts.py`：
  - 集中保存 detection RGB/depth、mask overlay、C object point cloud、
    projected bbox image 等 debug artifact。
- 新增 `navigation/object_cluster_merger.py`：
  - 抽离 C 类对象跨视角 point-cloud overlap merge；
  - 保留 asymmetric containment guard，避免小物体被大物体错误吞并。
- 新增 `navigation/object_view_projector.py`：
  - 将 merged object point cloud 投影回全景 RGB 证据图；
  - 输出 projected bbox，供 bbox tag refinement 使用。
- 新增 `navigation/bbox_refinement.py`：
  - 封装 legacy `ask_gpt_object_in_box()` 与 tag refine；
  - 只更新 mapper object label/confidence，不执行 final instruction stop。
- 新增 `navigation/path_progress.py`：
  - 抽离 `step_mod()` 中 path/path_index/waypoint 推进和 path exhausted replan。
- 新增 `navigation/goal_approach_controller.py`：
  - 抽离 found-goal 后的距离计算、朝目标转向和 final-instruction reject 后动作恢复。
- 新增 `mapping/frontier_clusterer.py`：
  - 抽离 `get_nodes()` 中 DBSCAN cluster 到 frontier/fallback center 的转换；
  - 仍复用 mapper geometry helper，但不创建 node、不更新 graph。
- 新增 `mapping/node_candidate_builder.py`：
  - 抽离 traversable candidate 到 topology node 的 materialization；
  - 负责 candidate bearing、80-degree local pcd sector 和 `add_node/add_edge` 调用。
- 新增 `artifact_utils/path_builder.py`：
  - 集中 episode/subdir/detection step debug 路径构造。
- 新增 `artifact_utils/pointcloud_writer.py`：
  - 集中 line-set / point-cloud debug writer 的目录创建和 Open3D 写入。

### Changed

- `objnav_benchmark_with_process_obs.py` 不再内联 HM3D-OVON 文件搜索和
  filtered dataset 构造，改为通过 benchmark provider 准备 `BenchmarkSpec`
  并将 `benchmark_spec.json` 写入日志目录。
- `config_utils.hm3d_config()` 支持显式 `dataset_path` 和 `success_distance`，
  避免 HM3D-OVON 被普通 HM3D 默认成功距离覆盖。
- `scripts/check_refactor.sh` 增加 `PYTHONPATH=. pytest -q tests`，让重构检查覆盖轻量回归测试。
- `navigation/panoramic_detection.py` 将 `combine_image` 改为函数内懒加载：
  - 运行时行为不变；
  - 纯 bbox/filter 单测不再依赖本机 cv2 安装。
- 清理 `objnav_agent_with_process_obs.py` 和 `mapper_with_process_obs.py` 中已确认废弃的历史注释块。
- `navigation/observation_pipeline.py` 扩展为全景观测采集模块：
  - 新增 panoramic buffer reset；
  - 新增 12-view RGB/depth/pose/point-cloud sweep 采集；
  - 保持 segmentation、object association 和 LLM bbox refine 在 agent 中。
- `objnav_agent_with_process_obs.py` 的 `rotate_panoramic()` 改为调用
  `collect_panoramic_observations()`，自身只保留 orchestration 和 segmentation 调用。
- `objnav_agent_with_process_obs.py` 的 `rotate_segmentation()` 改为清晰的五段流水线：
  - triplet detection；
  - B 类对象即时入图；
  - C 类对象缓存；
  - C 类对象几何合并；
  - bbox refinement 后统一 association。
- `objnav_agent_with_process_obs.py` 的 `step_mod()` 改为委托 path progress
  和 goal approach 几何控制；final verifier 调用点保持不变。
- `make_plan_mod_no_relocate()` 与 `make_plan_mod_relocate()` 复用
  `run_observation_mapping_cycle()`，减少两个规划入口的重复前缀。
- `mapper_with_process_obs.py` 的 `get_nodes()` 进一步委托：
  - `analyze_frontier_clusters()` 负责 frontier/fallback candidate center 分析；
  - `add_nodes_from_candidates()` 负责候选 node 创建和 edge 连接。
- `navigation.detection_artifacts`、`navigation.observation_pipeline` 和
  `mapping.frontier_clusterer` 开始复用 `artifact_utils` 路径/writer 基础层。

## 2026-06-10

### Added

- 新增 `navigation_core/view_geometry.py`：
  - 路径插值；
  - stop 半径裁剪；
  - 朝向矩阵计算；
  - class-agnostic 投影视角质量评分。
- 新增 `planning/viewpoint_policy.py`：
  - 从 Habitat shortest path 生成 check-again / view-control 候选视角；
  - agent 只负责状态推进，不再内联全部评分公式。
- 新增 `planning/object_search_policy.py`：
  - instruction mode 的 terminal target 与 anchor reference 选择从 mapper 中抽离；
  - anchor 仍只作为搜索参考物，不参与最终 stop。
- 新增 `planning/mode_policy.py`：
  - 集中管理 execution mode 比较，避免下划线模式字符串散落。
- 新增 `planning/room_policy.py`：
  - `--no_gpt_relocate` 时使用最近 frontier room 作为确定性 relocation 策略；
  - 该策略不读取目标类别或自然语言语义。
- 新增 `planning/exploration_policy.py`：
  - 从 mapper 中抽离 room/frontier node 选择逻辑；
  - 统一 `explore_in_room`、`explore_in_room_relocate`、
    `explore_after_check` 和 `explore_after_fully_explored`；
  - mapper 保留兼容 wrapper，策略层只选择下一探索 node。
- 新增 `mapping/room_segmenter.py`：
  - 从 mapper 中抽离 room segmentation 与 fallback_single_room；
  - 负责 2D obstacle histogram、free-space connected components、
    node-room assignment 和 frontier map labeling；
  - 当前仍写回 mapper runtime state，后续可继续收敛为
    `RoomSegmentationResult`。
- 新增 `mapping/topology_graph.py`：
  - 抽离 viewpoint graph / object-node graph 操作；
  - 承接 `add_node`、`visit_node`、`add_edge`、`remove_edge`、
    `update_obj`、`find_the_closest_path`、`check_connected` 等图操作。
- 新增 `mapping/map_serializer.py`：
  - 抽离 `to_json`、`to_json_wo_some_class`、`to_json_save_node_info`；
  - mapper 不再直接维护 prompt/debug JSON 构造。
- 新增 `mapping/frontier_extractor.py`：
  - 抽离 `get_nodes()` 中的 frontier debug 聚合、自适应局部半径、
    可见中心去重和 traversable candidate 追加逻辑；
  - 当前仍保留 DBSCAN、Open3D slicing、node 创建在 mapper 主流程中，
    以避免一次性迁移高耦合状态。
- 新增 `planning/target_selection_policy.py`：
  - 统一 benchmark target tag matching 与 instruction-mode object policy；
  - 返回 `TargetSelectionResult`，mapper 只负责兼容日志写入。
- 新增 `navigation/view_verification_controller.py`：
  - 抽离 `whether_to_check_again()` 中的 better-view proposal 选择；
  - agent 保留兼容 wrapper。
- 新增 `navigation/observation_pipeline.py`：
  - 抽离 panoramic observation point-cloud merge 与 debug PLY 保存；
  - 后续 `rotate_panoramic()` / `rotate_segmentation()` 将继续向该模块迁移。
- 新增 `navigation/action_controller.py`：
  - 集中 mapper-local waypoint 到 Habitat simulator waypoint 的坐标转换；
  - `step_mod()` 中主导航路径改为通过 action controller 获取 geodesic distance
    和 low-level planner action。
- 新增 `scripts/check_refactor.sh`：
  - 本地 py_compile 检查核心入口、adapter、planning 和 core 模块。
- 新增 `docs/refactor_architecture.md`：
  - 记录 Phase0-6 重构边界、模块输入输出、后续拆分路线。
- 新增 `prompting/`：
  - `templates.py` 集中维护 prompt 文本；
  - `schemas.py` 集中维护 LLM/VLM response schema；
  - `registry.py` 维护 prompt id、trace label、schema name 和模板版本。

### Changed

- `objnav_agent_with_process_obs.py` 的 `whether_to_check_again()` 改为调用
  `build_check_again_viewpoints()`，保留原 view-control 行为和日志输出。
- `mapper_with_process_obs.py` 的 instruction-mode 对象选择改为调用
  `InstructionObjectSearchPolicy`；普通 benchmark 没有 `InstructionPlan` 时仍走
  legacy tag 匹配路径。
- sequence/ordered 判断改为使用 `planning.mode_policy.is_ordered_execution()`。
- `make_plan_mod_no_relocate()` 新增 `use_gpt_relocate` 参数，统一 LLM /
  deterministic relocation 两条路径。
- 关闭 LLM relocation 时，mapper 改为调用
  `get_candidate_room_fully_explored_by_distance()`，日志写入 `room_policy/`，
  不再复用 `gpt_room/` 命名。
- instruction adapter、bbox VLM 复核和 mapper legacy room/object/relocate prompt
  改为从 `prompting` 层导入 prompt/schema/trace label，调用行为保持不变。
- `mapper_with_process_obs.py` 的 `segment_room()` / `_fallback_single_room()`
  改为委托 `mapping.room_segmenter`。
- `mapper_with_process_obs.py` 的 room/frontier exploration 方法改为委托
  `planning.exploration_policy`。
- `mapper_with_process_obs.py` 的 topology graph、map serialization、
  target selection 逻辑改为委托 `mapping.*` / `planning.*` 模块。
- `mapper_with_process_obs.py` 的 frontier 子步骤改为委托
  `mapping.frontier_extractor`，主流程仍负责 mapper 状态写入和 debug IO。
- `objnav_agent_with_process_obs.py` 的 check-again viewpoint 选择改为委托
  `navigation.view_verification_controller`。
- `objnav_agent_with_process_obs.py` 的 observation point-cloud helper 和
  `step_mod()` 主 action 获取路径改为委托 `navigation.*` 模块。

### Removed

- 删除 `whether_to_check_again()` 中确认无效的旧 depth debug 注释块。
- 删除重复的 `make_plan_mod_no_relocate_no_gpt()` 主流程。
- 删除旧的 `get_candidate_room_fully_explored_no_gpt()` 方法；等价行为迁移到
  `planning.room_policy.select_nearest_frontier_room()`。
- 清理 `mapper_with_process_obs.py` / `objnav_agent_with_process_obs.py` 中的大块
  历史注释代码：
  - 旧 `get_nodes_process()`；
  - 旧 `get_candidate_node()`；
  - 旧路径选择、旧 JSON 导出、旧可视化 debug 分支；
  - agent 中旧视频/debug 输出和旧 final waypoint 保存分支。

### Fixed

- `need_better_view` 且 `semantic_satisfied=true` 时不再把 candidate 写入
  rejected 列表，而是进入 `better_view_for_verified_pair` 执行状态。
- `InstructionObjectSearchPolicy` 在 pending verified pair 存在时优先返回该
  pair 的目标实例或同一局部几何簇，避免语义已确认后重新全图搜索新候选。
- `SemanticEdgeCache` / `RelationPairLedger` 增加单调保护：
  - VLM verified dynamic semantic edge 不会被后续 geometry reject 覆盖；
  - accepted relation pair 不会被 rejected relation pair 降级。
- `RuntimeConceptMatcher` 将高置信 LLM batch 结果缓存到
  instruction/concept/label 粒度，降低 mapper uid 漂移造成的重复 concept
  grounding 调用。
- 修复 `instruction_adapter/constraints.py` 中一处中文注释误改。

## 2026-06-09

### Added

- 新增统一 Concept grounding:
  - `ConceptQuery`
  - `RuntimeConceptMatcher`
  - `ConceptMatchRecord`
  - target 与 relation anchor 共用同一 prompt-first 概念匹配接口
  - `Constraint.object` 现在会生成 `object_concept`，不再只依赖精确字符串
- 新增 anchor-first relation search 支持：
  - LLM 决定 `execution.mode=anchor_first_relation_search`
  - anchor 只作为导航参考物，不参与最终 stop
  - `AnchorSearchLedger` 记录已搜索失败的 anchor 实例
  - anchor 失败只屏蔽该实例，不屏蔽整个类别/概念
- 新增 `instruction_adapter` 独立模块：
  - `InstructionPlan` / `TargetQuery` / `Constraint` canonical schema
  - `StriveInstructionSpec`
  - `StriveInstructionParser`
  - CogNav `episode.info` metadata compiler
  - CogNav 风格 LLM prompt parser
  - detector vocabulary grounding layer
  - structured prompt context renderer
- 新增动态语义边基础模块：
  - `DynamicSemanticEdgeVerifier`
  - `SemanticEdgeCache`
  - object-object relation geometry prefilter + VLM callback interface
- 新增原始指令终止验证模块：
  - `FinalInstructionVerifier`
  - `CandidateInstance`
  - `VerificationLedger`
  - `VerificationResult`
  - 实例级 hard rejection，避免错误候选反复触发停止
- 新增 Phase 2 运行时约束执行模块：
  - `InstructionExecutionState`
  - `TargetProgress`
  - `ConstraintStatus`
  - `ConstraintEvaluator`
  - 支持 count / any-success / sequence 的任务完成状态
- 新增 Phase 3 动态关系验证运行模块：
  - `InstructionSpatialGraph`
  - `ViewNode`
  - `ObjectNodeRecord`
  - `DynamicRelationService`
  - `VLMRelationVerifier`
  - object-view 共视索引、关系几何预筛、CogNav VLM 按需验证
- 新增 benchmark 参数：
  - `--custom_instruction`
  - `--enable_instruction_adapter`
  - `--instruction_adapter_backend`
  - `--instruction_adapter_strict_classes`
- 每个 episode 保存解析结果：
  - `logs/<save_dir>/episode-*/instruction_adapter/plan.json`
  - `logs/<save_dir>/episode-*/instruction_adapter/spec.json`
- final verifier 保存运行证据：
  - `logs/<save_dir>/episode-*/final_verifier/evidence_<step>.json`
  - `logs/<save_dir>/episode-*/final_verifier/result_<step>.json`
  - `logs/<save_dir>/episode-*/final_verifier/current_bbox_<step>.jpg`
  - `logs/<save_dir>/episode-*/final_verifier/object_crop_<step>.jpg`
- 新增 LLM/LVLM 原始返回追踪：
  - `logs/<save_dir>/episode-*/lvlm_calls/0001_<kind>.json`
  - 保存 `kind`、元数据和 `raw_response`
  - 不保存 prompt 图像 base64，避免日志膨胀
- 新增 `ask_gpt_object_in_box()` bbox crop 缓存：
  - `logs/<save_dir>/episode-*/detection/object_box_cache.json`
  - 同一 crop hash、bbox 和 detector class 列表不重复调用 LVLM
- instruction adapter 保存运行状态：
  - `logs/<save_dir>/episode-*/instruction_adapter/runtime_state_<step>.json`
  - 其中包含 `concept_matches`、`concept_matcher_stats`、`lvlm_call_counts`、`anchor_search_ledger`、`relation_pair_ledger`、`semantic_edges`
- `metrics.csv` 新增 instruction-level 字段：
  - `instruction_success`
  - `instruction_decision`
  - `instruction_accept_step`
  - `accepted_candidate_uid`
  - `accepted_relation_edge`
  - `lvlm_call_count_by_type`
  - `lvml_call_count_by_type`
- 新增中文设计文档：
  - `docs/instruction_adapter.md`

### Changed

- 指令解析从本地硬编码 ontology 改为 prompt-first 分层：
  - 优先复用 CogNav episode metadata。
  - metadata 不足时调用 LLM 输出结构化 plan。
  - `tv_monitor -> tv` 等类别差异移到 grounding 层处理。
- adapter 开启时，STRIVE 不再用 `ask_gpt_similar_objects()` 扩展目标同义词。
  - 目标检测词来自 `StriveInstructionSpec.target_detector_prompts`。
  - 新 grounding 层仅在精确匹配不足时可复用 legacy LLM similarity fallback。
- `mapper.object_found_no_gpt()` 从裸 `tag == target` 改为 normalized target alias 匹配。
- `mapper.object_found_no_gpt()` 会跳过已被 final verifier hard-rejected 的同一对象实例，但不会屏蔽同类别其它对象。
- `objnav_agent_with_process_obs.py` 在几何 `final_check()` 通过后新增原始指令满足度验证；只有 `decision=accept` 才允许最终 stop。
- final verifier 改为可插拔 `auto` 模式：普通 benchmark 没有 `InstructionPlan/InstructionSpec` 时直接旁路，不改变 STRIVE 原始停止逻辑。
- mapper 中增加 `instruction_plan` / `instruction_spec` / `target_aliases` 字段，但不改变 STRIVE 的 room/viewpoint/path planner。
- 启用 instruction adapter 时，`mapper.object_found_no_gpt()` 在 sequence 模式只暴露 active target，防止后续子目标提前触发 stop。
- `objnav_agent_with_process_obs.py` 在 final verifier 前调用 `ConstraintEvaluator`：
  - attribute/room 作为原始指令 verifier 的显式证据；
  - relation 走 geometry prefilter + VLM dynamic semantic edge；
  - final accept 后由 execution state 判断 count/sequence 是否真正完成。
- `mapper.object_found_no_gpt()` 在 instruction mode 中改为 runtime concept matching：
  - terminal target 由 `RuntimeConceptMatcher` 判断；
  - anchor-first 模式下，没有 terminal candidate 时可导航到未搜索 anchor；
  - anchor reference 到达后写入 `AnchorSearchLedger` 并继续搜索，不能作为成功。
- relation constraint anchor 匹配改为 concept-instance match：
  - 优先使用 `constraints[*].object_concept`；
  - runtime mapper label 不需要和 instruction 文本完全相同。
- 指令模式下 `check_again()` 改为 evidence-only：
  - 普通 benchmark 仍保留原 `check_again_object_in_bbox()` 类别复核；
  - instruction mode 只采集 bbox 图、几何和附近对象证据；
  - `FinalInstructionVerifier` 统一判断 candidate、relation、instruction satisfaction 和 view sufficiency。
- `FinalInstructionVerifier` 输出增加视角反馈字段：
  - `semantic_satisfied`
  - `view_sufficient_for_stop`
  - `view_feedback`
  - `preferred_view_goal`
  - `view_objective`
  - 只有语义和视角质量同时满足才允许 stop。
- 新增 `ViewControlState`：
  - `need_better_view` 后不再只做一次 retry；
  - 保存 baseline view quality、多个 viewpoint proposals、attempt history；
  - final verifier 会看到 `view_control` 历史；
  - 控制层只执行 VLM 给出的 `view_objective`，不再用几何改善阈值覆盖 VLM 的
    accept/retry 决策。
  - better-view 子目标会 pin 已验证的 `DynamicSemanticEdge`，避免靠近后
    mapper 实例 uid 漂移导致重新验证其它 pair。
- relation constraint 在 view-control active 时优先复用 `pinned_relation_context`。
- `check_again` 强视觉证据下，关系几何预筛失败可降级为 VLM override，并绕过旧的
  geometry failure cache。
- `FinalInstructionVerifier` 新增 prompt-side view guidance：
  - 使用 `view_quality_facts` 中的投影、中心偏移、边界余量、bbox 面积和距离；
  - 这些事实只进入 prompt 和 diagnostics，不再由 Python 侧强制转为
    `need_better_view`；
  - 不写目标/anchor 类别规则。
- `whether_to_check_again()` 从旧的早停阈值改为通用候选视角评分：
  - visibility；
  - centerability；
  - border margin；
  - projected area；
  - distance suitability。
  该评分只负责生成更清晰证据，最终语义仍由 final verifier 判断。
- 新增 `RelationPairLedger`：
  - 关系失败只拒绝 `(terminal_uid, relation, anchor_uid)`；
  - 关系成功写入 `SemanticEdge`，后续按动态语义边复用。
- `RuntimeConceptMatcher` 新增 `match_many()` 批量接口：
  - exact/cache 先行；
  - 非精确候选合并成一次 prompt-first grounding；
  - 小 crop 会自动 resize，避免 VLM provider 因尺寸过小拒绝请求。
- `RuntimeConceptMatcher` 记录真实调用统计：
  - `batch_llm_calls` 是实际批量请求次数；
  - `batch_items_requested` 是批量请求覆盖的候选 object 数量；
  - `cache_hits` 和 `exact_matches` 用于区分“模型没被调用”和“模型被调用但批量处理”。
- CogNav / Gemini / OpenAI compatible `parse()` 统一经过 trace wrapper，未显式标注的调用会退化为 `parse`，STRIVE 关键调用已补充语义化 `trace_label`。
- `semantic_edges` 几何预筛改为使用 STRIVE 点云 z 轴作为垂直方向，并优先使用点云边界判断 `on / inside / under`。
- CogNav episode metadata compiler 增强为 best-effort 读取 `complex_constraints` 中的 relations / attributes / room constraints。

### Boundary

- `support_objects` 只进入 room/search prompt context。
- `support_objects` 不进入目标检测停止链。
- parser 不保存“目标常见房间/常见支持物”硬编码表。
- 名称泛化不写 `shelf -> bookshelf`、`book -> books` 这类代码规则；所有非精确匹配交给 ConceptQuery grounding 与 RuntimeConceptMatcher，并记录到日志。
- 动态语义边只提供 verifier/cache 接口，具体共视角提取和 VLM 调用由运行时接入。
- final verifier 不写“红色椅子/TV/客厅”等目标常识规则，只把原始指令、plan、候选实例和证据包交给 VLM 判断。
- 关系约束失败默认拒绝对象对/关系边，不拒绝 detector 类别；属性失败才拒绝具体对象实例。

### Fixed

- 修复 execution mode 判断使用 `normalize_term()` 导致下划线模式失效的问题。
  - `anchor_first_relation_search` 现在会正确进入 anchor-first 分支。
  - `all_targets_success` 等计数/多目标模式也统一使用 `compact_key()` 比较。
