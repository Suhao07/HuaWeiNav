# STRIVE 重构后代码框架

本文记录 2026-06-10 开始的 Phase0-4 重构边界。目标不是一次性重写
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
```

这个脚本是后续重构的最低门槛。任何继续拆分 agent / mapper 的提交，都应先
通过它，再考虑 Docker smoke 或完整 benchmark。

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
  mode_policy.py
  object_search_policy.py
  room_policy.py
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

## Phase3：运行时边界

当前运行时边界如下：

```text
objnav_benchmark_with_process_obs.py
  -> episode/config/metrics orchestration

mapper_with_process_obs.py
  -> mapping state, object/node/frontier management
  -> benchmark-compatible target search branch
  -> instruction policy invocation and debug artifact writing

objnav_agent_with_process_obs.py
  -> Habitat action loop
  -> final verifier orchestration
  -> view-control state transition

instruction_adapter/
  -> prompt-first instruction plan
  -> concept grounding
  -> relation / final verifier
  -> ledgers and dynamic semantic edges

navigation_core/
  -> pure math and geometry helpers

planning/
  -> object/search/room/viewpoint policy wrappers
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

## Phase4：删除废代码

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
  4084 lines -> 3327 lines
  5 行以上注释代码块清零

objnav_agent_with_process_obs.py:
  2191 lines -> 2148 lines
  5 行以上注释代码块清零
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
