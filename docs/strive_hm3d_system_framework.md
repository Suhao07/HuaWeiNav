# STRIVE HM3D Baseline 框架文档

本文档面向代码维护和二次开发，说明本项目 HM3D ObjectNav baseline 的主要模块、输入输出、数据流、核心模型和关键公式。

## 1. 总体目标

STRIVE HM3D baseline 解决的是 Habitat HM3D ObjectNav 任务：

```text
给定目标类别，例如 toilet，智能体在未知室内环境中通过 RGB-D 观测、目标检测、分割、三维建图、前沿探索和 LLM 决策，导航到目标附近并执行 stop。
```

当前 Docker 路径复用 CogNav_ObjNav：

- Habitat/HM3D 数据。
- CogNav LLMClient。
- 本地或 CogNav 权重目录中的视觉模型权重。

## 2. 主要入口

### 2.1 benchmark 入口

文件：

```text
objnav_benchmark_with_process_obs.py
```

职责：

- 解析命令行参数。
- 构建 Habitat config。
- 创建 Habitat 环境。
- 创建 mapper 和 agent。
- 逐 episode reset。
- 调用 LLM 获取目标相关类别。
- 执行导航循环。
- 保存视频、点云、metrics。

关键输入：

```text
--eval_episodes: 运行到哪个 episode index。
--start_episode: 从哪个 episode index 开始。
--save_dir: 保存到 logs/<save_dir>。
--vlm: LLM/VLM 后端，当前推荐 cognav。
```

关键输出：

```text
logs/<save_dir>/metrics.csv
logs/<save_dir>/episode-<i>/fps.mp4
logs/<save_dir>/episode-<i>/depth.mp4
logs/<save_dir>/episode-<i>/metrics.mp4
logs/<save_dir>/episode-<i>/pcd/
```

### 2.2 Docker 入口

文件：

```text
docker/run_hm3d_baseline.sh
```

职责：

- 查找或下载 SAM/GroundingDINO 权重。
- 校验 CogNav HM3D 数据。
- 挂载 STRIVE 仓库、CogNav 仓库、HuggingFace cache、权重文件。
- 透传 LLM、AMap、Habitat 和权重环境变量。
- 在容器内启动 `objnav_benchmark_with_process_obs.py`。

## 3. 模块结构

```text
docker/
  Dockerfile                         # 在 CogNav 基础镜像上补 STRIVE 依赖
  build.sh                           # 构建 strive-hm3d:local
  run_hm3d_baseline.sh               # 运行 benchmark
  preflight.py / preflight.sh        # 运行前检查

config_utils.py                      # Habitat/HM3D/MP3D config 适配
constants.py                         # LLM key 和模型名基础常量
objnav_benchmark_with_process_obs.py # benchmark 主循环
objnav_agent_with_process_obs.py     # 导航 agent、规划、轨迹保存
mapper_with_process_obs.py           # 语义建图、节点/房间/前沿管理
mapping_utils/transform.py           # 相机内参、坐标变换
cv_utils/
  gpt_utils.py                       # LLM 问答工具函数
  sam.py                             # GroundingDINO + SAM 目标感知封装
llm_utils/
  cognav_llm_adapter.py              # CogNav LLMClient 适配 OpenAI parse 接口
instruction_adapter/
  contracts.py                       # InstructionPlan / ConceptQuery / TargetQuery / Constraint schema
  grounding.py                       # 编译期 target/anchor concept grounding
  concept_matcher.py                 # 运行时 concept-instance matcher 与 anchor ledger
  execution.py                       # count / any-success / sequence 执行状态
  constraints.py                     # room / attribute / relation 运行时约束评估
  spatial_graph.py                   # object-view 共视索引
  semantic_edges.py                  # 动态语义边几何预筛和缓存
  relation_verifier.py               # CogNav VLM relation verifier
  verifier.py                        # 原始指令 final verifier 和实例级 ledger
```

## 4. 端到端数据流

### 4.1 初始化阶段

```text
docker/run_hm3d_baseline.sh
  -> 挂载 /workspace/STRIVE
  -> 挂载 /workspace/CogNav_ObjNav
  -> 设置 HM3D_DATA_PATH、SAM_CHECKPOINT、GROUNDING_DINO_CHECKPOINT
  -> python objnav_benchmark_with_process_obs.py
```

### 4.2 每个 episode 的数据流

```text
Habitat episode
  -> env.reset()
  -> observation = {rgb, depth, semantic/goal info}
  -> HM3D_Objnav_Mapper 初始化地图
  -> HM3D_Objnav_Agent 初始化当前位置、轨迹、目标
```

### 4.3 每个导航周期的数据流

```text
RGB-D observation
  -> GroundingDINO 根据文本类别产生 2D boxes
  -> SAM 根据 boxes 产生 masks
  -> depth + camera intrinsic 反投影为局部 3D 点
  -> agent pose 将局部点变换到世界/地图坐标
  -> 将检测类别按 target_list 归一到任务目标，例如 tv_monitor -> tv
  -> mapper 更新点云、障碍、对象实例、节点、房间、前沿
  -> LLM 根据目标、房间、对象、距离选择候选房间/目标
  -> planner 计算下一动作
  -> env.step(action)
```

### 4.4 结束阶段

```text
found_goal 或 episode_over 或 step >= 500
  -> save_trajectory()
  -> 保存 rgb/depth/topdown mp4
  -> 保存点云 debug 文件
  -> 写入 metrics.csv
```

`found_goal` 不是“图像里出现了目标”这么简单。流程会先把 2D 检测结果变成 3D 对象，再用 `target_list` 做目标别名匹配，随后寻找可达停止点并做可视性检查。举例：任务目标是 `tv` 时，视觉模型可能输出 `tv_monitor`，需要归一成 `tv` 后才能触发目标确认。

启用 instruction adapter 后，终止链路增加一层原始指令验证：

```text
mapper.object_found_no_gpt()
  -> instance-level ledger 过滤已拒绝候选
  -> ConstraintEvaluator 检查 room/attribute/relation/count/sequence 运行状态
  -> agent.check_again() 优化视角复核 detector 目标
  -> FinalInstructionVerifier 判断原始自然语言是否满足
  -> InstructionExecutionState 判断 count/sequence 是否完成
  -> stop 或继续探索
```

普通 HM3D benchmark 不启用 `InstructionPlan` 时，这条链路完全旁路，保持原始 STRIVE 行为。

## 5. 核心输入输出

### 5.1 Habitat observation

输入来源：

```text
habitat.Env.reset()
habitat.Env.step(action)
```

常用字段：

```text
obs["rgb"]   : H x W x 3, uint8
obs["depth"] : H x W x 1, float32
```

Agent 内部使用：

```text
self.obs
self.position
self.rotation
self.metrics
```

### 5.2 视觉感知输出

GroundingDINO 输入：

```text
image: RGB panorama 或当前视角
text prompts: target list / object classes
```

GroundingDINO 输出：

```text
boxes: N x 4, 每个框为 [x1, y1, x2, y2]
scores: N
labels: N
```

SAM 输入：

```text
image
boxes
```

SAM 输出：

```text
masks: N x H x W
```

### 5.3 Mapper 输出

Mapper 维护：

```text
节点 nodes: 探索候选点和已访问点
房间 rooms: 由节点和空间聚类得到的 room-level 表示
对象 objects: 由检测、分割和点云融合得到的对象实例
前沿 frontiers: 未探索区域边界
点云 pcd: 障碍、对象、可通行区域等三维信息
```

关键输出给 agent：

```text
candidate nodes
candidate rooms
object candidates
final waypoint
```

### 5.4 LLM 输出

通过 `llm_utils/cognav_llm_adapter.py` 适配后，LLM 输出被解析为 Pydantic 结构体。

典型输出：

```text
similar object list
object_found flag
candidate room id
reason / explanation
```

若启用：

```bash
STRIVE_LLM_FALLBACK=1
```

当 LLM 返回空内容或非 JSON 时，会返回保守默认结构，只用于 smoke 测试。

### 5.5 指令约束输出

启用 `--enable_instruction_adapter` 或 `--custom_instruction` 后，每个 episode 会额外输出：

```text
instruction_adapter/plan.json          # canonical InstructionPlan
instruction_adapter/spec.json          # STRIVE legacy spec
instruction_adapter/runtime_state_*.json
final_verifier/evidence_*.json
final_verifier/result_*.json
```

`runtime_state_*.json` 中包含：

```text
execution_state: active target、已接受实例、已拒绝实例、count/sequence 进度
spatial_graph: ObjectNodeRecord / ViewNode / 共视索引
semantic_edges: on/near/inside/under 等动态关系验证缓存
```

## 6. 核心几何和模型公式

### 6.1 相机内参

给定图像宽高 `W, H` 和水平视场角 `hfov`：

```text
fx = W / (2 * tan(hfov / 2))
fy = fx
cx = W / 2
cy = H / 2
```

相机内参矩阵：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

代码位置：

```text
mapping_utils/transform.py::habitat_camera_intrinsic
```

### 6.2 RGB-D 反投影

对像素点 `(u, v)` 和深度 `z`：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = z
```

齐次形式：

```text
p_camera = z * K^-1 * [u, v, 1]^T
```

### 6.3 相机坐标到世界坐标

给定 agent 位姿旋转 `R` 和平移 `t`：

```text
p_world = R * p_camera + t
```

对于点云集合：

```text
P_world = R * P_camera + t
```

这些点进入 mapper 后被聚合到对象点云、障碍点云和可通行区域中。

### 6.4 节点距离和路径选择

节点间欧氏距离：

```text
d(p_i, p_j) = ||p_i - p_j||_2
```

候选前沿/房间的代价通常由以下因素共同决定：

```text
cost = travel_distance + semantic_penalty - target_likelihood_bonus
```

当前代码中该代价不是单一显式函数，而是由 mapper 计算候选、agent 计算路径距离、LLM 综合语义和距离后选择。

### 6.5 成功判定

Habitat ObjectNav 的成功通常由 stop 动作和目标距离共同决定：

```text
success = 1 if action == STOP and distance_to_goal <= success_distance else 0
```

当前配置中：

```text
success_distance = 1.0
max_episode_steps = 500
```

## 7. LLM 决策接口

STRIVE 原始代码使用 OpenAI 风格：

```python
client.beta.chat.completions.parse(...)
```

本项目通过适配器把 CogNav LLMClient 包装成相同接口：

```text
llm_utils/cognav_llm_adapter.py
```

适配流程：

```text
Pydantic response_format
  -> 生成 JSON schema prompt
  -> 调用 CogNav LLMClient.chat_completion()
  -> 提取 JSON object
  -> response_format.model_validate()
  -> 返回 OpenAI-compatible parsed response
```

这样 `cv_utils/gpt_utils.py`、`mapper_with_process_obs.py` 等调用方无需直接依赖 CogNav LLMClient 的具体实现。

## 8. 规划和动作

动作来自 Habitat ObjectNav action space，常见为：

```text
0: STOP
1: MOVE_FORWARD
2: TURN_LEFT
3: TURN_RIGHT
```

Agent 的规划逻辑：

```text
当前位姿 + mapper 目标 waypoint
  -> pathfinder / PID planner
  -> next action
  -> env.step(action)
```

核心代码：

```text
objnav_agent_with_process_obs.py::make_plan_mod_no_relocate
objnav_agent_with_process_obs.py::step_mod
```

## 9. 日志和可视化

每个 episode 输出：

```text
fps.mp4      # RGB 第一视角
depth.mp4    # 深度图
metrics.mp4  # top-down map + 指标
```

`save_trajectory()` 会把不同阶段产生的 frame 统一 resize 到首帧尺寸，避免 `imageio` 写 mp4 时因为尺寸不同失败。

指令模式还会输出可审计的结构化日志：

```text
instruction_adapter/plan.json
instruction_adapter/spec.json
instruction_adapter/runtime_state_<step>.json
final_verifier/evidence_<step>.json
final_verifier/result_<step>.json
lvlm_calls/0001_<kind>.json
detection/object_box_cache.json
```

其中 `lvlm_calls/*.json` 保存每次真实 LLM/LVLM 调用的 `kind`、元数据和
`raw_response`，不保存 prompt 图像 base64。`object_box_cache.json` 缓存同一
bbox crop 的类别复核结果，避免重复调用视觉模型。

## 10. 终止验证数据流

STRIVE 的最终停止现在由三层共同决定：

```text
候选目标实例
  -> check_again: 普通 benchmark 做 bbox 类别复核；指令模式只采集 bbox evidence
  -> instruction mode 下立即复用该图做约束、原始指令和视角质量验证
  -> final_check: 几何可见性检查（未提前完成时）
  -> FinalInstructionVerifier: 原始自然语言指令满足度 + view feedback
```

`FinalInstructionVerifier` 复用 CogNav 风格 LLM client。它的输入是：

```text
raw_instruction
InstructionPlan
CandidateInstance
当前视角 bbox 图
目标 crop 图
bbox 几何事实
view_quality_facts：中心偏移、边界余量、bbox 面积、距离、投影状态
附近对象摘要
预留空间关系证据
```

对于带空间关系的指令，`ConstraintEvaluator` 会在 final verifier 前先验证
动态语义边：

```text
candidate object record
anchor object record
check_again/current bbox image
  -> DynamicRelationService
  -> SemanticEdge(subject_uid, relation, object_uid)
```

成功边写入 `semantic_edges`；失败 pair 写入 `relation_pair_ledger`。失败粒度
是对象对，不是目标类别或 anchor 概念。

输出是结构化 `VerificationResult`：

```text
accept: 可以 stop
reject_candidate: 当前实例不是目标，写入 VerificationLedger 并继续探索
need_better_view: 语义可满足但当前证据视角不足，复用 STRIVE 视角优化再验证一次
need_relation_check: 预留给动态语义边 verifier
uncertain: 不终止，按 soft rejection 继续
```

指令模式的停止条件是双条件：

```text
semantic_satisfied == true
view_sufficient_for_stop == true
```

`view_sufficient_for_stop` 先由 VLM 根据 `bbox_center_norm`、
`bbox_area_ratio`、`visible_projected_points`、距离和关系证据做软判断；随后
`FinalInstructionVerifier` 会执行一层可配置的通用 view guard：

```text
projection must exist
center_offset_norm should not be too large
border_margin_norm should not be too small
bbox_area_ratio should not be too small
```

这些是相机停止证据的几何硬约束，不是目标语义规则。默认阈值可通过
`STRIVE_FINAL_VIEW_MAX_CENTER_OFFSET`、`STRIVE_FINAL_VIEW_MIN_BORDER_MARGIN`、
`STRIVE_FINAL_VIEW_MIN_BBOX_AREA` 调整，也可用 `STRIVE_FINAL_VIEW_GUARD=0`
关闭。

如果 verifier 返回 `need_better_view`，agent 进入 `ViewControlState`。这不是
物体规则状态机，而是围绕 verifier 给出的 `view_objective` 执行通用视角控制：

```text
ViewObjective:
  keep_visible_roles
  improve_goals
  minimum_expected_improvement
  accept_if_no_better_view
```

`whether_to_check_again()` 不写“book/shelf/chair”等规则，而是在到候选实例的可达路径上生成多个候选姿态，并用通用几何质量排序：

```text
score = visibility + centerability + border margin + projected area + distance suitability
```

`ViewControlState` 记录 baseline、已尝试 proposal、观测质量和剩余候选。若
verifier 想 accept，但当前 evidence 相对 baseline 没有达到
`STRIVE_VIEW_CONTROL_MIN_IMPROVEMENT` 且还有候选未尝试，控制层会继续
`need_better_view`。这只约束执行闭环，最终是否满足原始指令仍由 relation
verifier 和 FinalInstructionVerifier 决定。

为避免 VLM 在弱远景上第一次就过度自信 `accept`，instruction mode 的首次
accept 也会经过 initial-accept deferral：如果通用几何 proposal 预测可得到明显更好的
final evidence，agent 会先执行该 better-view 子目标；如果没有明显改善空间，则保留
verifier 的 accept。

better-view 子目标会 pin 已验证的 `DynamicSemanticEdge`。例如
`book_uid on cabinet_uid` 已经通过后，后续视角优化只重新评估视角质量，不重新把关系发现
从零开始。若 mapper 在近距离把同一书架区域切成新的 book uid，约束层会通过
`pinned_relation_context` 把它视为同一已接受语义区域的证据更新。

对 `check_again` 产生的强视觉证据，关系几何预筛失败不再直接 hard reject；系统会允许
VLM relation verifier 覆盖几何不确定性，并在缓存层绕过旧的 geometry failure。

关键原则是“屏蔽实例，不屏蔽类别”。例如找红色椅子时，蓝色椅子被 verifier 拒绝后，只跳过该椅子实例；检测器后续发现其它 `chair` 实例仍会继续验证。

### Concept Grounding And Anchor-First

指令模式下，STRIVE 现在维护三类运行时状态：

```text
ConceptQuery:
  编译期 LLM/VLM grounding 的指令概念。terminal target 和 relation anchor
  使用同一 schema，但 terminal 标志不同。

ConceptMatchRecord:
  RuntimeConceptMatcher 判断 mapper object 是否满足某个 ConceptQuery。
  非精确匹配由 LLM/VLM 决定，并按 instruction/concept/object 缓存。
  多对象场景使用 `match_many()` 批量询问，避免每个 object 单独调用 LVLM。

AnchorSearchLedger:
  anchor-first 模式的参考物搜索记忆。某个 anchor 附近未找到 terminal
  target 时，只屏蔽这个 anchor uid，不屏蔽整个 anchor concept。

RelationPairLedger:
  空间关系验证记忆。某个 terminal-anchor pair 失败后，只跳过该 pair；
  其它 terminal 实例和其它 anchor 实例仍可继续组合验证。

ViewControlState:
  final verifier 请求 better view 后的通用视角控制记忆。它保存
  view objective、baseline quality、proposal attempts 和 observed quality，
  防止“一步 retry 后 VLM 直接 accept”导致提前停止。
```

`anchor_first_relation_search` 的数据流：

```text
InstructionPlan(book on shelf)
  -> terminal concept: book
  -> anchor concept: shelf
  -> RuntimeConceptMatcher(book/books...)
  -> 如果 terminal 未出现，RuntimeConceptMatcher(shelf/bookshelf/cabinet...)
  -> 导航到 anchor reference
  -> anchor 到达后记录 AnchorSearchLedger
  -> 后续 terminal candidate + anchor instance 进入 DynamicRelationService
  -> relation verified 后才允许 FinalInstructionVerifier accept
```

anchor reference 不会进入最终成功链；agent 在 final verifier 前会识别
`_instruction_reference_role=anchor`，写入 ledger 后继续探索。

日志位置：

```text
logs/<save_dir>/episode-*/final_verifier/
  evidence_<step>.json
  result_<step>.json
  current_bbox_<step>.jpg
  object_crop_<step>.jpg
```

开关：

```bash
export STRIVE_FINAL_VERIFIER=auto  # 默认：仅在指令模式存在 InstructionPlan/InstructionSpec 时介入
export STRIVE_FINAL_VERIFIER=1     # 强制开启，要求调用侧传入 instruction plan
export STRIVE_FINAL_VERIFIER=0     # 关闭，恢复旧停止行为
```

普通 benchmark 模式不传 `--enable_instruction_adapter` / `--custom_instruction` 时没有 `InstructionPlan`，final verifier 在 agent 层直接旁路，不额外调用 VLM，也不改变原始 STRIVE 停止条件。

### Instruction-Level Metrics

`metrics.csv` 保留 Habitat 官方字段，例如：

```text
success
spl
distance to goal
steps
```

这些字段仍按 HM3D ObjectNav 原始目标类别和距离阈值计算。自然语言指令模式额外写入：

```text
instruction_success
instruction_decision
instruction_accept_step
accepted_candidate_uid
accepted_relation_edge
lvlm_call_count_by_type
lvml_call_count_by_type
```

复杂指令实验需要同时看两类指标：

```text
success:
  Habitat 原始 ObjectNav 成功，适合和 baseline 对齐。

instruction_success:
  原始自然语言指令、实例约束、动态语义边和 final verifier 的执行结果。
```

例如 `book on shelf` 在指令链路中可以 `instruction_success=True`，但 Habitat
原始 `success` 仍可能是 0，因为官方 episode 只知道单一 object goal，不知道
“on shelf” 这类动态语义关系。

`lvlm_call_count_by_type` 是 JSON 字符串，包含：

```text
calls:
  instruction_parser / concept_grounding / concept_match_batch /
  relation_verifier / final_instruction_verifier / room_selection / ...
cache_hits:
  bbox_object_in_box / ...
total_calls
total_cache_hits
```

`lvml_call_count_by_type` 是早期文档拼写的兼容别名，内容与
`lvlm_call_count_by_type` 相同。

`instruction_adapter/runtime_state_<step>.json` 还会记录
`concept_matcher_stats`，其中 `batch_llm_calls` 是真实批量请求次数，
`batch_items_requested` 是这些请求覆盖的候选 object 数量。

## 11. 可替换模块

### 11.1 替换 LLM

入口：

```text
llm_utils/cognav_llm_adapter.py::get_client_and_model
cv_utils/gpt_utils.py
```

推荐保持返回接口兼容：

```python
client.beta.chat.completions.parse(...)
```

### 11.2 替换检测/分割模型

入口：

```text
cv_utils/sam.py
config_utils.py 中的 GROUNDING_DINO_* 和 SAM_CHECKPOINT
```

替换模型时需要保证输出仍能转换为：

```text
boxes
masks
labels
scores
```

### 11.3 替换数据集

入口：

```text
config_utils.py::hm3d_config
config_utils.py::mp3d_config
```

需要对齐：

```text
scene_datasets
episode json.gz
success_distance
sensor config
measurements
```

## 12. 当前已验证路径

已验证：

```text
Docker image: strive-hm3d:local
Data root: CogNav_ObjNav/data
LLM client: CogNav LLMClient + Ark provider
Vision: MMDetection GroundingDINO Swin-L + SAM ViT-H
Benchmark: HM3D ObjectNav val episode 0
```

真实 LLM 测试结果：

```text
success: 1.0
Found Goal: True
Episode Steps: 128
输出目录: logs/hm3d_cognav_real_llm_smoke/
```
