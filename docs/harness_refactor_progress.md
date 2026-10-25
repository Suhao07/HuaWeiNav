# STRIVE Harness 重构执行记录

## 范围与基线

用户已授权实施 P0–P7 全部计划，并完成同一个真实 HM3D episode 的前后对比。
不能将只完成指令拆分、离线单测或启动进程作为整体完成。

- 基线：`cf8dd8ec9bfd7bb6f76fbbead084fc3cada04337`，开始时工作树干净。
- 完整需求：`/home/ubuntu/.codex/attachments/58829ad5-5482-4026-b8f0-0c3ed4df2268/pasted-text-1.txt`。
- 基线归档及结果目录：`outputs/harness_refactor_20261023/baseline/`。
- `source.tar` 和 `source/` 保存原始受版本管理的代码；不随重构变动。
- 基线运行配置：`baseline/run-config.json`，输出 `baseline/episode.log`。
- 本记录不含凭证；使用用户先前指定的凭证文件，不将内容写入报告。

## 当前完成的具体修改

1. `instruction/contracts.py` 提供深层不可变 TaskSpec、TargetSpec、ConstraintSpec 和显式 CompileResult。
2. 语言解析迁到 `instruction/parser.py`；严格入口不会在不可用时返回 dataset 替代目标。
3. `instruction/compiler.py` 只编译语义，检查歧义及不支持的硬条件，不调用检测词映射或搜索策略。
4. dataset metadata 解析迁到 `backends/habitat/task_input.py`；简单 ObjectNav 任务可确定性生成 TaskSpec。
5. 检测词映射迁到 `perception/detector_adapter.py`；移除了策略选择模型调用和清除 ordered/exhaustive 的行为。
6. 搜索策略判定仍在 planning；anchor-first 不修改用户任务要求。
7. 执行进度不再写回 plan，也不再在每次 bind 时从 plan 重置；拒绝未来阶段和空 UID 的接受事件。
8. Habitat 的普通 ObjectNav 安装路径已使用显式类别适配，并保存 task_spec.json；不再为已知类别调用语言解析。
9. scene/contracts.py 成为共享场景类型的唯一定义；real_robot/contracts.py 暂时重导出这些类型以迁移调用方。
10. instruction_adapter 的旧解析/grounding 模块仅薄转发；包初始化改为懒加载，避免读取合同就导入整个感知/验证图。

`instruction_adapter/task_compat.py` 是 TaskSpec 到旧 runtime plan 的单向、可丢弃投影；任务状态迁移后应删除。
当前 `scene/query.py` 已直接查询原生快照，`TaskSession` 持有任务服务；旧 SnapshotConfidence/ObjectAdapter 和旧对象选择器已删除。查询仍使用旧 plan 投影，P2 的合同迁移尚待收口。

## 后续实现进展

- Habitat 通过 `backends/habitat/scene.py` 导出观测快照，通过共享查询选择候选；地图标签与任务匹配结果分开。
- `task/reducer.py` 接管最终验证结果的任务推进，Habitat 与 ROS 使用同一入口；已删除 ConstraintEvaluator 中的进度更新方法。
- 验证器明确关闭或依赖不可用时产生 unknown；模型调用异常保留原始异常与调用栈，交由运行时处置。
- 接受结果需要明确的语义及视点确认；关系失败保留对象对作用域，证据不足保持任务进度。
- 当前全套宿主测试：262 passed；编译检查和 git diff --check 通过。
- 基线 episode 的 exit.json 已确认 returncode=0。重构后 episode 尚待完整主链路收口后运行。

### 当前边界

进度现由 `task/state.py:TaskProgress` 持有，使用不可变 TaskSpec 的数量、顺序与完成模式；旧 `instruction_adapter/execution.py` 已删除。TaskSession 的显式 `spec/progress` 字段替代旧动态字段。验证结果携带 task_revision，reducer 与 ROS 回调按当前版本准入；同版本查询保留进度，新版本安装创建独立进度与索引。对象、图像和坐标依赖尚待接入统一证据协议，P3–P7 保持未完成。

本次新增版本隔离、合同计数独立于运行时 plan、任务修订与重复安装测试；全套宿主测试 266 passed。运动与 Harness 主链路仍待实施，重构后 episode 尚未运行。

### 共享验证协议与 ROS 完成规则

- `verification/contracts.py` 承载 CandidateInstance、VerificationResult 和 VerificationRecord；task reducer 直接依赖值协议。
- `verification/store.py:EvidenceStore` 替代旧 VerificationLedger 实现。索引包含任务版本、谓词、目标和 anchor，保留追加历史；三值 verdict 分别表达确认、反证和证据不足。
- 最终验证结果记录目标 UID、实际发送的图像引用及输入 dependencies；reducer 和 ROS 回调核对任务版本与目标身份。
- 修复 ROS 在单个目标 accept 后直接 STOP 的路径：运行时还读取任务所有者的 `task_complete()`。数量与顺序由 reducer 判断；WAIT/smoke policy 明确保持任务未完成。
- 新增旧版本索引、对象对隔离、三值语义与任务未完成时 STOP 抑制测试。全套宿主测试 **270 passed**，编译检查与 diff 检查通过。

当前 dependencies 已进入结果和存储协议，图像内容/对象几何/坐标版本的生产端绑定及统一有效性检查仍待实现；不能据此声明完整证据失效机制已完成。

### ROS 运动所有权与运行时拆分

- 新增 `motion/supervisor.py`，从 RuntimeState 提取活动目标、已完成目标和 HOLD 所有权。运行时与 ViewpointEvidenceLoop 共用同一个 supervisor；当前两条 ROS 执行路径通过它提交和查询运动。
- request_id 重试返回原 goal_id；已有运动占有执行槽位时拒绝第二次提交。取消保留占有权，直至控制器返回终态。
- 提交/查询异常原样上抛并记录 unknown；提交响应丢失后通过 backend-confirmed goal ID 对账。HOLD 与恢复更新 control_epoch。
- `backends/ros/scene.py` 承载 SysNav 快照桥接，`backends/ros/smoke.py` 承载记录控制器与 smoke policy，`harness/trace.py` 承载 JSONL 和序列化。real_robot 包按需加载公开符号，消除后端直接导入时的环依赖。
- 新增运动占有、幂等重试、取消确认等待、未知提交对账和 HOLD/epoch 测试。全套宿主测试 **273 passed**；编译与 diff 检查通过。

P4 仍待：完整任务 revision 的运动准入、物理停止确认语义核对。Habitat 尚未接入该运动所有者。当前终态依据现有控制器协议，新增测试是软件状态证据，不是实机停止证明。

### 实时 ROS 异步推理

- ROS instruction_runtime_node 显式注入单线程 ThreadPoolExecutor；策略决策与到达后验证在 worker 执行，timer 查询 Future 后返回 THINKING/VERIFYING。测试和确定性回放保留显式的 inline 执行配置。
- 验证结果回到 timer 后才进入 policy 回调及 task reducer；worker 异常通过 Future.result 原样传播。
- 异步结果绑定输入修订序号与 control_epoch。输入从 A 改为 B 再改回 A 仍产生新序号；HOLD/恢复后丢弃旧提案。
- `scene/dependencies.py` 对相关目标和 anchor 的标签、几何、房间及坐标系版本生成依赖指纹；决策和验证返回时检查当前快照。无关对象、置信度和机器人位姿刷新保持依赖稳定。
- 新增真实线程阻塞测试及可控 Future 测试，覆盖 timer 响应、迟到控制提案、异步验证、目标几何变化、输入修订和依赖范围。日志：`outputs/harness_refactor_20261023/async-tests.log`。

边界：frame_revision 目前由后端可选提供，当前代码尚未证明各实际定位后端都能报告坐标系重置。图像内容绑定、Habitat 对应准入、完整 Tools/Skills 和最终 episode 对比仍待完成。

### 图像内容绑定

- `verification/images.py` 将实际发送的图片字节按 SHA-256 归档；模型请求与归档使用同一次读取，图片 MIME 根据受支持的文件类型生成。
- 最终验证结果保存归档引用、内容指纹和原始路径对应关系。源路径被复用时仍可回放原请求；归档丢失或改变后，reducer、ROS 完成入口与证据索引停止复用该判断。
- Habitat 将归档放到运行 save_dir/evidence_blobs，ROS 放到 run_directory/evidence_blobs。显式引用缺失、类型错误或其他 IO 错误保留原始原因。
- 新增请求字节一致性、源文件覆盖、归档丢失和缓存失效测试。全套 **281 passed**；相关模块编译与 diff 检查通过。

该变化保证图像来源可追溯；图像是否满足“到达后的当前观测”仍由现有采集屏障和观测时间负责。剩余完整目标继续包括 Habitat 运动接入、Tools/Skills、主链路兼容清理及真实 episode 对比。

### Habitat 步进接入与请求记录

- `backends/habitat/motion.py` 承载 Habitat 的唯一 env.step 调用；agent 的全部活动步进路径调用 HabitatExecution，再经共享 MotionSupervisor 执行。保留原规划动作，回执明确标记 completion_scope=primitive。
- `motion/contracts.py` 成为公共运动类型的定义来源，real_robot/contracts 只重导出相应类型。MotionGoal 增加显式 action 字段，request_id 去重签名区分不同离散动作。
- 新增仿真动作幂等、HOLD/恢复和观测缓存释放测试。普通每步观测在返回后释放适配器引用。
- 所有三类模型适配器在调用前记录实际 messages、schema 和生成参数；响应记录通过 request_id 关联。失败请求保留输入。解析 fallback 改为显式开启的历史兼容选项，默认传播原始错误。
- 当前工作树全套测试 **284 passed**；编译与 diff 检查通过。

### 中间集成 episode（非最终交付对照）

- 运行源码冻结在 `outputs/harness_refactor_20261023/integration/source/`，清单为 source-manifest.json；后续工作树修改不会改变本次实验。
- 启动器 `/tmp/strive_refactor_integration.py`，容器 `strive-refactor-integration-20261023`，执行 session 73837。参数与 baseline 使用同一模型/镜像、episode index 0、500 steps。
- 日志 `integration/episode.log`；`integration/exit.json` 确认 returncode=0。独立 metrics.csv 显示 baseline/integration 均 success=1、SPL=0.47088843247658624、128 步、travel=9.750003725290298 m。
- 模型调用 baseline=15、integration=13；减少 instruction_parser 和 similar_objects 各一次，bbox/concept/final verifier 次数分别保持 6/5/2。
- 两次导出的 benchmark_source_episode_id 均为空；目前配对依据为相同 episode index/config、初始 RGB 哈希及轨迹汇总，最终实验仍需导出实际 scene/episode/start 身份。
- `integration/initial-image-check.json` 确认 baseline/integration 的前 3 帧 RGB 内容完全相同。
- 本次用于提前发现真实主链路问题；Tools/Skills 和剩余兼容清理完成后仍需最终 episode 对照，不能把中间实验作为整体完成。

### 实现纪律

根据用户补充要求：只修改当前根因涉及的代码；内部协议直接使用明确类型；在模型、传感器和异步执行边界检查实际风险；错误保留原因并向上传递。抽象由实际共用调用方驱动，核心语义以中文注释和必要数学模型说明。

## 验证证据

- 改动前宿主定向单测：35 passed，日志 `baseline/unit-tests.log`。
- 增加语义回归后，定向检查逐步通过；覆盖 ordered+count+anchor、计划不被进度修改、深层不可变、严格编译失败、能力拒绝和兼容投影隔离。
- 宿主默认 Python 缺 Pydantic；全套收集失败，跳过该依赖测试后有 schema smoke 失败。已在未改动基线重复出相同 schema 失败，不能归为本次回归。
- 使用已有 `/home/ubuntu/anaconda3/envs/UrbanNav/bin/python`，关闭无关 pytest 插件自动加载：**245 passed**。此为纯 Python 单测证据，不是 Habitat 或物理执行证据。
- 官方已维护 `strive-hm3d:local` 镜像预检通过：Habitat 0.2.1、Habitat-Sim 0.2.4、Torch 2.0.1+cu118、CUDA、DINO/SAM 和数据可用。
- 基线真实 episode 已启动，模型 `doubao-seed-2-0-pro-260215`，episode index 0、最大 500 steps，原始执行器。结果必须读取退出状态与 evaluator 产物确认。

## 剩余工作（完整目标保持不变）

- P1：进一步清理 legacy plan/spec，确保生产路径的失败处置与能力声明完整；不能只检查 JSON 格式。
- P2：SceneQuery 直接消费 snapshot；移除 SnapshotConfidence/ObjectAdapter；mapper 不持有任务策略或修改对象标签。
- P3：统一 VerificationResult/EvidenceStore，三值语义、证据依赖与 revision；唯一 task reducer；严格模式不允许 verifier fallback accept。
- P4：提取 MotionSupervisor、异步模型请求、过期提案准入、取消确认、未知执行结果对账；拆出 runtime 杂项。
- P5：8 个薄工具及最小 Skills，唯一决策所有者；模型上下文和工具事件可追溯。
- P6：Habitat/ROS 使用同一领域查询与验证，保留后端差异；完成可用 ROS 回放检查。
- P7：删除已被替代的实现和配置，清理转发兼容层；全套测试、真实同 episode 对比及差异报告。

对比时固定 baseline/run-config.json 的模型、episode、预算、感知、执行器和 scorer；保留原始 episode ID/scene/start 信息。
新增严格语义和确定性 ObjectNav 编译是有意变化，应与结构迁移区分。单 episode 只能提供样例行为对比，不能推出总体 SR 提升。


### reducer 写入入口与 STOP 回执修正

- 删除 TaskProgress 的公开状态修改方法，保留状态初始化、读取和序列化；接受计数、拒绝、补拍目标和完成状态统一由 task/reducer.py 更新。
- 顺序检查覆盖所有验证事件。后续目标的补拍或拒绝结果与接受结果一样，保持当前目标状态；新增四种结果的回归案例。
- MotionSupervisor 仅根据执行端终态释放控制权，STOP 请求本身不代表停止已确认。新增 STOP 仍在 RUNNING 时第二次运动被拒绝、终态后正常提交的案例。
- 首个薄工具 query_scene 已接入 Habitat 与 ROS 快照策略的活动调用链，复用同一领域查询和任务服务绑定。其余工具、模型调度与 Skills 仍待完成，本项不代表 P5 完成。
- 当前全套测试 **289 passed**。本轮未运行新的 Habitat episode；上面的中间实验是更早源码快照的结果。


### Agent 工具循环与 Skills（主入口接入待完成）

- harness/tools.py 新增 NavigationTools，八个工具绑定现有 TaskSession、MotionSupervisor、验证器与后端取证函数；query_scene 的共享函数仍供现有 Habitat/ROS 主链路调用。
- harness/runtime.py 新增单步 AgentRuntime，使用既有模型适配器发起结构化 ToolCall；记录 proposed/returned/failed 事件，异常向上传播。每个 episode 限制决策模型调用次数，模型上下文保留当前任务/观测及最近四个工具结果。
- skills/ 包含对象关系搜索、主动观测、阻塞恢复、完成检查四份过程说明，由运行时实际读取到系统提示词。
- 运动工具使用观测实例/视点 ID，从快照解析位姿，经唯一 supervisor 提交；绑定任务 revision、control epoch。模型等待期间相关对象或视点变化时拒绝迟到提案。验证等待期间相关对象变化时保留任务进度。
- tests/test_agent_runtime.py 使用回放模型与模拟控制器，实际通过八工具链路执行查询、取消、再次到达、取证、reducer 计数及完成；覆盖预算耗尽、旧 revision、迟到几何和验证期间对象消失。全套 **293 passed**。
- 本轮范围仍有明确缺口：AgentRuntime 尚未替换 Habitat/ROS 主入口的策略所有者；当前 submit_motion 只支持关联对象的已观测视点，探索/frontier 后端绑定、完整验证预筛接入、共享模型预算和实际 agent episode 仍未完成。此处的取证回放不构成传感器同步或真机闭环证据。


### Habitat Agent 主入口接入（2026-10-25）

- benchmark 增加 --decision_owner legacy|agent（默认 legacy）和 --agent_max_model_calls。agent 分支调用 backends/habitat/agent_runtime.py；legacy 分支继续原规划/step_mod，两者互斥。
- Agent 工具复用原对象 approach/recheck 视点生成、约束预筛、最终验证器和观测建图；任务进度仍经共享 reducer。
- HabitatMotionController 支持 mapper_local waypoint，episode driver 使用原局部规划器推进离散动作；poll_status 只读回执，waypoint 到达不发送 episode STOP。任务完成工具通过后由 driver 发送 STOP。
- 全景扫描、目标复核和路径推进辅助函数的四个直接 env.step 已迁入 HabitatExecution。update_trajectory 中硬编码第 499 步递归 STOP 已移出；episode driver 负责截止，避免运动执行中递归提交第二个动作。
- mapper 快照导出观测生成的 nodes/frontiers；frontier 运动使用显式 frame_id。motion_status 保留 Agent 提交的目标回执，避免后续传感扫描的 primitive 回执覆盖导航结果。
- waypoint 坐标转换、只读查询、离散推进、停止以及 frontier 参考系测试通过；全套 **295 passed**。
- 首次 Agent 实验 agent_integration_20261025 在导入时因 Python 3.9 注解求值失败，exit=1，尚未进入 episode。已在 harness/tools.py 添加 future annotations，独立冻结新快照 agent_integration_20261025_r2 重跑。
- r2 启动器 /tmp/strive_refactor_agent_r2.py，执行 session 76625；当前已进入真实感知处理。其结果以 exit.json、工具 trace 和 metrics.csv 为准，本记录不预判成功。r2 是 Agent 主链路集成实验，ROS Agent 接入和其余完整目标仍待完成。


### 单提交与 5 对 episode 对照准备

- r2 已退出（returncode=1，131.0 秒），第二次模型决策提出 submit_motion 但 arguments={}。原任意字典 schema 没有显式参数字段；ToolArguments 现明确列出对象/视点/frontier ID、revision、epoch、request_id，并拒绝额外字段。
- 已增加字段 schema 和参数保真回归；全套 **296 passed**。模型保留 ID 前缀是协议要求，系统提示词明确逐字复制观测 ID。
- 原理性提交说明位于 harness_refactor_design.md；正式实验脚本 compare_harness_episodes.py 按 Git commit 冻结源码，不将中途修复混入同一批结果。
- 正式比较为 baseline 原策略与重构 Agent 各 5 个 episode。失败计入运行记录；单独报告独立 evaluator 结果与工程失败，避免仅汇总成功运行样本。

### 五例选样与工具协议根因修复（2026-10-25，重跑中）

- `outputs/harness_paired5_20261025` 的旧结果保留为无效配对实验：baseline 重复首个任务，Agent 五次均因 `query_scene(target_id=...)` 异常退出，不能计算五个独立样本的性能。
- 工具参数原先共用一个可选字段集合，schema 允许把其他工具的字段交给 `query_scene`。现将工具名与各自参数模型绑定，运动参数区分对象视点与 frontier；schema 拒绝跨工具字段与缺失必填字段。调用异常继续向上传播，日志保留失败事实。
- 当前 Habitat 在构造时预选任务，首次 reset 保留该任务。旧选择函数只重排 iterator，未替换预选任务。现经公开 `env.current_episode` setter 绑定索引，reset 后逐项核验场景、episode ID、起点、朝向、目标类别，索引越界直接报错。
- 单元回归覆盖非法工具参数、缺失运动参数、独立进程首次 reset 与重复选样；全套 **298 passed**。真实豆包 schema 请求返回有效 frontier 运动参数。
- 真实 Habitat 已逐一 reset 五例；身份清单为 `outputs/agent_repair_20261025/verified-episodes.json`，五个不同场景，目标分别为 toilet、chair、plant、bed、tv_monitor。episode ID 在场景内解释，两个场景的 ID=5 不是同一任务。
- Agent 重跑使用 `outputs/agent_repair_20261025/agent_r3/source` 冻结源码与 SHA256 清单；baseline 使用原 `cf8dd8e` 加选样修复，补丁保存在 `outputs/agent_repair_20261025/paired5/baseline/selection-only.patch`。两组使用同一豆包模型、镜像、五例清单及 500 步预算；并发期间的耗时不用于性能结论。
- 当前记录为重跑中的工程状态，最终完成数量、错误与导航指标以各次退出记录、identity、termination 和 metrics.csv 为准。

#### 真实运行继续定位：观测前缀截断

- r3 在第 155 步已观测到 toilet（`node_info_155.json` 中 index=43），但 `observe()` 的 `objects[:32]` 隐藏该对象，后续模型反复判断未发现目标。已保留模型请求和地图对象清单，主动中止 r3（exit=137），该运行不计入完整五例。
- 删除对象与 frontier 的固定前缀截断，发送当前观测事实；未增加新策略、重试或静默 fallback。新增超过 32 个实例/探索节点的完整性回归，全套 **299 passed**。
- r4 使用新的独立冻结源码重跑同一首例。baseline 首例已完成并核验身份：success=1、SPL=0.4708884325、128 步，其余四例继续运行。

#### 真实运行继续定位：NumPy 几何导出边界

- r4 在第 380 步识别到 toilet 并提出 `verify_target`，证明目标已进入模型上下文；随后 `scene.dependencies.object_dependencies` 在 JSON 编码包围盒几何时抛出 `TypeError: Object of type float32 is not JSON serializable`。保留 exit=1 与原始失败 trace。
- 根因在 `backends/habitat/scene.py`：点云包围盒中心/尺寸经 `tuple(ndarray)` 导出，元素仍为 NumPy 标量。改为在导出边界用 `.tolist()` 转为 Python 数值；依赖哈希和 JSON 编码器保持原协议。
- 新测试用真实 NumPy float32 点云覆盖完整快照 JSON 编码与对象依赖哈希；相关 12 项回归通过。r5 冻结新源码，先运行固定清单的 index=1，再运行其余四例，包括重新执行 index=0；不复用旧版本失败记录充当最终样本。

- 实验执行约定：修正选样后的 baseline 仅执行这一批五例，结果固定复用；Agent 后续修复不触发 baseline 重跑。当前 baseline 已完成 index=0..3，index=4 运行中。

#### 真实运行继续定位：对象视点生命周期

- r5 已通过目标查询、视点生成、对象导航、依赖哈希与最终验证；验证返回 `need_better_view` 后，请求先前发布的 recheck ID 时异常退出：`Viewpoint is not associated with the observed instance`。
- 根因是 `refresh_scene()` 每次从 mapper 重建快照时丢弃工具生成的对象视点。修复为保留仍属于当前观测实例的对象视点，实例消失时清理；拓扑节点仍由 mapper 更新。视点表示场景位置，机器人移动不使其失效；该保留规则不改变证据有效性检查。
- 新回归覆盖“到达后刷新 → 使用原 recheck ID 提交运动 → 实例消失后清理视点”，相关测试通过。r6 冻结同一源码，以两个独立进程并行跑 Agent 清单；保留 r5 失败记录。
- baseline 单批五例全部正常结束，五份身份和首帧哈希均不同：SR=1.0，平均 SPL=0.6006675182。逐例 SPL：0.4708884325、0.4415102046、0.2635974417、0.9810523365、0.8462891758；步数：128、109、274、87、33。baseline 已固定，后续只迭代 Agent。

#### 真实运行继续定位：实例身份与几何版本混用

- r6 的 tv_monitor 补拍运动已成功执行；后续验证因 `Unknown observed instance` 退出。原实例 UID 来自当前点云尺寸/中心的量化哈希，融合观测后，同一地图实例获得另一 UID。
- 原生 `ObjectNode` 现在创建时分配稳定 UID；mapper 已有的关联逻辑原位更新该对象并保留 UID。`candidate_from_object` 直接使用原生 UID，旧无 UID 输入仍使用原兼容路径。几何变化继续由 `scene.dependencies` 的内容哈希表达，避免稳定身份掩盖旧证据失效。
- 新测试覆盖同一原生对象更新位置、点云、标签后 UID 不变、几何依赖哈希变化，以及不同实例 UID 不同；全套 **302 passed**。
- 保留 r6 的 TV 失败及主动中止的 chair 诊断记录。r7 先跑清单 index=4 的较短场景验证完整闭环，确认后再运行其余四例；baseline 固定不变。

#### r7 首个完整 Agent episode

- 稳定 UID 版本在 index=4（tv_monitor）完成 `query_viewpoints → submit_motion → verify_target(need_better_view) → recheck → verify_target(accept) → propose_finish → STOP`，两次验证引用同一实例 UID。
- 正常退出，身份核验通过，`termination=verified_finish`，75 步、7 次 Agent 决策；Habitat success=1、SPL=0.5557742140。总模型调用 11（Agent 7、检测框核验 2、最终验证 2）。其余四例使用同一 r7 冻结源码，以两个进程继续执行。


#### 最终五例统计口径

- baseline 只执行一次修正选样后的五例，冻结在 `outputs/agent_repair_20261025/paired5/baseline/`；后续 Agent 修复和诊断均复用该结果。旧重复 episode 的批次保留为选样故障记录。
- 最终 Agent 使用统一 r7 源码及 SHA256 清单。五例分别启动独立进程，使用相同模型、镜像、权重配置、500 步上限；Agent 决策上限为 60 次，感知和验证调用单独统计。
- 每例比较完整身份（场景、episode ID、起点、朝向、目标类别）及首帧哈希。场景不同而 ID 相同属于独立任务。统计分母固定为五，不排除导航失败。
- 指标由 Habitat evaluator 输出：SR = Σ successᵢ / 5，mean SPL = Σ SPLᵢ / 5。工程退出错误与有效的步数耗尽分别记录，前者先修复，后者保留为导航失败。
- 比较对象是原 STRIVE legacy 策略和 STRIVE Agent 主链路，包含策略/指令路径差异；本实验不是纯结构消融，也不是官方 SuperNav 复现或完整 HM3D benchmark。并发运行耗时不用于延迟比较。
- r7 的 plant 在 500 步终止，Habitat success=0、SPL=0，工具未报错。其地图日志没有 plant/pot/flower 标签，当前证据只支持“未形成可验证目标”，尚不能区分搜索覆盖不足与检测遗漏。
- chair 成功，但路径从 8.75 米增至 19.75 米；总模型调用从 17 增至 55，其中 38 次为检测框核验。后续应分别分析 frontier 选择、补拍几何和感知调用，避免把此次工程修复等同于性能提升。


#### r7 五例最终结果（已全部结束）

完整回执：`outputs/agent_repair_20261025/paired5/combined-results.json`；逐例 command、result、episode.log 分别保存在该目录的 baseline 和 agent_r7 子目录。r7 完整工具日志位于其冻结 source/logs 目录。

| index | 目标 | success（baseline / Agent） | SPL（baseline / Agent） | 步数（baseline / Agent） | Agent 终止原因 |
| --- | --- | --- | --- | --- | --- |
| 0 | toilet | 1 / 0 | 0.4709 / 0.0000 | 128 / 500 | episode_limit |
| 1 | chair | 1 / 1 | 0.4415 / 0.1975 | 109 / 241 | verified_finish |
| 2 | plant | 1 / 0 | 0.2636 / 0.0000 | 274 / 500 | episode_limit |
| 3 | bed | 1 / 1 | 0.9811 / 0.7091 | 87 / 118 | verified_finish |
| 4 | tv_monitor | 1 / 1 | 0.8463 / 0.5558 | 33 / 75 | verified_finish |

- baseline：5/5 成功，SR=1.0000，平均 SPL=0.6006675182；只运行这一批有效五例。
- Agent r7：3/5 成功，SR=0.6000，平均 SPL=0.2924670952；toilet、plant 均因 500 步上限结束，成功案例均通过 verified_finish。
- 两组均完成五个独立 episode；逐例身份及首帧哈希相同，组内身份与首帧哈希各有五个不同值。所有进程 exit=0，r7 五份工具 trace 均无 failed 事件。
- SR 差值为 −40 个百分点，平均 SPL 差值为 −0.3082004230。此五例说明当前 Agent 性能回退，不能声称重构提高导航效果，也不能外推全量 HM3D 结论。
- 本轮完成工具协议、选样和四处后续运行根因修复；保留 r3–r6 失败/中断证据。全套回归 302 passed；未增加静默参数过滤、自动降级或吞错路径。
- 下一步应冻结本次 baseline，针对搜索覆盖、frontier 选择和补拍开销做单因素诊断。ROS Agent 接入与实物验收不在本次已完成范围。

### baseline-guided 实施与真实运行（2026-10-25）

- 增加 baseline_guided 入口并设为默认；legacy 与自由 Agent 入口保留为明确模式。共享 select_exploration_node 执行当前房间优先、原距离排序和原配置的房间选择，普通推进不调用 decision Agent。
- 候选查询和 HabitatNavigationTools 使用同一个快照发布函数，保留仍关联有效实例的对象视点。guided 对象到达复用 check_again、final_instruction_check、view-control 和 reducer；探索到达才执行全景更新。
- guided 目标选择通过同一 MotionSupervisor 提交，地图 visited 状态在探索观测完成后更新。blocked 回执进入 Agent 恢复，包含实际状态与目标；不吞模型/工具错误。
- r1 TV 发现补拍目标继承首次复核 0.5m 到达半径，导致第 30 步原地重复验证。已停止该轮两例并保留退出与诊断记录；首次复核保留原阈值，反馈补拍改用原 planner 的实际到达判定。
- 增加房间策略优先级、距离并列、终态分派、定向验证反馈及补拍到达条件回归，完整测试 312 passed。r2 冻结后运行同一五例，baseline 五例继续复用。最终指标待真实运行结束。
- 此版本复用原房间语义选择，在执行/视点恢复事件才进入 AgentRuntime。多目标语义重排序的新能力和异步模型 worker 尚未新增，不能用本组 ObjectNav 五例宣称这些能力完成。

- r2 在 chair 同步图像对照中发现 waypoint 驱动遗漏 on_node_flag：出发前原地转向进入沿途点云缓存。恢复首次平移前后的原输入语义，并加入实际 controller 的转向→前进→转向回归。r2 已完成的三例与中断的两例保留为诊断记录，最终统计统一使用 r3。
- 最新完整测试 313 passed；r3 使用冻结源码重新执行原五例，baseline 未重跑。


#### baseline-guided r3 五例最终验收

本轮五个独立 episode 已全部结束，使用 Doubao `doubao-seed-2-0-pro-260215`、原 STRIVE Docker 镜像及 HM3D v2 数据，500 步上限、最多两个并发进程。baseline 复用原有一次有效五例，没有重跑。

| 版本 | 成功率 | 平均 SPL | 总步数 | 总模型调用 |
| --- | --- | --- | --- | --- |
| baseline | 5/5 | 0.6007 | 631 | 73 |
| 自由 Agent r7 | 3/5 | 0.2925 | 1434 | 133 |
| baseline-guided r3 | 5/5 | 0.5531 | 690 | 68 |

| 目标 | baseline / guided SPL | baseline / guided 步数 | guided 模型调用 | guided Agent 决策调用 |
| --- | --- | --- | --- | --- |
| toilet | 0.4709 / 0.4483 | 128 / 133 | 15 | 2 |
| chair | 0.4415 / 0.2988 | 109 / 157 | 20 | 0 |
| plant | 0.2636 / 0.2604 | 274 / 275 | 15 | 0 |
| bed | 0.9811 / 0.9811 | 87 / 89 | 13 | 0 |
| tv_monitor | 0.8463 / 0.7771 | 33 / 36 | 5 | 0 |

**结论：这组任务成功率恢复，路径效率尚未完全恢复。** 相对 baseline 平均 SPL 下降 0.04753（相对约 7.91%），步数增加 59，模型调用减少 5。相对自由 Agent，总步数减少约 51.9%，模型调用减少约 48.9%。这些是五例配对回归结果，不代表全量 HM3D，也不是官方 SuperNav 指标。解析路径和策略组织同时改变，因此总调用减少不能全部归因于调度方式。

- 五例均 exit=0、`verified_finish`、task_complete=true，Habitat success=1。完整场景、episode ID、起点、朝向、类别与 baseline/r7 配对一致，首帧哈希一致，组内五个身份与五个首帧哈希各不相同。工具日志 failed 事件为零。
- 313 项测试通过。Python 源码与测试文件 SHA256 与 r3 冻结清单逐项一致。测试覆盖房间优先策略、距离并列、耗尽分支、终态分派、视点保留、反馈移动到达判定及 on_node_flag 建图输入。
- 普通探索 Agent 决策调用为零。toilet 在第 124 步出现补拍耗尽，距目标约 1.033m，要求 1m；Agent 两次调用 query_viewpoints / submit_motion，131 步运动到达，132 步验证通过，133 步 STOP 后 evaluator 成功。该例证明一次恢复闭环实际执行，尚不能推断一般恢复增益。
- toilet 与 plant 的全景观测周期分别与 baseline 对齐。chair 的观测周期从 [13,37,75,101] 变为 [13,37,75,102,133]，多一轮观测，路径从 8.75m 增至 13.00m。第 13/37/75 步图像一致，但第 75 步 frontier 几何不同。恢复 on_node_flag 后差异仍在，故该接口遗漏不是唯一解释。visited 更新时间有意改变；点云/节点生成的差异来源仍需固定输入诊断，不能当作已证实的因果结论。
- plant 比 baseline 多 1 步、约 0.25m；bed 路程几乎相同，多 2 步；TV 多 3 步、约 0.25m。保留这些差距，不做场景特例或放宽完成条件。

运行证据目录：`outputs/baseline_guided_20261025/r3/`。`comparison.json` 为逐例三组对照，`validation.json` 为验收摘要，`navigation-evidence.json` 为观测周期与新入口候选步骤，`mechanism-audit.json` 为机制证据。baseline 缺少 scene_query 文件时候选步骤记 null，表示日志不可用。`source-manifest.json`、`source/`、`runner.py`、逐例 command/result/episode.log 保存复现输入与回执；脚本只引用本机凭证路径，未保存密钥。r1/r2 诊断与中断结果单独保留。

当前实现边界：默认链路直接消费原视点算法和验证反馈，复用原房间级语义选择；Agent 介入运动失败和视点耗尽。复杂多目标语义重排序、面向模型的完整视点评分发布、异步推理 worker 及实物验收仍是后续工作。本次五例验收覆盖默认探索与一次补拍耗尽恢复。


### P1–P4 效率优化实现（2026-10-25）

- P1：路径采样使用 ceil(length/spacing)+1，保留短路径端点；最终视点以 d_object(v)+planner_radius≤required_stop_distance 为几何可行条件。可行视点按 NavMesh 距离优先，视觉评分用于同代价比较；首次视点和反馈视点统一用局部规划器到达判定。guided 模式将目标表面点投影到 NavMesh，为可行近距提案提供路径。最终语义和当前视野仍由 verifier/reducer 判断。
- P2：按四方向一批采集、检测，发现目标类别候选即暂停全景；域查询否决后续扫同位姿剩余方向。图像、深度、相机位姿随帧保存，有限扫描边界不首尾拼接；对象证据按真实相机方向选图。完整 12 方向只执行 11 次旋转。未扫描完整时保留节点未探索状态。跨位置重新观察，当前实现没有把旧位置图像当作新位置证据。
- P3：在当前房间候选中，完全相同的 frontier 栅格集合只保留最近代表；路径代价相差一个前进步长以内，优先覆盖更多 frontier 栅格，再按距离及稳定节点 ID 打破并列。保持房间级语义策略。该收益是 frontier 覆盖代理，不是已标定的目标发现概率。
- P4：标签缓存绑定地图实例 UID。对象点云在 10cm 内双向重叠至少 80% 且匹配唯一时复用；任务版本、检测标签或地图标签改变时重新核验；面积或清晰度提升超过 25% 时刷新证据。距离、可见性、关系和最终完成结果均实时验证。缓存命中及原证据步数写入 semantic_reuse.jsonl。没有新增异步 worker。
- 回归：320 passed。真实 r1 中 TV 成功，toilet 暴露无有效深度帧的空点云下采样错误。已修复空帧处理与 Nx3 空点云形状，增加续扫/空帧测试；r1 仅作为诊断，统一冻结 r2 后重新运行五例，baseline 不重跑。
- 此次合并 P1–P4 验证整体效果，各单项的独立收益需要后续消融；P0 的几何生成分叉仍是已知未决因素。


#### P1–P4 r2 真实五例结果（已完成）

| 版本 | 成功率 | 平均 SPL | 总步数 | 总路程 m | 模型调用 |
| --- | --- | --- | --- | --- | --- |
| 原 baseline（复用） | 5/5 | 0.6007 | 631 | 49.2329 | 73 |
| guided r3（复用） | 5/5 | 0.5531 | 690 | 54.4740 | 68 |
| P1–P4 r2 | 5/5 | 0.6255 | 635 | 56.9829 | 71 |

| 目标 | guided r3 → P1–P4 步数 | guided r3 → P1–P4 SPL | 新版调用 | 标签缓存命中 |
| --- | --- | --- | --- | --- |
| toilet | 133 → 76 | 0.4483 → 0.7652 | 11 | 0 |
| chair | 157 → 126 | 0.2988 → 0.3524 | 30 | 2 |
| plant | 275 → 325 | 0.2604 → 0.1825 | 15 | 0 |
| bed | 89 → 75 | 0.9811 → 0.9811 | 12 | 1 |
| TV | 36 → 33 | 0.7771 → 0.8463 | 3 | 0 |

**验收结论：实现与真实闭环通过；性能是混合结果。** 平均 SPL 比 guided r3 增加 0.07234，总步数减少 55（约 7.97%）；总路程增加约 2.509m，调用增加 3 次。平均 SPL 比 baseline 高约 0.02481，但总步数多 4、总路程多约 7.75m，不能表述为所有效率指标都超过 baseline。SPL 是逐例归一化后求平均，和总路程的变化方向可以不同。

- 五例身份、起点、朝向、类别和首帧哈希与已有两组逐项一致；五个独立身份、五个不同首帧。全部 exit=0、verified_finish、task_complete=true，Habitat success=1。源码与测试文件逐项匹配冻结 SHA256；320 tests passed；工具 failed 事件为零。
- 五例 final_instruction_verifier 均仅调用一次，Agent 决策调用均为零。toilet 最终距离约 0.764m，完成条件满足，原先补拍耗尽的恢复事件未发生。
- 标签缓存真实命中三次（chair 2、bed 1），每条记录保存稳定实例 UID 和来源步数。缓存不复用最终完成判断。
- toilet 最后一轮只观察 4 个方向；chair 最后一轮 8 个方向，plant 最后一轮 4 个方向。发现候选后提前转入接近，未宣称未观察方向已覆盖。
- plant 路程由约 20.983m 增至 29.983m，探索路线回退。P0 几何状态差异、P2 扫描终态和 P3 覆盖排序同时存在，当前不能将回退单独归因于 P3。最小后续诊断是冻结相同候选状态比较原距离排序与覆盖排序，再决定是否做单因素 episode。
- chair 检测框核验由 13 次增至 24 次，尽管缓存命中两次，总调用仍从 20 增至 30。分批感知改变了跨视角候选合并时机，部分新/更清晰证据继续触发模型；缓存命中证明复用路径有效，不证明总体感知成本下降。下一步应先按实例记录比较重复调用与有效证据刷新，保留数据后再调整批次或复用条件。
- 此轮没有进行独立 P1/P2/P3/P4 消融，没有新增模型异步 worker，也没有做完整 HM3D 或实物验收。

证据目录：`outputs/navigation_efficiency_20261025/r2/`。`comparison.json` 包含逐例三组指标、扫描记录与标签复用记录；`summary.json` 保存聚合指标；`validation.json` 保存身份、版本与执行核验；`source-manifest.json` 和 `source/` 保存实际运行源码；`runner.py`、`summarize.py` 与每例 command/result/episode.log 可复核复现过程。r1 工程失败和 TV 成功仅用于诊断，不混入 r2 汇总。
