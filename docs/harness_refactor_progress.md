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
