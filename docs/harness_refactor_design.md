# STRIVE 任务合同与 Harness 重构

本提交将指令语义、任务进度、地图观测和运动占有分开，接入一个可选择的 Habitat Agent 工具入口。现有感知、建图、局部规划和 benchmark evaluator 继续承担原职责。

## 状态所有权

- `instruction/TaskSpec` 在同一 revision 内不可变，保存顺序、数量和约束；DetectorAdapter 生成检测词表。
- `task/reducer.py` 根据验证结果推进进度。对目标 i，数量为已接受实例集合的基数：N_i = |A_i|；顺序约束作用于接受、拒绝和补拍结果。
- `scene/` 描述观测快照；任务匹配独立于地图原始标签。
- `verification/` 保存三值判断和图像内容依赖；请求的图片字节与 SHA-256 归档一致。
- `motion/MotionSupervisor` 唯一持有执行槽位，满足 |M_active(robot)| ≤ 1；取消或 STOP 请求由执行端终态确认，未知提交结果保留占有权。

因此任务更新遵循 q_(k+1) = Reduce(q_k, e_verified; TaskSpec_r)，普通运动到达不能直接当作任务成功。

## 调用闭环

Habitat 的 `--decision_owner legacy|agent` 选择唯一高层决策入口。Agent 经八个工具查询场景、生成视点、提交/查询/取消运动、请求验证和完成检查。episode driver 推进原导航器生成的离散动作，验证器提供证据判断，reducer 更新任务进度。到达回执、任务完成和 evaluator 的 success/SPL 分别保留。

四份 Skills 提供搜索、主动观测、恢复、完成流程。模型调用前记录消息与 schema；工具记录提案、结果和失败。接口显式列出参数字段，避免任意字典在 structured-output schema 中丢失参数。

ROS 现有语义策略已经共享任务、场景、运动所有者和验证结果；推理从控制 timer 移入 worker，相关对象/任务/控制版本变化会使迟到响应失效。

## 本提交的边界

这是阶段性架构提交。ROS 的新 Agent 工具入口、统一全模型预算、剩余 legacy plan/spec 与证据索引清理仍待完成。ROS 回放和真机闭环不能用 Habitat 或纯 Python 测试代替。当前 Habitat 快照的姿态/坐标能力也需要继续统一说明。

此前原始 baseline 与中间结构迁移各一个 episode 均成功，SPL=0.470888、128 步；模型调用为 15 与 13。这个结果对应早期冻结快照，不代表当前 Agent 版本。

## 验证与实验

宿主已有 UrbanNav 环境全套 296 项测试通过。真实 Agent 集成暴露的 Python 3.9 注解问题和空工具参数 schema 已修正。最终性能以冻结 commit 的实验为准。

`scripts/compare_harness_episodes.py` 冻结 baseline/refactor Git 修订，对相同 0–4 episode 各运行一次；每个样本独立进程，保留工程失败、原始指标和初始 RGB 哈希。模型、镜像、数据配置和 500 步预算相同。baseline 使用旧策略，refactor 使用 Agent，差异包含策略变化，不能解释为纯目录调整的收益。
