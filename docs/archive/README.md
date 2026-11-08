# 历史文档归档索引

返回 [文档门户](../README.md)；从零部署使用 [交付 README](../../README.md)。

本目录由20篇精简为11篇历史材料（不含本索引）：9篇早期实物记录合并为1篇摘要，停止距离诊断并入正式参数手册，另保留10篇历史材料（含 IntentionNav 与验收范围原文副本）。历史“当前/下一步”、测试数和验收状态仅对记录当时版本有效，不是最新部署指令。合并摘要保留关键成功/失败事实、测量、证据路径和未验收边界；九篇完整版可按摘要中的实际 Git 基线与原文件对照表追溯，不再完整保存在本目录。其余材料保留原文，历史命令、机器路径、源码行号和结果值按当时工作目录与版本解释，Markdown 链接按归档位置调整。

本次只整理文档，**不删除、移动或复制 `logs/`、`outputs/` 及忽略产物 `docs/logs/`**。日志与输出继续留在原位置；实物无运动/离线测试不等于真实行驶验收，历史记录不授予新的控制权限。当前操作、安全边界与回滚要求以正式页为准。

## Harness（4篇）

当前正式页：[Harness 与运动所有权](../harness_refactor_design.md)、[指令、检索与验证](../instruction_adapter.md)、[HM3D 操作指南](../hm3d_docker_benchmark_guide.md)。

- [重构执行记录](harness_refactor_progress.md)
- [十个新 episode 的失效诊断（2026-10-25）](harness_10episode_diagnosis_20261025.md)
- [r7 五例性能回退诊断](harness_r7_regression_analysis.md)
- [baseline-guided 修改与验证计划](harness_baseline_guided_implementation_plan.md)

## 先验（2篇）

当前正式页：[轻量先验地图方案](../prior_map_mode.md)、[实施清单](../prior_map_todo_checklist.md)、[手机 Stray Scanner 导入](../prior_map_stray_scanner.md)、[HM3D 先验评测](../hm3d_prior_map_evaluation.md)。

- [HM3D 先验采集首轮实现记录](prior_map_capture_progress.md)
- [旧版接口与命令参考](prior_map_legacy_reference.md)

## 实物（3篇）

当前正式页：[导航入口](../real_robot_navigation_entry.md)、[参数调节](../real_robot_parameter_tuning.md)、[部署状态与待验收事项](../real_robot_deployment_todo.md)、[控制安全计划](../real_robot_controller_safety_plan.md)、[静态候选回滚手册](../real_robot_static_candidate_runbook_20261105.md)。

- [实物部署历史摘要（2026-08—09）](real_robot_history.md)（合并9篇早期记录，含原文 Git 恢复基线与章节对照）
- [椅子录包离线诊断（2026-11-05）](real_robot_chair_detection_diagnosis_20261105.md)
- [椅子实例对应与掩码离线核查（2026-11-05）](real_robot_correspondence_mask_diagnosis_20261105.md)

停止距离诊断、回归对照表与证据路径已并入 [参数手册](../real_robot_parameter_tuning.md#停止距离离线诊断)，不再保留单独历史文件。

## benchmark（2篇）

当前正式页：[IntentionNav 部署与评分](../intentionnav_integration.md)、[4,000条验收条件](../acceptance_4500.md)、[Benchmark Provider](../benchmark_providers.md)。

- [IntentionNav 接入与验收原文（2026-11-05）](intentionnav_integration_20261105.md)
- [验收范围原文（2026-11-08）](acceptance_scope_20261108.md)
