# 文档入口与归档约定

**从零部署统一使用 [交付 README](../README.md)。** 安装步骤不在多份文档中重复维护；以下页面按操作、合同、验收与历史证据分工。实验日志、模型回执及渲染产物不新增到 `docs/`。

## 部署与评测

| 文档 | 维护内容 |
|---|---|
| [交付 README](../README.md) | 从零准备数据/权重、Docker、模型、单例运行与交付顺序 |
| [LVLM 接入](lvlm_server_deployment.md) | 商业 API、独立 Qwen3.5 部署包、可选 ms-swift、认证与错误边界 |
| [HM3D 操作指南](hm3d_docker_benchmark_guide.md) | 简单2,000候选制作、去重与哈希；容器路径、固定身份、完整重跑/配对 |
| [IntentionNav 部署与评分](intentionnav_integration.md) | Isaac6 资源、六字段输入、官方绑定、运动与独立评分 |
| [4,000条验收条件](acceptance_4500.md) | 需求2,000与困难500制作、全量复核冻结路径/哈希、复现命令及验收边界 |
| [Benchmark Provider](benchmark_providers.md) | 数据来源、成功阈值与 GOAT 运动证据评分 |
| [LangMap 接入](langmap_integration.md) | 保留研究入口，不计入当前交付条件 |

## 当前运行合同

- [指令、检索与验证](instruction_adapter.md)：TaskSpec、初始解释、搜索修订、需求义务、三值关系与当前图像证据。
- [Harness 与运动所有权](harness_refactor_design.md)：三种 Habitat 决策模式、八工具、五份 Skills、版本准入与预算。
- [证据复用与并发边界](p3_p4_evidence_reuse_and_concurrency.md)：专题设计与实现边界，在线表现仍须独立验收。

## 核心技术方案

- **[项目技术白皮书](project_technical_whitepaper.md)**：第3.2、5.9、6.3—6.4节概括共享地图、定位校准、区域决策与局部导航数学模型。
- **[轻量先验地图方案](prior_map_mode.md)**：详细设计、实施边界与验收计划。

## 实物模式：已实现

真实输入的规划与安全速度输出已完成 bridge-off 验证；默认不驱动底盘，真实行驶/到达/视觉 STOP 独立验收。常用调试值：停止 **0.5 m**、到达 **1.0 m**、前视/减速 **1.0 m**；外参、车体、速度与保护参数统一查阅下方参数手册。

- [正式实物模式一键入口](real_robot_navigation_entry.md)：默认关闭底盘 bridge；启动、日志及后续 live 配置。
- [重要参数调节与迁移说明](real_robot_parameter_tuning.md)：当前值、调节依据、参数配合和验证方法。
- [部署状态与后续现场验证](real_robot_deployment_todo.md)：完成口径和历史证据；实际行驶到达尚未验收。

- [控制接口合同](real_robot_controller_contract_zh.md)、[控制安全计划](real_robot_controller_safety_plan.md)：保留实物权限、安全与验收要求。
- [坐标与 TF](real_robot_tf_graph.md)、[相机/雷达标定采集](real_robot_d435i_mid360_calibration_capture.md)：传感器迁移依据。
- [静态候选回滚手册](real_robot_static_candidate_runbook_20261105.md)、[影子部署记录](real_robot_shadow_deployment_20261105.md)：仍含操作与回滚步骤，不作为无用过程文件删除。

## 先验地图

- [HM3D 先验评测](hm3d_prior_map_evaluation.md)：生成、加载及有/无先验比较。
- [手机 Stray Scanner 导入](prior_map_stray_scanner.md)：观测协议、转换与标定边界。
- [先验地图实施清单](prior_map_todo_checklist.md)：保留未完成事项；不能把历史“通过”记录当作实物验收。

## 历史记录

[归档索引](archive/README.md) 保留 Harness 回退诊断、先验采集记录及已替代评测范围；早期实物连接、存储和阶段验收已合并为历史摘要，停止距离诊断并入参数手册，不再各占一份文件。摘要保留关键结果与失败边界，原文可由记录的 Git 基线追溯；原始日志和输出不删除。历史命令不作为最新部署指令，现役安全/回滚步骤仍保留在正式页。

## 目录约定

```text
docs/
  *.md                         项目架构、部署、接口和实验方法文档
  *.tex                        技术白皮书的 LaTeX 源文件
  diagrams/                    Mermaid、Graphviz 和渲染配置源文件
  assets/                      文档引用的静态素材
  archive/                     已替代的过程说明、诊断与历史范围（含索引）

references/papers/             外部论文和只读参考资料
outputs/docs/figures/          文档图表渲染产物，默认不纳入 Git
logs/                          仿真、实物回放和验收运行产物
```

## 文档与产物边界

| 类型 | 位置 | 是否提交 Git | 说明 |
|---|---|---:|---|
| 项目说明、接口合同、设计文档 | `docs/*.md` | 是 | 面向开发、部署和评审的稳定内容 |
| 白皮书源文件 | `docs/*.tex` | 是 | 论文正文和可复现排版源文件 |
| 图表源文件 | `docs/diagrams/` | 是 | 由源文件重新渲染得到图片 |
| 文档静态素材 | `docs/assets/` | 视用途 | 仅保留项目文档确实引用的素材 |
| 外部论文 | `references/papers/` | 视仓库策略 | 只读参考资料，不作为项目实现文档 |
| PNG/SVG 渲染结果 | `outputs/docs/figures/` | 否 | 可由图表源文件重新生成 |
| 运行日志、视频、JSON 回执 | `logs/` 或 `outputs/` | 否 | 由 benchmark、部署或验收命令生成 |

`outputs/` 和 `logs/` 已在 `.gitignore` 中忽略。正式文档中只引用生成命令、源文件和
产物 schema，不提交某一次运行产生的图片、模型路径、机器信息或临时回执。

## 图表复现

Graphviz 图表从 `docs/diagrams/*.dot` 生成：

```bash
mkdir -p outputs/docs/figures
dot -Tsvg docs/diagrams/prior_map_floorplan_builder_flow.dot \
  -o outputs/docs/figures/prior_map_floorplan_builder_flow.svg
dot -Tpng docs/diagrams/prior_map_floorplan_builder_flow.dot \
  -o outputs/docs/figures/prior_map_floorplan_builder_flow.png
```

Mermaid 图表从 `docs/diagrams/*.mmd` 生成。渲染时使用同目录下的
`mermaid_puppeteer_config.json`，输出放入 `outputs/docs/figures/`，不要回写到
`docs/`。

## 运行产物记录

实验或部署文档应记录：

- 生成命令和代码版本；
- 输入数据或模型的版本标识；
- 输出目录和回执 schema；
- 失败时的最小复现信息。

不要把大模型权重、API key、真实机器人传感器数据或含敏感信息的部署回执复制到
文档目录。
