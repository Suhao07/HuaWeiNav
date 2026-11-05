# Documentation Layout

本目录只保存可审阅、可维护的项目文档和文档源文件。实验运行结果、模型验收回执和
渲染后的图片不应写入 `docs/`。

## 核心技术方案

- **[项目技术白皮书](project_technical_whitepaper.md)**：第3.2、5.9、6.3—6.4节概括共享地图、定位校准、区域决策与局部导航数学模型。
- **[轻量先验地图方案](prior_map_mode.md)**：详细设计、实施边界与验收计划。

## 实物模式（已完成 bridge-off 规划与速度输出验收）

- [正式实物模式一键入口](real_robot_navigation_entry.md)：默认关闭底盘 bridge；启动、日志及后续 live 配置。
- [重要参数调节与迁移说明](real_robot_parameter_tuning.md)：当前值、调节依据、参数配合和验证方法。
- [部署状态与后续现场验证](real_robot_deployment_todo.md)：完成口径和历史证据；实际行驶到达尚未验收。

## 目录约定

```text
docs/
  *.md                         项目架构、部署、接口和实验方法文档
  *.tex                        技术白皮书的 LaTeX 源文件
  diagrams/                    Mermaid、Graphviz 和渲染配置源文件
  assets/                      文档引用的静态素材

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
