# LangMap 接入与 105 验收

**状态（2026-11-05）：数据和单目标适配完成；输出契约与模型路由修复后，同一 episode 已正常结束。工程闭环通过，该例导航失败。**

## 数据与复用范围

- 官方代码：<https://github.com/bo-miao/LangMap>，固定版本 `3f2cd6e37a7f06818f515b480b0e81293a8269af`。
- 官方标注：<https://huggingface.co/datasets/bo-miao/LangMap>，版本 `b6683a38d5ed441114cd62bc6426948b3d07a53f`。
- 105 标注目录：`/mnt/nvme0/haosu/strive/data/datasets/langmap/annotations`。
- 105 场景目录：`/mnt/nvme0/haosu/strive/data/scene_datasets/hm3d_v0.2/val`。
- 36 个标注文件，共 2,793,985 字节，逐文件 SHA256 校验通过；对应场景的 basis.glb、semantic.glb、semantic.txt 全部存在。
- **复用场景资产、原指令编译、感知融合、baseline-guided 探索和局部导航。** 不覆盖原语义标签文件；文本类别由 LangMap 标注提供。

| 简洁描述单目标任务 | 数量 |
|---|---:|
| Object | 1,460 |
| Room | 2,289 |
| Region | 2,687 |
| Instance | 2,878 |
| 合计 | **9,314** |

106 条官方黑名单任务已剔除。对全部 9,314 条，与官方 `LangMapDatasetV1.from_json` 的实际转换逻辑逐条比较：**ID、指令、起始位姿、目标视点完全一致**。简洁/详细描述不是独立几何任务，不能混加为独立 episode 数。首版接入简洁描述单目标；原始数据中的序列任务已经下载，连续多目标执行尚未接入。

## 最小模块改动

| 文件 | 职责 |
|---|---|
| `benchmark/langmap.py` | 官方任务转为现有导航输入，保留来源哈希；定义终态指标 |
| `benchmark/langmap_habitat.py` | 保留每条任务的独立目标集合，注册 Stretch NavMesh 与逐动作路径计量 |
| `benchmark/langmap_exclusions.json` | 固定官方黑名单，来源为上述代码版本的 task.py |
| `benchmark/providers.py` | 注册 `--benchmark langmap` |
| `objnav_benchmark_with_process_obs.py` | 自动使用语言指令；正式结果写入 LangMap 指标，原 Habitat STOP 指标另列 |
| `tests/test_langmap.py` | 目标身份、数据往返和成功阈值的必要契约测试 |

**按类别合并目标集合会破坏实例和区域任务。** 因此新加载器复用 Habitat 的文件/场景加载与 ObjectGoal 数据结构，逐 episode 保存合法目标；模型输入只接收任务文本，GT 视点留在评测器。

## 官方指标

设合法目标视点集合为 \(V\)，机器人位姿为 \(x_t\)：

\[
d_t=\min_{v\in V}d_{\mathrm{geo}}(x_t,v),\qquad
L=\sum_{t=1}^{T}\|x_t-x_{t-1}\|_2,
\]
\[
S=\mathbf1[d_T<0.25\mathrm{m}],\qquad
\mathrm{SPL}=S\frac{d_0}{\max(d_0,L)}.
\]

**成功不要求 STOP，采用实际结束位置；不可达任务计失败。** 无目标视点时，遵循官方加载器使用对象位置。路径长度在每个仿真动作后累积，不能用稀疏建图帧轨迹替代。零路径且起点已满足任务时按 SPL=1 的退化约定处理。

每次场景载入后按官方 Stretch 参数在内存重建 NavMesh：半径 0.17m、高度 1.41m、最大攀爬 0.10m、cell_height 0.05m、最大坡度 45°。原始资产保持原样。

**策略距离与评测距离分别定义。** 原导航器对象接近阈值保留 1.0m（预测对象表面距离），官方 0.25m 用于合法视点测地距离；二者不可互相替代。模型通过最终验证，也不自动等于官方成功。

## 评测条件边界

本次为 **STRIVE RGB-D 配置的接入验证**，不是 PlaNaVid 论文配置复现：

| 设置 | 当前 STRIVE | 官方 PlaNaVid LangMap 配置 |
|---|---|---|
| 观测 | RGB-D，640×480 | RGB，640×480 |
| RGB HFOV | 79° | 120° |
| 相机高度 | 0.88m | 1.25m |
| 转角 | 30° | 30° |
| 动作 | 包含 LOOK_UP/DOWN | STOP、前进、左右转 |
| sliding | False | True |
| 场景 NavMesh、任务、终态 SR/SPL 定义 | 对齐官方 | 官方 |

这些差异是复用当前导航策略的显式实验条件；正式跨方法比较需要统一或明确报告。首例未启用历史先验地图。

## 首例记录与阻塞

- Task ID：`00802-wcojb4TFT35_region_64`。
- 指令：`Find the ceiling fan in the bedroom that has bed with purple-and-gray bedding`。
- 模型：105 本机 `huaweivln-qwen3.5-4b`，服务日志确认 `thinking=false`。
- 运行：18:43:22–18:47:17（UTC+8），墙钟 **235 秒**；最后记录步骤 **12**；退出码 **1**。
- 数据加载、NavMesh 重建、指令解析、初始扫描与对象感知已执行。
- 阻塞：候选匹配输入 16 个 UID，模型返回空列表，触发 `Batch concept matcher UID mismatch`。
- 另一个问题：concept_grounding 产生重复词串，在 1,024 和 2,048 token 处截断。现有 detector_adapter 捕获错误后返回空映射，随后退回原词；本轮保留原始证据，尚未修改这条模型链。
- 指令解析还将床品属性表达为 bed 与 ceiling fan 的共现约束；该语义转换值得单独检查，当前不能视为指令正确理解的证明。

**本次是工程异常退出，不生成有效 episode SR/SPL，不能记作正常导航失败或成功。** 下一项必要工作是修复检测词映射的输出契约和候选全覆盖响应，再以相同 task ID 继续验收；不更换为更容易的任务。

105 证据：

```text
/home/haosu/Workspace/Navigation/STRIVE-deploy/langmap-episode.log
/home/haosu/Workspace/Navigation/STRIVE-deploy/langmap.started
/home/haosu/Workspace/Navigation/STRIVE-deploy/langmap.finished
/home/haosu/Workspace/Navigation/STRIVE-deploy/langmap.exitcode
/home/haosu/Workspace/Navigation/STRIVE/logs/deployment105-langmap-instance-001/episode-0/
/mnt/nvme0/haosu/strive/data/datasets/langmap/audit.json
/mnt/nvme0/haosu/strive/data/datasets/langmap/task_parity.json
```

日志目录的 `instance-001` 是初次启动命名；实际身份以 `episode_identity.json` 为准，是上述 region 任务。

必要测试：本地 provider/转换/指标测试 3 项通过；本地缺少 Habitat 的往返测试在 105 维护镜像内执行，镜像内同一文件的 3 项契约全部通过。没有安装额外测试依赖。镜像回执为 `STRIVE-deploy/langmap-contract-direct.log`。

## 2026-11-05：两处语义输出契约修复

**根因：生成协议比调用方的任务要求更宽松。** 检测词映射允许无限词表和长文本；候选匹配允许空数组，但调用方要求全量候选判断。

- `GroundingResult`：每组最多 4 个短语，每个短语最多 64 字符；描述和理由分别最多 240/160 字符。保留开放词汇，约束输出规模。提示词与生成 schema 使用同一限制。
- `perception/detector_adapter.py`：模型/解析异常原样向上传递，移除捕获所有异常后返回空映射的路径。
- `concept_batch_response_model(U)`：为本批 UID 集合构造必填对象字段，所有候选均需返回判断，包括否定匹配。每个判断的布尔、置信度、理由也是必填项；原数组响应已退出生产链及部署验收链。

\[
\operatorname{dom}(Y)=U,\quad
Y[u]=(m_u,c_u,t_u,r_u),\quad u\in U.
\]

空响应不能表示“全部不匹配”；全部不匹配应显式返回每个候选的否定判断。继续使用已有结构化解析及完整上下文重试机制，未增加重试次数。

保存请求复验（105 本机 Qwen3.5，thinking=false）：

| 原始输入 | 修复后耗时 | 输出 tokens | 结果 |
|---|---:|---:|---|
| bedroom 词映射 | 7.62s | 120 | 首轮 stop，完整返回 |
| ceiling fan 词映射 | 1.71s | 58 | 首轮 stop，映射为 fan |
| 16 候选匹配 | 27.70s | 1009 | 16/16 UID，全部显式否定，首轮 stop |

三条复验共 37.02s；本地必要测试 17 项通过。复验回执：`outputs/deployment105/langmap-semantic-replay/{receipt,passed}.json`。

**协议通过不等于语义完全正确。** bedroom 仍被映射为家具提示词；原指令的床品属性转换也未在本轮重构。这些领域语义问题与本次重复生成/漏项故障分开记录。

随后使用冻结的 `episode_manifest.json` 重跑 `00802-wcojb4TFT35_region_64`；任务、起点、场景、Qwen 服务与导航配置保持相同。新结果目录：`logs/deployment105-langmap-region64-semanticfix`，启动及退出记录为 `STRIVE-deploy/langmap-fixed.*`。最终结果见下方运行回执。

### 切换房间的统一模型路由

第一次修复后重跑在第 165 步触发 `UnboundLocalError: completion`：旧 mapper 的选房间/重定位方法按供应商分支，只覆盖 Gemini、CogNav、OpenAI，没有调用 self_hosted。两处现在直接调用已有 `get_client_and_model(self.vlm)`；移除重复供应商分支和硬编码模型名，保留原提示、输出结构与房间策略。

第 165 步保存的选房间输入复验通过（18.47s，选择候选房间 0）。首次修复后运行 19:12:24–19:18:55，391s，异常退出，不产生有效 SR/SPL。最终重跑目录为 `logs/deployment105-langmap-region64-semanticfix2`，任务身份继续使用同一冻结 manifest。

### 原始指令示例（wcojb4TFT35）

| 官方任务 ID 后缀 | 英文指令 | 中文说明 |
|---|---|---|
| object_0 | Find the laundry basket. | 找到洗衣篮 |
| room_0 | Find the laundry basket in the Bedroom | 找到卧室里的洗衣篮 |
| region_0 | Find the laundry basket in the walk-in closet that has clothes near bathtub | 找到衣帽间里的洗衣篮，区域特征是浴缸附近有衣物 |
| instance_0 | Find the striped bed near black bookshelf | 找到黑色书架附近的条纹床 |
| region_64 | Find the ceiling fan in the bedroom that has bed with purple-and-gray bedding | 找到有紫灰色床品的卧室中的吊扇 |

示例来自固定版本原始标注，按官方指令模板生成；中文为释义。这些任务侧重目标描述和空间消歧。

### 最终同任务重跑结果

2026-11-05 19:21:32–19:28:10（UTC+8），墙钟 **398s（6分38秒）**，退出码 0。任务 `00802-wcojb4TFT35_region_64` 的场景及起始位姿与原运行相同。

| 指标 | 实测 |
|---|---:|
| 完整任务数 | 1 |
| SR | 0/1 = 0 |
| SPL | 0 |
| 步数 | 165 |
| 实际路径长度 | 14.000m |
| 初始最短距离 | 2.426m |
| 最终合法视点距离 | 1.862m |
| 终止原因 | search_exhausted |
| 模型调用总数 | 16 |

16 次调用包含：指令解析1、检测词映射2、候选匹配4、框内类别判断4、最终指令验证3、导航Agent1、房间选择1。原两处输出故障和 self_hosted 房间路由错误未复现。最终验证仍为 need_better_view，任务未完成。**工程闭环通过不代表该任务成功；本例不能用于宣称总体成功率提升。**
