# 4,000条评测条件交付验收集

**2026-11-08范围调整：需求2,000条与困难500条统一采用IntentionNav；GOAT退出交付验收，历史数据和代码保留。简单物体导航2,000条保持不变。**

**以下 SR/SPL 是用户确认的验收目标，不是实测结果。**

| 分组 | 任务数 | SR目标 | SPL目标 | 配置状态 |
|---|---:|---:|---:|---|
| 简单物体导航 | 2,000 | 0.70 | 0.29 | HM3D/OVON 独立任务待冻结；标准 HM3D val 只有1,000例 |
| 需求指令导航 | 2,000 | 0.51 | 0.24 | IntentionNav 500个几何任务 × 4种表述；原文清单已冻结 |
| 困难指令导航 | 500 | 0.31 | 0.17 | 已冻结模型辅助复杂条件子集；332个几何任务、160场景；单独报告、不重复计权 |
| 不同评测条件总计 | 4,000 | 0.57 | 0.25 | 简单2,000 + 需求2,000；尚未开展新范围全量评测 |

2026-11-08已核对本地与105原始数据：均为500个任务，每项包含`formal_en`、`natural_en`、`casual_en`、`emotional_en`四种表达，共2,000条任务-表述条件、1,994种不同文本。176个场景中的`intents.json`合计仍是同一批500项，没有额外原始池。同一任务的四种表达共享场景、起点及目标，独立几何任务数另列。

用户已确认需求组采用上述2,000条条件，困难500条从中筛选并单独报告。困难组属于需求组，不增加独立条件数，也不重复计入总体SR/SPL。现有`difficulty`仅27项有标注，其中hard为2项，不能据此直接冻结困难500；按原文中的复杂条件及条件关系筛选，不以情绪、句长或本系统运行失败作为困难定义。

需求原文清单已于2026-11-08冻结到`../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/demand_2000.jsonl`，SHA256为`ca07522f36e4b365ff54206ad86578fcbbb7f12a225561dad609170c9095dad4`；数量、四种表述、源数据哈希与原文绑定回执为同目录`demand_freeze_receipt.json`。困难500已冻结到同批次`manifests-final/hard_500.jsonl`，按独立条件数、关系条件数、条件种类数及seed=20261108哈希确定性排序。标注仅使用105本地Qwen；保留两轮原始响应与错误，并直接核对修正87条。第二轮25条无效引用已全部修正；固定10条来自已逐条核对的复杂条件原文，不按资产可用性或导航结果选择。500条是模型辅助筛选结果，尚非全量独立人工复核或官方难度真值。

当前清单以`manifests-final/selection_receipt.json`的哈希为准；早期`manifests/`保留为被替代的筛选证据。困难500覆盖332个几何任务、160场景，固定10条覆盖10个不同几何任务。原始指令和官方任务绑定不变；不把标注/筛选当成导航成绩。新批次仅安排固定10条，运行状态以独立部署的`navigation_start.json`和逐例轨迹/评分回执为准，没有开展全量导航测试。

目标配置：`configs/benchmarks/acceptance_4500.json`。总体汇总仅纳入简单与需求两组；保留原有各组及总体SR/SPL验收目标。历史`configs/benchmarks/demand_2000/`仍是旧的IntentionNav 1,000 + LangMap 1,000清单，不作为新交付输入。

\[
\mathrm{SR}_{all}=\frac{2000\mathrm{SR}_{simple}+2000\mathrm{SR}_{demand}}{4000},\qquad
\mathrm{SPL}_{all}=\frac{2000\mathrm{SPL}_{simple}+2000\mathrm{SPL}_{demand}}{4000}.
\]

简单与需求分组目标的加权结果分别为0.605与0.265；原总体验收目标0.57/0.25保持不变，不将计算值改称新目标。实测聚合使用4,000条不同条件的结果；困难500子集单独检查0.31/0.17门槛，不额外计权。每个benchmark的成功条件单独声明，项目总指标不等同于任一原论文指标。

## 历史资料：GOAT困难子集（已退出交付验收）

官方来源：<https://github.com/Ram81/goat-bench>；下载使用 README 的 Google Drive episode 包，压缩包 SHA256：`d731a81dccb1c4ba2b825a03d8ff6ed0cd8fc22935d7aacd98e78b53dfdbf904`。固定源码版本保存在 105 的 `goat_bench_source/revision.txt`。

- 官方 val_unseen：360 条复合 episode、856 个 description 子任务。
- 去重依据：场景、原始起点/朝向、目标实例、原始指令，得到 **585** 个独立组合。
- 按 seed=20261105 的 SHA256 排序固定选取 **500** 个，覆盖 **36** 个场景；105 场景文件全部存在。
- 保留官方描述，不增加模型改写；使用官方 <=55词过滤规则。
- 每条语言子任务从父 episode 原始起点独立运行，保留父 ID、子任务索引、源文件和目标实例。
- **协议名 `goat_language_single_parent_start_v1`：这是项目派生单目标集，不是原版多模态连续任务。** 不继承此前子任务的地图记忆与实际结束位置。“困难”是项目名义分组，尚未通过实测证明比其他组更难。

原版 README 概述使用1m物体距离；固定源码 `config/tasks/goat_stretch_hm3d.yaml` 和 `measurements/nav.py` 实际使用 **目标视点测地距离 <0.25m 且执行子任务STOP**。本派生集以实际代码为依据，单目标 STOP 结束任务，复用 Habitat 的 Success/SPL：

\[
S_i=\mathbf1[\mathrm{STOP}_i\land d_{geo}(x_T,V_i)<0.25],\quad
\mathrm{SPL}_i=S_i\frac{d_{geo}(x_0,V_i)}{\max(d_{geo}(x_0,V_i),L_i)}.
\]

保留 STRIVE 当前 RGB-D、相机和运动设置；尚未复现原 GOAT 的整套传感器/复合动作配置，不能将派生集成绩直接与原论文数字比较。

## 105 目录及入口

```text
/mnt/nvme0/haosu/strive/data/datasets/goat_bench_source/goat-bench.zip
/mnt/nvme0/haosu/strive/data/datasets/goat_bench_source/raw/data/datasets/goat_bench/hm3d/v1/
/mnt/nvme0/haosu/strive/data/datasets/goat_language_single_500/tasks.json.gz
/mnt/nvme0/haosu/strive/data/datasets/goat_language_single_500/tasks.json.manifest.json
/mnt/nvme0/haosu/strive/data/datasets/goat_language_single_500/asset_audit.json
```

任务包 SHA256：`7aea09faa6837eb646269235c976d13e975736561e184cc260ff9ba897cd76de`。

- 构建入口：`scripts/prepare_goat_language.py`。
- 评测入口：现有 `objnav_benchmark_with_process_obs.py --benchmark goat_language --dataset_path ...`。
- 共享 `benchmark/instance_goal_dataset.py` 与 LangMap 使用同一个逐任务实例加载器，避免同类别不同实例被合并。
- 原图像/对象GT留在环境与评测器；指令模块读取原始语言文本。

## 历史GOAT接入记录（2026-11-05，不作为当前交付条件）

- 本地必要任务转换/去重/provider测试通过；真实 Habitat 上完成500条任务加载、首例reset和距离计量。
- 首次加载任务：`val_unseen:00815-h1zeeAwLh9Z:8:3`；初始距离4.447m，成功阈值0.25m。
- 真实加载回执：`/home/haosu/Workspace/Navigation/STRIVE/outputs/deployment105/goat-loader-receipt.json`。
- **没有启动500例导航评测，也没有得到GOAT实测SR/SPL。** 其余499例的完整导航可达性和模型闭环仍待正式运行。

## 历史需求指令集冻结（2026-11-05，已被当前范围替代）

- **IntentionNav：500 × 2 = 1,000 条条件**，采用 `natural_en` 与 `emotional_en`，每种措辞分别统计；配对分析按 `task_id` 关联。
- **LangMap：1,000 个独立任务**，region 330、instance 368、room 302；单纯 object 类别任务留在简单组。seed=20261105，固定 SHA256 排序。
- 清单、源文件哈希、数量回执：`configs/benchmarks/demand_2000/`。`langmap_runtime.json` 可直接传给已有 `--episode_manifest`，同时加载全部 LangMap annotations（9,314 条），设置 `--dataset_episodes 9314 --eval_episodes 1000`。
- 构建入口：`scripts/prepare_demand_acceptance.py`。清单本身不证明场景、渲染或闭环已经验收。
- **据 IntentionNav 接入记录，Isaac 6.0.1 的单场景渲染验收已通过；STRIVE 任务/感知/运动闭环仍待实现与验收，当前没有导航指标。** 详见 `docs/intentionnav_integration.md`。
