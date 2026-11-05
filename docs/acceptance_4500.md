# 4,500 任务验收集

**以下 SR/SPL 是用户确认的验收目标，不是实测结果。**

| 分组 | 任务数 | SR目标 | SPL目标 | 配置状态 |
|---|---:|---:|---:|---|
| 简单物体导航 | 2,000 | 0.70 | 0.29 | HM3D/OVON 独立任务待冻结；标准 HM3D val 只有1,000例 |
| 需求指令导航 | 2,000 | 0.51 | 0.24 | IntentionNav 1,000 条条件 + LangMap 1,000 个独立任务，清单已冻结 |
| 困难指令导航 | 500 | 0.31 | 0.17 | GOAT val_unseen 语言单目标派生子集已冻结、部署 |
| 总计 | 4,500 | 0.57 | 0.25 | 全量评测尚未启动 |

隐式需求与显式目标描述分别报告样本数、SR、SPL。**用户已确认 IntentionNav 使用 500 个几何任务 × 两种措辞，总计 1,000 条评测条件；共享几何任务明确标注。** 总表按运行条件加权，独立几何任务数另列。目标配置：`configs/benchmarks/acceptance_4500.json`。

\[
\mathrm{SR}_{all}=\frac{\sum_g N_g\mathrm{SR}_g}{4500},\qquad
\mathrm{SPL}_{all}=\frac{\sum_g N_g\mathrm{SPL}_g}{4500}.
\]

分组目标的加权结果分别为 0.572222 与 0.254444，四舍五入后为 0.57/0.25。实测聚合直接使用逐任务结果，保留各组门槛，不能用总分掩盖某组未达标。每个 benchmark 的成功条件单独声明；项目总指标不等同于任一原论文指标。

## GOAT 困难指令子集

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

## 当前验收边界

- 本地必要任务转换/去重/provider测试通过；真实 Habitat 上完成500条任务加载、首例reset和距离计量。
- 首次加载任务：`val_unseen:00815-h1zeeAwLh9Z:8:3`；初始距离4.447m，成功阈值0.25m。
- 真实加载回执：`/home/haosu/Workspace/Navigation/STRIVE/outputs/deployment105/goat-loader-receipt.json`。
- **没有启动500例导航评测，也没有得到GOAT实测SR/SPL。** 其余499例的完整导航可达性和模型闭环仍待正式运行。

## 需求指令集冻结（2026-11-05）

- **IntentionNav：500 × 2 = 1,000 条条件**，采用 `natural_en` 与 `emotional_en`，每种措辞分别统计；配对分析按 `task_id` 关联。
- **LangMap：1,000 个独立任务**，region 330、instance 368、room 302；单纯 object 类别任务留在简单组。seed=20261105，固定 SHA256 排序。
- 清单、源文件哈希、数量回执：`configs/benchmarks/demand_2000/`。`langmap_runtime.json` 可直接传给已有 `--episode_manifest`，同时加载全部 LangMap annotations（9,314 条），设置 `--dataset_episodes 9314 --eval_episodes 1000`。
- 构建入口：`scripts/prepare_demand_acceptance.py`。清单本身不证明场景、渲染或闭环已经验收。
- **据 IntentionNav 接入记录，Isaac 6.0.1 的单场景渲染验收已通过；STRIVE 任务/感知/运动闭环仍待实现与验收，当前没有导航指标。** 详见 `docs/intentionnav_integration.md`。
