# 4,000 条评测条件交付验收集

本页维护2026-11-08确认的交付范围。文件名与配置名保留 `acceptance_4500` 是历史命名，**当前不同条件总数为4,000，不是4,500**。筛选过程及旧 GOAT / LangMap 清单说明见 [历史快照](archive/acceptance_scope_20261108.md)，不作为当前执行清单。

## 1. 分组与验收目标

以下 SR/SPL 是验收目标，不是实测结果；实时运行进度以独立批次回执为准。

| 分组 | 条件数 | SR目标 | SPL目标 | 计数方式 |
|---|---:|---:|---:|---|
| 简单物体导航 | 2,000 | 0.70 | 0.29 | 冻结 HM3D 独立身份清单；标准 val 仅1,000例，不能重复补数 |
| 需求指令导航 | 2,000 | 0.51 | 0.24 | IntentionNav 500个几何任务 × 4种原始英文表述 |
| 困难指令子集 | 500 | 0.31 | 0.17 | 需求2,000的子集，单独报告、不重复计权 |
| 不同评测条件总计 | 4,000 | 0.57 | 0.25 | 仅简单2,000 + 需求2,000 |

四种表述为 `formal_en`、`natural_en`、`casual_en`、`emotional_en`。同一任务的四种表述共享场景、起点和目标，不是四个独立几何任务；同时报告几何任务数、条件数和按风格分组结果。

目标配置见 [acceptance_4500.json](../configs/benchmarks/acceptance_4500.json)。旧 `configs/benchmarks/demand_2000/` 为 IntentionNav 1,000 + LangMap 1,000 的历史清单，不能因目录名相同就用作当前交付输入。GOAT 与 LangMap 的代码、数据和研究结果保留，但退出当前交付条件。

## 2. 需求清单冻结

[prepare_demand_acceptance.py](../scripts/prepare_demand_acceptance.py) 只做筛选与身份冻结，不运行导航或生成成绩。准备官方 metadata、500个任务与每条原始表述的复杂条件标注后：

```bash
python scripts/prepare_demand_acceptance.py \
  --intention-metadata /path/to/official-metadata.jsonl \
  --episodes /path/to/episodes.jsonl \
  --annotations /path/to/constraint-annotations.jsonl \
  --output outputs/acceptance/demand-new \
  --seed 20261108
```

三份输入均为每行一个对象的 JSONL，字段以脚本校验和 [回归样例](../tests/test_demand_acceptance.py) 为准。输出目录必须不存在，不能覆盖已冻结批次。需要指定已核对的固定10条时，传 `--regression-condition-ids` 后跟恰好10个实际条件 ID。

| 产物 | 含义 |
|---|---|
| `demand_2000.jsonl` | 500任务×4表述；保留原文、身份、来源哈希 |
| `hard_500.jsonl` | 满足复杂条件门槛的确定性困难子集 |
| `regression_10.jsonl` | 困难子集中10个不同几何任务的固定回归条件 |
| `selection_receipt.json` | 输入/输出哈希、计数、选择协议及种子 |

标注必须精确引用原文并匹配源文本哈希；不能根据句长、情绪措辞、资产是否齐全或导航是否失败定义困难。令 n 为独立条件数、r 为关系条件数、t 为条件种类数：候选要求 n≥2 或 r≥1，再按 n、r、t 降序及带种子的 SHA256 排序。符合条件不足500时失败，不填充简单条目。

已冻结困难集覆盖332个几何任务、160个场景；它是模型辅助筛选结果，仅部分条目经过直接复核，不宣称500条均经独立人工审阅或属于官方难度真值。具体批次路径、原始响应、87条直接核对及修正过程保留在历史快照与受控证据目录，不复制到正式配置。

## 3. 从冻结到执行

1. 简单组按 [HM3D 操作指南](hm3d_docker_benchmark_guide.md) 加载固定身份清单；不要将需求 JSONL 直接作为 Habitat `--episode_manifest`。
2. 需求组按 [IntentionNav 部署指南](intentionnav_integration.md)，从实际官方任务/风格生成六字段公开输入；策略与模拟器使用同一份冻结文件。
3. 验证场景/资产、真实渲染与一例完整导航后，先运行已授权的固定10条。新一轮全量评测需要明确授权，不因筛选成功而自动启动。
4. 保存源码、镜像 ID、模型/权重版本、原文、起点、场景、轨迹与评分证据。不同版本的补跑不能拼成同一个代码版本结果。
5. 不排除失败样本；工程异常、预算终止、任务验证与独立评分分列。公开输入不匹配官方身份时不提供官方 SR/SPL。

## 4. 汇总口径

\[
\mathrm{SR}_{all}=\frac{2000\mathrm{SR}_{simple}+2000\mathrm{SR}_{demand}}{4000},\qquad
\mathrm{SPL}_{all}=\frac{2000\mathrm{SPL}_{simple}+2000\mathrm{SPL}_{demand}}{4000}.
\]

困难500已包含在需求2,000中，只检查其单独目标，不额外加权。简单/需求分组目标的加权值为0.605/0.265；原总体验收目标0.57/0.25保持不变，不能将算术结果擅自改称新目标。

每个 benchmark 保留自身成功阈值与评分协议；项目混合指标不等同于任何原论文指标。缺评分标记 unavailable / null，并报告覆盖率和原因，不伪造0分，也不剔除后声称总体通过。全部要求的结果与证据齐全后，才能发布完整交付结论。
