# 4,000 条评测条件交付验收集

本页维护2026-11-08确认的交付范围。文件名与配置名保留 `acceptance_4500` 是历史命名，**当前不同条件总数为4,000，不是4,500**。筛选过程及旧 GOAT / LangMap 清单说明见 [历史快照](archive/acceptance_scope_20261108.md)，不作为当前执行清单。

## 1. 分组与验收目标

以下 SR/SPL 是验收目标，不是实测结果；实时运行进度以独立批次回执为准。

| 分组 | 条件数 | SR目标 | SPL目标 | 计数方式 |
|---|---:|---:|---:|---|
| 简单物体导航 | 2,000 | 0.70 | 0.29 | 已有 HM3D train 候选包；正式身份选择待确认，不能重复 val 补数 |
| 需求指令导航 | 2,000 | 0.51 | 0.24 | IntentionNav 500个几何任务 × 4种原始英文表述 |
| 困难指令子集 | 500 | 0.31 | 0.17 | 需求2,000的子集，单独报告、不重复计权 |
| 不同评测条件总计 | 4,000 | 0.57 | 0.25 | 仅简单2,000 + 需求2,000 |

四种表述为 `formal_en`、`natural_en`、`casual_en`、`emotional_en`。同一任务的四种表述共享场景、起点和目标，不是四个独立几何任务；同时报告几何任务数、条件数和按风格分组结果。

目标配置见 [acceptance_4500.json](../configs/benchmarks/acceptance_4500.json)。旧 `configs/benchmarks/demand_2000/` 为 IntentionNav 1,000 + LangMap 1,000 的历史清单，不能因目录名相同就用作当前交付输入。GOAT 与 LangMap 的代码、数据和研究结果保留，但退出当前交付条件。

简单2,000的候选制作、去重、类别/场景抽样和任务包哈希见 [HM3D制作步骤](hm3d_docker_benchmark_guide.md#41-制作简单2000候选包)。现有 train 候选包已通过包哈希与2,000个唯一身份核验，但正式交付配置仍待确认简单组来源，不能将候选包或历史 train 运行自动认定为完整验收。

## 2. 需求2000与困难500的制作

### 2.1 来源、标注与筛选规则

需求组采用官方原始64类别版本的 `selected_500_intents.jsonl` 和 `episodes.jsonl`，按 `selection_id` 绑定同一任务与场景。保留原起点、朝向、目标及四种英文原文，不混入 reviewed-v3 的65类别版本。500个几何任务分布于176场景，展开为2,000个任务-风格条件，实际有1,994种不同指令文本；这里的2,000是条件数，不是独立几何任务数或不同句子数。

每个条件的标注包含 `condition_id`、`instruction_sha256`、`constraints` 和 `annotation_call_id`；每个约束包含 `kind`、原文精确子串 `quote` 与 `requirement`。必须覆盖全部2,000条件并匹配原文哈希，拒绝无效引用与重复要求；标注不能改写任务真值，也不传给运行策略。

[prepare_demand_acceptance.py](../scripts/prepare_demand_acceptance.py) 只消费已准备的标注，完成校验、筛选与身份冻结，**不调用模型生成新指令，不运行导航或生成成绩**。令 n 为独立条件数、r 为关系条件数、t 为条件种类数：

- 合格条件要求 `n >= 2 or r >= 1`；关系类型为 spatial、temporal、conditional、exclusion、reference。
- 固定 `seed=20261108`，按 `(-n, -r, -t, sha256(f'{seed}:{condition_id}'))` 排序，取前500条；不足500时报错，不填充简单条目。
- 固定回归10条取困难排序中最先出现的10个不同 `selection_id`，不按场景下载进度重新选取。
- 句长、情绪措辞、历史 difficulty 标签、资产可用性、模型失败及导航成绩均不参与困难选择。

当前版本对全部2,000条件进行了 AI 辅助离线复核，修正1,268条标注及25条无效引用；得到706条合格条件，冻结困难500条，覆盖 **303个几何任务、152个场景**。这是标注复核和确定性筛选，不是独立人工语义验收或官方难度真值；详见 [全量复核回执](../../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/annotation-reviewed/full_review_receipt.json)。

### 2.2 当前冻结输入与产物

当前证据根目录为仓库同级 `STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/`；清单统一使用其中的 **`manifests-full-review/`**。[交付配置](../configs/benchmarks/acceptance_4500.json) 的清单路径相对 STRIVE 根目录解释。原 `manifests/`、`manifests-final/` 和历史快照保留，但不再作为当前困难500或回归10的输入，不能覆盖历史批次来替换版本。

| 输入 | SHA256 |
|---|---|
| [官方指令 metadata](../../IntentionNav-deploy-105/metadata/selected_500_intents.jsonl) | `fcf28886c8c853f9e3f9158dc420d63090a6fec58c44f094e54f9657e2b22b27` |
| [官方 episodes](../../IntentionNav-deploy-105/metadata/episodes.jsonl) | `b9cd9699b90f6275e631c47d447abf52ee48c525d95668851948dafadb29a9bc` |
| [全量复核标注](../../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/annotation-reviewed/annotations_full_review.jsonl) | `bdd5b0bf9b23cf9a180eb063adf5ab94d08ce9aa72cb80a36d7f09cac815e03a` |

| 产物 | 含义 | SHA256 |
|---|---|---|
| [demand_2000.jsonl](../../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/manifests-full-review/demand_2000.jsonl) | 500任务×4表述，保留原文与身份 | `ca07522f36e4b365ff54206ad86578fcbbb7f12a225561dad609170c9095dad4` |
| [hard_500.jsonl](../../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/manifests-full-review/hard_500.jsonl) | 706条合格条件中确定性选取500条 | `d1e08439497f437176ca512b234f8d7a5d83dc172d3e2ba139dc8a23c3c2a3f2` |
| [regression_10.jsonl](../../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/manifests-full-review/regression_10.jsonl) | 困难子集中10个不同几何任务 | `acfd387db06fe83db7386886e60f43008756ae003bbcba71df49859da041ca96` |
| [selection_receipt.json](../../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z/manifests-full-review/selection_receipt.json) | 输入/输出哈希、计数、选择规则及种子 | `f6a4efc2a31ea51ff2ea0abdc3f4c535886ff6e2748b2edd31d61767e103cfc8` |

### 2.3 复现与制作验收

在 STRIVE 根目录执行，先准备上述同级 metadata、全量复核标注和冻结证据。以下命令复用已审核标注，不重新生成标注或调用模型：

```bash
BATCH=../STRIVE-deploy-105/evidence/intentionnav-complex-20261108T0517Z
OUTPUT=outputs/acceptance/intentionnav-full-review-rebuild
python scripts/prepare_demand_acceptance.py \
  --intention-metadata ../IntentionNav-deploy-105/metadata/selected_500_intents.jsonl \
  --episodes ../IntentionNav-deploy-105/metadata/episodes.jsonl \
  --annotations "$BATCH/annotation-reviewed/annotations_full_review.jsonl" \
  --output "$OUTPUT" \
  --seed 20261108 || exit 1
for name in demand_2000.jsonl hard_500.jsonl regression_10.jsonl; do
  cmp "$OUTPUT/$name" "$BATCH/manifests-full-review/$name" || exit 1
done
```

输出目录必须不存在，不能覆盖已冻结批次。三份输入均为每行一个对象的 JSONL，字段校验见 [回归测试](../tests/test_demand_acceptance.py)。当前固定10条采用默认确定性选择，不传 `--regression-condition-ids`；另行指定10条属于新版本，不能沿用这里的回执或哈希。

`selection_receipt.json` 记录输入的绝对路径：在相同原始路径复现时，四份产物都应逐字节一致；迁移目录后，三份清单仍须一致，而回执应比较 `source_sha256`、`manifest_sha256`、`counts`、`selected10` 和筛选规则，不能仅因路径变化认定清单发生改变。

2026-11-08制作核验已从固定输入重现四份产物，逐字节一致；来源、产物、复核分片哈希及2,000条复核溯源覆盖通过，45项离线测试通过。**这只验收制作与冻结的技术可复现性**：不等于人工语义真值、176场景资产完整、模拟器验收通过或导航SR/SPL达标，后续各阶段仍须独立回执。

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
