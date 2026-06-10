# STRIVE Benchmark Provider 设计文档

本文档说明 STRIVE 当前的 benchmark 抽离层。目标不是改变导航策略，而是把
Habitat 数据集选择、单 episode 过滤、success distance 和 provenance 从
`objnav_benchmark_with_process_obs.py` 中抽出来，形成可复现、可扩展的输入接口。

## 1. 设计原则

Benchmark provider 只负责：

```text
benchmark args
-> BenchmarkSpec
-> Habitat config
-> Habitat Env
```

它不负责：

```text
instruction parsing
object detector prompt
room / viewpoint planning
target verification
stop decision
```

因此 provider 不会改变 STRIVE 的主导航链：

```text
observation -> mapper/agent -> selected viewpoint/action -> Habitat metrics
```

## 2. 核心数据结构

`benchmark/contracts.py` 定义 `BenchmarkSpec`：

```text
BenchmarkSpec:
  benchmark              # hm3d_objectnav / hm3d_ovon / gibson_objectnav ...
  split                  # 显式 benchmark split
  dataset_path           # 本次真正传给 Habitat 的 dataset json.gz
  dataset_root           # split root 或 data root
  scene_id
  object_category
  episode_rank
  source_file            # 如果 materialize 单 episode，则记录原文件
  source_episode_id
  filtered_dataset_path
  success_distance
  provenance
```

这个结构是实验复现的最小事实源。日志目录会写入：

```text
logs/<save_dir>/benchmark_spec.json
```

## 3. HM3D-OVON Provider

推荐正式实验使用显式 split：

```bash
python objnav_benchmark_with_process_obs.py \
  --benchmark hm3d_ovon \
  --benchmark_split val_seen_instruction_balanced_3k \
  --dataset_root /path/to/CogNav_ObjNav/data/datasets/objectnav/hm3d_ovon/v1 \
  --scene_id wcojb4TFT35 \
  --object_category tv_monitor \
  --episode_rank 0
```

provider 只会在该 split 下查找：

```text
{dataset_root}/{split}/content/{scene_id}.json.gz
{dataset_root}/{split}/{split}.json.gz
```

这样可以避免旧逻辑在多个 OVON/instruction/custom HM3D split 之间隐式搜索，导致
benchmark 来源不可控。

### Auto 模式

`--benchmark auto` 仍保留旧单场景调试行为：

```text
scene_id + object_category -> 按历史路径优先级找一个匹配 episode
```

这个模式只用于 smoke test。正式 ablation 不建议使用，因为它的 split 来源是
兼容逻辑决定的，而不是用户显式指定的。

## 4. Success Distance

普通 HM3D ObjectNav 默认：

```text
success_distance = 1.0
```

HM3D-OVON provider 默认：

```text
success_distance = 1.5
```

原因是 CogNav 生成的 OVON ObjectNav config 使用 1.5m。provider 会把该值显式传给
`hm3d_config()`，避免被普通 HM3D 默认值覆盖。若需要对齐其他设置，可以显式传：

```bash
--success_distance 1.0
```

## 5. Gibson Provider

当前提供两个 provider 名称：

```text
gibson_objectnav
gibson_custom
```

`gibson_objectnav` 只解析标准 Habitat ObjectNav Gibson dataset：

```text
datasets/objectnav/gibson/v1.1/{split}/{split}.json.gz
```

`gibson_custom` 是 CogNav custom Gibson wrapper 的显式迁移边界。CogNav custom Gibson
依赖：

```text
*_episodes.json.gz
*_info.pbz2
semantic map
custom goal index / metric wrapper
```

这些逻辑不能混入 HM3D/OVON provider。后续迁移应单独实现 custom env wrapper，而不是
让 HM3D provider 继承 Gibson 的 metric 逻辑。

## 6. 为什么要抽离

旧入口脚本中存在三类耦合：

```text
dataset path fallback
scene/object episode filtering
Habitat config success distance
```

这些和导航策略无关，但会直接影响 benchmark 可复现性。抽离之后：

```text
BenchmarkProvider -> 只决定数据与 config
InstructionAdapter -> 只决定自然语言目标结构化
STRIVE Agent -> 只决定导航行为
Habitat Env -> 只决定 benchmark metrics
```

这能避免 HM3D-OVON、Gibson、instruction mode 和普通 HM3D ObjectNav 在同一个脚本中
互相污染。

## 7. 后续 TODO

1. 将 Docker 文档中的 HM3D-OVON 命令改为显式 `--benchmark hm3d_ovon`。
2. 给 `gibson_custom` 接入 CogNav 的 semantic-map episode wrapper。
3. 将 provider provenance 写入最终 CSV 之外的聚合结果。
4. 增加 provider 单测：split 路径解析、单 episode materialize、success distance。
