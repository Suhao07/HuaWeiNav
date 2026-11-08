# VLN Benchmark Provider 设计文档

本文档说明 VLN 当前的 benchmark 抽离层。目标不是改变导航策略，而是把
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

因此 provider 不会改变 VLN 的主导航链：

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
VLN Agent -> 只决定导航行为
Habitat Env -> 只决定 benchmark metrics
```

这能避免 HM3D-OVON、Gibson、instruction mode 和普通 HM3D ObjectNav 在同一个脚本中
互相污染。

## 7. 冻结身份与当前入口

从零部署统一见 [README](../README.md)，HM3D 命令与批次区间语义见 [评测操作指南](hm3d_docker_benchmark_guide.md)。前文 OVON 的 `val_seen_instruction_balanced_3k` 是已有数据源示例，不是当前交付需求2,000的默认来源。

[episode_selection.py](../benchmark/episode_selection.py) 经公开接口选择任务，并在 reset 后核对完整身份。入口输出 `episode_identity.json`；正式配对以场景、原始 episode ID、起点、朝向、目标类别等完整身份为准，不能仅比较局部索引或不同场景中重复的 episode ID。

[IntentionNav](intentionnav_integration.md) 使用独立 Isaac 启动器，不通过 Habitat provider 冒充环境等价；六字段公开输入必须匹配官方任务才可评分。当前 [交付验收](acceptance_4500.md) 是简单2,000 + IntentionNav需求2,000，困难500为子集；GOAT / LangMap 保留研究入口但不额外计入。

## 运动证据与独立评分

GOAT 语言派生任务在实际运行时通过 [HabitatEvaluationTrace](../benchmark/habitat_evaluation.py) 冻结：

```text
episode-*/evaluation/reference.json
episode-*/evaluation/runtime.navmesh
episode-*/evaluation/trajectory.jsonl
```

轨迹记录由运动所有者执行的实际 `env.step` 前后位姿，而不是模型建议路径。评分器重载冻结 NavMesh，核对其哈希、轨迹连续性、目标视点与终端 STOP，并记录参考文件及轨迹文件哈希：

```bash
python -m benchmark.habitat_evaluation /absolute/path/to/episode-directory
```

需要具有 Habitat-Sim 的环境；纯 Python schema 测试不能代替该评分。成功要求执行 STOP 且目标视点测地距离严格小于0.25m，SPL 根据实际路程计算。缺文件、缺终端 STOP 或缺可用参考路径时返回 unavailable；协议/哈希不符或轨迹不连续时抛出校验错误，两者都不伪造0分，也不使用内部 `instruction_success` 替代。

`goat_language_single_parent_start_v1` 从父 episode 原始起点独立执行一个语言子任务，不是官方多模态连续 GOAT。连续任务另由 [audit_goat_episode.py](../scripts/audit_goat_episode.py) 审计，需要上游源码、原始 shard、场景/传感器/动作及图像回执，不能与上述派生集互换。

[导航接受审计](../scripts/audit_navigation_acceptance.py) 使用 GT 作离线复核，不进入在线策略上下文。分别报告模型/任务接受、真实运动终态、官方评分和工程退出；新评分实现不等于已经完成全量 benchmark。
