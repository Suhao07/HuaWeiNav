# HM3D Docker 评测操作指南

从零安装、数据和权重准备、两层 Docker 构建统一见 [交付 README](../README.md#仿真器构建流程)。本文只维护评测输入、运行命令、结果与排障，不再维护第二套 CogNav 基础镜像安装流程。模型先按 [LVLM 部署指南](lvlm_server_deployment.md) 验证。

## 1. 环境与路径

命令在 STRIVE 根目录执行。`STRIVE_DATA_ROOT` / `HM3D_DATA_ROOT` 优先于 `configs/strive_weights.yaml`；迁移时不要依赖开发机的 CogNav 绝对目录。容器挂载为：

| 宿主内容 | 容器路径 | 约束 |
|---|---|---|
| STRIVE 源码及输出 | `/workspace/STRIVE` | 正式运行应使用冻结副本 |
| 数据根目录 | `/workspace/data` | 只读 |
| SAM / GroundingDINO 权重 | `/weights` 下对应文件 | 只读 |

`--dataset_path` 和 `--episode_manifest` 由容器内 Python 读取，必须使用容器可见路径。数据放在挂载之外时，仅设置宿主绝对路径不会使容器看到它。

```bash
bash docker/preflight.sh
```

成功的 preflight 只验证依赖，不证明模型服务、指令理解或导航成功。

## 2. 单例 smoke 与模式选择

配置模型变量和私有密钥后：

```bash
bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_objectnav \
  --eval_episodes 1 --start_episode 0 \
  --decision_owner baseline_guided \
  --agent_max_model_calls 100 --max_steps 500 \
  --vlm self_hosted --save_dir hm3d_single_smoke
```

- `baseline_guided` 为默认：领域探索/接近策略与按需语义选择共用运动所有者。
- `legacy` 为历史路径；`agent` 为自由工具决策对照。报告必须注明模式，不能把不同策略的结果解释为纯结构消融。
- 初始语言编译与 Agent 决策共享 `agent_max_model_calls`；感知与验证调用另列。
- `--max_steps` 是环境步预算；达到预算、模型预算耗尽、工程错误和验证完成是不同终止原因。
- `LLM_OFFLINE=1` / `STRIVE_LLM_FALLBACK=1` 不属于正式评测配置；不得用离线默认值绕过严格完成条件。

单场景调试可用：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 --object_category tv --episode_rank 0 \
  --decision_owner baseline_guided --vlm self_hosted \
  --save_dir hm3d_scene_smoke
```

该入口筛选并物化单 episode，适合调试；正式数据来源应通过显式 benchmark / split 固定，见 [Benchmark Provider](benchmark_providers.md)。

## 3. 自然语言与数据集目标

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 --object_category tv \
  --enable_instruction_adapter \
  --custom_instruction "find the television in the scene" \
  --decision_owner baseline_guided --vlm self_hosted \
  --save_dir hm3d_instruction_smoke
```

不开 adapter 的 ObjectNav 按数据集目标确定性编译。非 legacy 的自然语言入口使用有界初始解释，保存原文和编译诊断，不在失败后悄悄改成数据集类别。`rules` 是历史数据集兼容路径，不是通用自然语言规则解析器。任务义务、动态检索词和当前验证的职责见 [指令合同](instruction_adapter.md)。

## 4. 固定清单与批量重跑

标准 HM3D val 的规模不足以凭 `--eval_episodes 2000` 自动产生2,000个独立任务。需先准备具有足够独立身份的实际 split / 任务包；不要通过重复采样补数。

已准备好任务包与身份清单后，示例：

```bash
bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_objectnav \
  --dataset_path /workspace/data/datasets/hm3d_simple_2000/tasks.json.gz \
  --episode_manifest /workspace/data/datasets/hm3d_simple_2000/manifest.json \
  --dataset_episodes 2000 --eval_episodes 2000 --start_episode 0 \
  --decision_owner baseline_guided --agent_max_model_calls 100 \
  --max_steps 500 --vlm self_hosted --save_dir hm3d_simple_2000_new
```

上述路径是部署示例，不会自动生成或下载数据。清单是包含 `runtime_episodes` 数组的 JSON，每项使用 [episode_identity](../benchmark/episode_selection.py) 的完整身份；不要将需求筛选的 JSONL 清单直接传给这个参数。reset 后入口保存并核对场景、原始 episode、起点、朝向和目标类别。

**区间语义：**入口循环是 `range(start_episode, eval_episodes)`，后者是结束索引而非“从起点再跑多少例”。例如从100跑到199应设置 `--start_episode 100 --eval_episodes 200`，并固定原始 `--dataset_episodes` 与同一清单。

正式比较要求：

1. 冻结完整源码（含需要交付的未提交文件时，另存清单）、镜像 ID、数据/权重哈希、模型版本、参数与种子。
2. 新旧运行目录分离；完整重跑2,000条不等于只补跑旧批次缺失条目。
3. 优先安排旧批次已有结果的身份以尽早配对；其余任务仍保留在总清单，不重复、不遗漏。
4. 并发 worker 领取不相交的身份集合，保存每例命令、退出回执、完整身份、轨迹和评分。
5. “文件已生成”“进程已结束”“评分可用”分别核验。先报告完成数/异常数/待完成数，再计算具有有效评分的子集指标并明确分母；最终交付总指标不得通过剔除工程失败美化。
6. 旧代码若跨多个版本，按版本分层；配对只比较身份相同且双方评分可用的样本。并发耗时不直接作为延迟提升证据。

当前交付的分组和困难子集定义见 [验收口径](acceptance_4500.md)，不将实时进度数字写进部署说明。

## 5. 运行产物与评分

输出默认在 `logs/<save_dir>/`：

| 产物 | 用途 |
|---|---|
| `run_manifest.json`、`benchmark_spec.json` | 运行参数、数据来源与成功阈值 |
| `metrics.csv` | Habitat 指标、内部验证与模型调用统计 |
| `episode-*/episode_identity.json` | reset 后实际任务身份 |
| `episode-*/initial_interpretation.jsonl` | 初始语言解释及有界修正（发生时） |
| `episode-*/instruction_adapter/` | 任务合同、兼容投影与进度诊断 |
| `episode-*/harness/` | 决策、工具、终止回执 |
| `episode-*/lvlm_calls/` | 实际模型请求和响应追溯 |
| `episode-*/final_verifier/`、`evidence_blobs/` | 验证结果与归档图像 |
| `episode-*/rgb/`、视频、`frontier_map/` | 观测和路径诊断 |

`success` / `spl` 是外部 evaluator 结果；`instruction_success`、`final_stop_success` 是内部任务/验证结果，不能相互替代。`accepted_distance_to_target` 是接受实例的证据距离，不等同于 GT `distance_to_goal`。`lvlm_call_count_by_type` 记录调用与缓存命中；`lvml_call_count_by_type` 是历史拼写别名。

GOAT 语言独立评分另见 [运动证据与评分](benchmark_providers.md#运动证据与独立评分)，普通 ObjectNav 不自动生成那套证据。模型请求、图像及响应可能含敏感数据，应限制输出目录访问，不提交 Git 或复制进文档。

每次使用新 `save_dir`。`--clean_save_dir` 会删除旧目录内容，不用于正式批次的续跑或配对复核。

## 6. 常见问题

- **Docker socket / GPU 权限**：由部署者确认 Docker 与 NVIDIA runtime 权限，不改业务代码绕过。
- **缺权重**：检查 YAML 与环境覆盖；确需下载时显式使用 `STRIVE_DOWNLOAD_WEIGHTS=1`，不要在每个 worker 中重复下载。
- **模型401 / 错误模型名**：检查变量是否存在，以及模型名与 endpoint 是否匹配，不打印密钥。

```bash
python - <<'PY'
import os
names = ("ARK_API_KEY", "LLM_MODEL", "LLM_API_BASE_URL",
         "VLN_LVLM_API_KEY", "VLN_LVLM_MODEL", "VLN_LVLM_BASE_URL")
print({name: bool(os.getenv(name)) for name in names})
PY
```

- **连接正常但结构化失败**：分别排查传输/服务端错误和实际非法响应，保留原始原因。不能增加静默字段过滤或自动接受来让批次“跑完”。
- **导航进程异常**：保留退出状态和最后一次动作/请求，不把不完整记录伪装为正常预算终止。

进入同一镜像调试：

```bash
bash docker/run_hm3d_baseline.sh bash
```
